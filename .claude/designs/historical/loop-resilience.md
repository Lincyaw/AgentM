**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](../pluggable-architecture.md) and
[extension-as-scenario.md](../extension-as-scenario.md).

---

# Design: Loop Resilience — Five Mechanisms for Robust Agent Loops

**Status**: DRAFT
**Created**: 2026-03-31
**Last Updated**: 2026-03-31
---

## Overview

Five mechanisms that harden `SimpleAgentLoop` against context blowup, runaway costs, stale constraints, and serial bottlenecks. All are implemented as middleware or small loop-level changes — zero modification to the core `SimpleAgentLoop.stream()` control flow.

| # | Mechanism | Layer | Touches |
|---|-----------|-------|---------|
| 1 | Tool Result Budget | Middleware (`on_tool_call`) | New: `ToolResultBudgetMiddleware` |
| 2 | Tool Concurrency Partitioning | Loop (`stream()`) + `Tool` | Edit: `simple.py`, `tool.py` |
| 3 | Critical System Reminder | Middleware (`on_llm_start`) | New: `SystemReminderMiddleware` |
| 4 | Micro-Compact | Middleware (`on_llm_start`) | New: `MicroCompactMiddleware` |
| 5 | Cost Budget | Middleware (`on_llm_end`) + `LoopContext` | New: `CostBudgetMiddleware`, edit: `types.py` |

---

## 1. Tool Result Budget

### Motivation

Workers (especially RCA scouts) execute SQL queries that can return arbitrarily large result sets. A single `duckdb_sql` call returning 500 rows × 20 columns can produce 50K+ characters, rapidly consuming the context window. Claude Code caps per-tool results at 50K chars and per-message aggregate at 200K chars, persisting oversized results to disk and giving the model a preview + file path.

The right handling depends on the scenario: RCA workers should refine their queries rather than read back raw data dumps, but code analysis agents may need the full content. Therefore the overflow strategy must be **configurable per scenario**.

### Design

A new `ToolResultBudgetMiddleware` that wraps tool execution in the `on_tool_call` hook:

```python
class OverflowStrategy(StrEnum):
    """What to do when a tool result exceeds the size limit."""
    TRUNCATE = "truncate"    # discard overflow, hint to refine query
    PERSIST = "persist"      # write full result to disk, return preview + path

@dataclass(frozen=True)
class ToolResultBudgetConfig:
    max_result_chars: int = 30_000        # per-tool-call limit
    max_aggregate_chars: int = 150_000    # per-message (all tool results in one turn)
    preview_chars: int = 2_000            # preview size when truncated/persisted
    overflow_strategy: OverflowStrategy = OverflowStrategy.TRUNCATE
    persist_dir: str = ""                 # directory for persisted results (PERSIST mode)
```

**Flow**:

```
on_tool_call(name, args, call_next, ctx):
    result = await call_next(name, args)
    if len(result) > config.max_result_chars:
        result = handle_overflow(result, config, ctx)
    return result
```

**TRUNCATE strategy** (default) — discard overflow, guide the agent to query more precisely:

```
<truncated_result>
Output too large (52,341 chars). Showing first 2,000 chars:

{preview}
...

[Remaining 50,341 chars truncated. Refine your query to reduce output size.]
</truncated_result>
```

**PERSIST strategy** — write full result to disk, return preview + file path:

```
<persisted_result>
Output too large (52,341 chars). Full output saved to: {filepath}

Preview (first 2,000 chars):
{preview}
...
</persisted_result>
```

When PERSIST is active, the middleware writes the full result to `{persist_dir}/{tool_call_id}.txt`. The agent needs a file-read tool to access the full content. The `persist_dir` is typically set to a session-scoped temp directory by `WorkerLoopFactory`.

**Scenario configuration example**:

```yaml
# RCA scenario — workers should write better SQL, not read back dumps
tool_result_budget:
  max_result_chars: 30000
  overflow_strategy: truncate

# Code analysis scenario — full file content may be needed
tool_result_budget:
  max_result_chars: 50000
  overflow_strategy: persist
  persist_dir: "${SESSION_DIR}/tool-results"
```

**Aggregate budget**: Tracked via a mutable counter on the middleware instance. After each tool result, the running total is checked. If the aggregate exceeds `max_aggregate_chars`, further results in the same turn are aggressively truncated to `preview_chars` regardless of the configured strategy (hard safety net).

### Why middleware, not loop-level

The `on_tool_call` hook wraps tool execution — the result passes through middleware before being appended to `messages`. This is the natural interception point. No `SimpleAgentLoop` changes needed.

### Interaction with existing middleware

- **BudgetMiddleware** (step/tool counts): Orthogonal — that tracks call counts, this tracks result sizes.
- **CompressionMiddleware**: This reduces input to compression, making LLM-based compression cheaper and less frequent.

---

## 2. Tool Concurrency Partitioning

### Motivation

`SimpleAgentLoop.stream()` currently runs all tool calls in a single `asyncio.gather()` batch when there are multiple calls. This is unsafe: a read-only `vault_read` and a state-mutating `memory_write` should not run concurrently. Claude Code solves this with `isConcurrencySafe(input)` per tool and a partition algorithm that groups consecutive safe tools into parallel batches.

### Design

**Step 1**: Add `concurrency_safe` field to `Tool`:

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolCallable
    readonly: bool = False
    concurrency_safe: bool = False   # NEW
```

`concurrency_safe = True` means: this tool has no side effects that would conflict with concurrent execution of other `concurrency_safe` tools. Read-only tools (`duckdb_sql`, `vault_read`, `vault_search`, `think`) are safe. Write tools (`vault_write`, `vault_edit`, `memory_write`) are not.

**Step 2**: Partition logic in `SimpleAgentLoop.stream()`, replacing the current all-or-nothing gather:

```python
def _partition_tool_calls(
    tool_calls: list[dict[str, Any]],
    tools_dict: dict[str, Tool],
) -> list[tuple[bool, list[dict[str, Any]]]]:
    """Partition tool calls into (is_concurrent, batch) groups.

    Consecutive concurrency_safe tools are batched together.
    Non-safe tools are isolated into single-item batches.
    """
    batches: list[tuple[bool, list[dict[str, Any]]]] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        tool = tools_dict.get(name)
        safe = tool.concurrency_safe if tool else False
        if safe and batches and batches[-1][0]:
            batches[-1][1].append(tc)
        else:
            batches.append((safe, [tc]))
    return batches
```

**Step 3**: Execute batches in order:

```python
for is_concurrent, batch in _partition_tool_calls(tool_calls, self._tools):
    if is_concurrent and len(batch) > 1:
        results = await asyncio.gather(*[_run_one(tc) for tc in batch])
    else:
        results = [await _run_one(tc) for tc in batch]
    # append results to messages...
```

### Concurrency limit

Environment variable `AGENTM_MAX_TOOL_CONCURRENCY` (default: 8). Applied via `asyncio.Semaphore` inside `_run_one`.

### What changes in `simple.py`

The current two-branch `if len(tool_calls) == 1: ... else: asyncio.gather(...)` block is replaced with the partition-based loop above. The event emission (tool_start/tool_end) stays the same.

---

## 3. Critical System Reminder

### Motivation

In long conversations (10+ turns), the system prompt's constraints drift out of the model's attention window. Claude Code addresses this with `criticalSystemReminder_EXPERIMENTAL`, injecting constraint text into every tool result as an attachment. AgentM can achieve the same with a simpler approach: periodic re-injection into the system message via `on_llm_start`.

### Design

A new `SystemReminderMiddleware`:

```python
@dataclass(frozen=True)
class SystemReminderConfig:
    reminder_text: str                  # the constraint block to re-inject
    interval: int = 5                   # re-inject every N steps
    start_after: int = 5                # don't inject in the first N steps
```

**Flow**:

```
on_llm_start(messages, ctx):
    if ctx.step >= config.start_after and ctx.step % config.interval == 0:
        return inject_into_system_message(messages, config.reminder_text)
    return messages
```

### Content

The `reminder_text` is scenario-specific, set by `ScenarioWiring`. Examples:

- **RCA worker (scout)**: `"<system-reminder>\nYou are in DATA-ONLY MODE. Report measurements, do not suggest root causes.\n</system-reminder>"`
- **RCA worker (verify)**: `"<system-reminder>\nYou are in DISPROOF MODE. Find evidence AGAINST the hypothesis.\n</system-reminder>"`
- **READONLY agent**: Already handled by `PermissionMiddleware`. But if both are active, `SystemReminderMiddleware` runs in addition (belt + suspenders).

### Why `on_llm_start` and not tool-result attachment

Tool-result attachment (Claude Code's approach) requires modifying the message construction in the loop. `on_llm_start` is already the established injection point for `MemoryMiddleware`, `PermissionMiddleware`, `SkillMiddleware`, and `BudgetMiddleware`. Using the same pattern keeps the architecture consistent.

### Interaction with `inject_into_system_message`

Reuses the existing shared utility. The reminder is appended to the system message, after any memory or permission blocks already injected by earlier middleware in the chain.

---

## 4. Micro-Compact (Lightweight Context Cleanup)

### Motivation

`CompressionMiddleware` uses an LLM call to summarize history — expensive and slow (~2-5s). For many situations, a simple rule-based cleanup is sufficient: old tool results that were already consumed by the model don't need to stay in full. Claude Code's microCompact clears specific tool results older than N turns, replacing them with `[Old tool result content cleared]`. This defers or avoids the expensive LLM-based compression entirely.

### Design

A new `MicroCompactMiddleware` that runs **before** `CompressionMiddleware` in the middleware chain:

```python
@dataclass(frozen=True)
class MicroCompactConfig:
    enabled: bool = True
    stale_after_steps: int = 6          # results older than N steps are stale
    compactable_tools: frozenset[str] = frozenset({
        "duckdb_sql", "vault_read", "vault_search",
    })
    cleared_message: str = "[Old tool result content cleared]"
```

**Flow**:

```
on_llm_start(messages, ctx):
    if ctx.step < config.stale_after_steps * 2:
        return messages  # not enough history to compact

    cutoff_step = ctx.step - config.stale_after_steps
    new_messages = []
    for msg in messages:
        if is_tool_result(msg) and came_from_compactable_tool(msg) and msg_step <= cutoff_step:
            new_messages.append(cleared_version(msg))
        else:
            new_messages.append(msg)
    return new_messages
```

### Step tracking

The middleware needs to know which step each tool result came from. Two options:

**Option A (preferred)**: Use message position as a proxy. Tool result messages are interspersed with assistant messages. Count assistant messages from the end to determine age. No metadata changes needed.

**Option B**: Tag tool result messages with a `_step` metadata field when they're created in `SimpleAgentLoop.stream()`. Requires a small loop change.

**Decision**: Option A. It avoids loop changes and is sufficient for the "N turns ago" heuristic. The middleware counts backward from the end of `messages` to identify which tool results are old enough to clear.

### Determining the source tool

Tool result messages in AgentM are `{"role": "tool", "content": "...", "tool_call_id": "..."}`. The tool name is not directly on the result message — it's on the preceding assistant message's `tool_calls` list. The middleware builds a `tool_call_id → tool_name` index from assistant messages before scanning.

### Interaction with CompressionMiddleware

`MicroCompactMiddleware` runs first (`on_llm_start` order = middleware list order). By clearing old tool results, it reduces the token count that `CompressionMiddleware` sees, potentially keeping it below the compression threshold entirely.

---

## 5. Cost Budget

### Motivation

`RunConfig.timeout` exists but is never enforced (known issue HIGH #4). There's no cost tracking at all. A runaway loop can burn through API credits without limit. Claude Code enforces `maxBudgetUsd` as a hard stop after every turn.

### Design

Two changes:

### 5.1 Usage tracking in LoopContext

Add accumulated usage to `LoopContext`:

```python
@dataclass(frozen=True)
class LoopContext:
    agent_id: str
    step: int
    max_steps: int | None
    tool_call_count: int
    metadata: JsonDict
    # NEW
    total_input_tokens: int = 0
    total_output_tokens: int = 0
```

`SimpleAgentLoop.stream()` extracts `usage` from the LLM response (via `response.usage_metadata` on LangChain models) and accumulates it when building `LoopContext` for the next step.

### 5.2 CostBudgetMiddleware

```python
@dataclass(frozen=True)
class CostBudgetConfig:
    max_cost_usd: float | None = None
    max_total_tokens: int | None = None
    # Per-model cost table (input $/1M tokens, output $/1M tokens)
    cost_table: dict[str, tuple[float, float]] = field(default_factory=dict)
    model_name: str = ""
```

**Flow**:

```
on_llm_end(response, ctx):
    usage = extract_usage(response)
    cost = calculate_cost(usage, config)
    accumulated_cost += cost
    if config.max_cost_usd and accumulated_cost > config.max_cost_usd:
        raise CostBudgetExceeded(accumulated_cost, config.max_cost_usd)
    if config.max_total_tokens and total_tokens > config.max_total_tokens:
        raise TokenBudgetExceeded(total_tokens, config.max_total_tokens)
    return response
```

### Exception handling in the loop

`SimpleAgentLoop.stream()` catches `CostBudgetExceeded` and `TokenBudgetExceeded` the same way it handles max-steps exhaustion: yield an error event, attempt output synthesis, and return a `FAILED` result with error message.

### 5.3 Timeout enforcement (piggyback)

While we're here, enforce `RunConfig.timeout`:

```python
# In SimpleAgentLoop.stream():
async with asyncio.timeout(config.timeout) if config.timeout else contextlib.nullcontext():
    # ... existing loop ...
```

`asyncio.TimeoutError` is caught at the outer level, producing a `FAILED` result with `"Timeout after {config.timeout}s"`.

---

## Middleware Ordering

The full middleware stack, from outermost to innermost:

```
1. PermissionMiddleware      — deny early (on_tool_call)
2. CostBudgetMiddleware      — stop on budget (on_llm_end)
3. SystemReminderMiddleware   — re-inject constraints (on_llm_start)
4. MemoryMiddleware          — inject memory (on_llm_start)
5. SkillMiddleware           — inject skills (on_llm_start)
6. MicroCompactMiddleware    — clear old results (on_llm_start)
7. CompressionMiddleware     — LLM-based summarization (on_llm_start)
8. BudgetMiddleware          — step/tool urgency (on_llm_start)
9. LoopDetectionMiddleware   — detect loops (on_llm_start)
10. DedupMiddleware          — dedup tool calls (on_tool_call)
11. ToolResultBudgetMiddleware — truncate large results (on_tool_call)
12. TrajectoryMiddleware     — record events (all hooks)
```

**Rationale for ordering**:
- Permission and cost checks are outermost — deny/stop before any other processing.
- System reminder, memory, skills inject content before micro-compact/compression — the injected content is part of the "current state" not subject to cleanup.
- MicroCompact runs before CompressionMiddleware — reduces token count, potentially avoiding the expensive LLM call.
- ToolResultBudget wraps tool execution — truncation happens before the result enters the message history.
- Trajectory is innermost — records the final state after all other middleware has processed.

---

## Interface Changes Summary

| File | Change |
|------|--------|
| `core/tool.py` | Add `concurrency_safe: bool = False` to `Tool` |
| `harness/types.py` | Add `total_input_tokens`, `total_output_tokens` to `LoopContext` |
| `harness/loops/simple.py` | Partition-based tool execution; usage tracking; timeout enforcement |
| `harness/middleware.py` | New: `ToolResultBudgetMiddleware`, `SystemReminderMiddleware`, `MicroCompactMiddleware`, `CostBudgetMiddleware` |
| `harness/worker_factory.py` | Wire new middleware into the stack |

---

## What Is NOT in Scope

- **File-read tool for PERSIST mode** — When `overflow_strategy=persist`, the agent needs a way to read back persisted files. This may require adding a generic file-read tool to the worker's toolset, which is outside this design's scope.
- **Session recovery / JSONL conversation persistence** — Valuable for orchestrator, but separate design scope (see `recovery_system` concept).
- **Hooks system** — Claude Code's 29-event hook system is a framework-level feature; AgentM's middleware already covers the needed extension points.
- **Deferred tool loading / ToolSearch** — AgentM's tool sets are small and fixed per agent; no need for lazy loading.
- **Denial tracking / auto-mode classifier** — These are interactive UI features; AgentM runs non-interactively.

---

## Related Concepts

- [agent-harness](agent-harness.md) — `SimpleAgentLoop` and middleware system
- [middleware_system](sdk-consistency.md) — Middleware ordering and composition
- [sub-agent](sub-agent.md) — Worker lifecycle and `WorkerLoopFactory`
- [tool-filter](tool-filter.md) — Tool filtering (orthogonal to concurrency)
- [permission-mode](permission-mode.md) — Permission enforcement (ordering dependency)
- [prompt-patterns](prompt-patterns.md) — System reminder content patterns

## Constraints and Decisions

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| Overflow strategy as config | Different scenarios have different needs: RCA prefers truncate (force better queries), code analysis prefers persist (full data needed) | Hardcode one strategy for all |
| `on_llm_start` for reminders, not tool-result attachment | Consistent with existing injection pattern (memory, permissions, skills) | Attach to each tool result message |
| Option A (position-based age) for micro-compact | Avoids loop changes; sufficient accuracy for "N turns ago" heuristic | Tag messages with step metadata |
| `concurrency_safe` as static field, not method | AgentM tools have fixed safety semantics; no input-dependent logic needed | `is_concurrency_safe(input)` method like Claude Code |
| Middleware-based cost check, not loop-level | Keeps `SimpleAgentLoop` minimal; budget policy is a cross-cutting concern | Check inside `stream()` directly |

## Open Questions

- [ ] Should `MicroCompactConfig.compactable_tools` be configurable per scenario, or is the default set sufficient?
- [ ] Should `CostBudgetMiddleware` attempt a graceful output synthesis before stopping, or hard-stop immediately?
- [ ] What's the right `max_result_chars` default for RCA scenarios? 30K is a guess — needs eval data.
- [ ] When using `overflow_strategy=persist`, should the SDK automatically add a generic `read_file` tool to the worker, or leave it to the scenario to provide one?