**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](pluggable-architecture.md) and
[extension-as-scenario.md](extension-as-scenario.md).

---

# Design: Permission Mode

**Status**: DRAFT
**Created**: 2026-03-31

---

## 1. Background & Motivation

### 1.1 Why an SDK Needs Permission Modes

AgentM is a multi-agent SDK framework where an orchestrator spawns worker agents, each equipped with tools that can read data, write files, execute commands, or call external APIs. Currently, there is no mechanism to constrain what an agent is allowed to do at runtime.

Real-world use cases demand different levels of agent autonomy:

| Use case | Required behavior |
|----------|------------------|
| Dry-run / preview | Agent plans actions but executes nothing destructive |
| Production with audit trail | Agent executes freely but every tool call is logged |
| Supervised execution | Human or parent agent must approve certain tool calls |
| Fully autonomous pipeline | No permission checks, maximum throughput |

Claude Code addresses this with five permission modes (`default`, `plan`, `acceptEdits`, `bypassPermissions`, `bubble`). However, Claude Code is a CLI product with interactive user confirmation flows. AgentM is an SDK — there is no interactive terminal. The permission model must be **programmatic, composable, and middleware-based**.

### 1.2 Design Goals

1. **Non-invasive** — Permission enforcement does not modify `SimpleAgentLoop` core logic. It is implemented entirely as a middleware (`on_tool_call` hook).
2. **Declarative** — Permission mode is declared per-agent at construction time, not scattered across tool implementations.
3. **Three-layer enforcement** — Prompt reinforcement + tool filtering + middleware interception. Defense in depth.
4. **Extensible** — Custom permission policies can be injected without modifying SDK internals.

---

## 2. Design

### 2.1 PermissionMode Enum

Four modes, adapted from Claude Code's five for SDK context:

| Mode | Behavior | Claude Code Equivalent |
|------|----------|----------------------|
| `DEFAULT` | Standard mode. All tools available. No special constraints. | `default` |
| `READONLY` | Agent can only call tools marked as readonly. Mutating tool calls are rejected. | `plan` |
| `SUPERVISED` | All tool calls permitted but every call is logged to an audit trail. | (no equivalent) |
| `UNRESTRICTED` | No permission checks. Middleware short-circuits immediately (no overhead). | `bypassPermissions` |

**Why not five modes?** Claude Code's `acceptEdits` is a UI concern (auto-accept file edits). `bubble` delegates permission to a parent agent — this is an orchestration concern handled by the existing `inject`/`abort` mechanism in `AgentRuntime`, not a permission mode.

### 2.2 Tool Readonly Annotation

The `Tool` dataclass gains a `readonly` attribute:

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolCallable
    readonly: bool = False  # NEW: True for tools that only read data
```

The `@tool` decorator and `tool_from_function` factory gain a corresponding `readonly` parameter:

```python
@tool(readonly=True)
async def search_logs(query: str) -> str: ...

@tool
async def delete_file(path: str) -> str: ...
```

**Design decision**: Annotation on Tool (not allowlist in policy). The tool author knows best whether their tool mutates state. A simple boolean is sufficient. The `PermissionPolicy` can override individual tools if needed (see below).

### 2.3 PermissionPolicy

```python
@dataclass(frozen=True)
class PermissionPolicy:
    """Immutable permission policy for an agent."""

    mode: PermissionMode = PermissionMode.DEFAULT

    # Per-tool overrides: tool_name -> allowed (True/False).
    # Takes precedence over the mode's default behavior.
    tool_overrides: dict[str, bool] = field(default_factory=dict)

    # Audit callback for SUPERVISED mode.
    # Signature: (tool_name, tool_args, result) -> None
    audit_fn: Callable[..., Awaitable[None]] | None = None

    def can_execute(self, tool_name: str, tool: Tool | None = None) -> bool:
        """Determine if a tool call is allowed.

        Resolution order:
        1. tool_overrides (explicit per-tool allow/deny)
        2. Mode-based rule:
           - DEFAULT / SUPERVISED / UNRESTRICTED -> always True
           - READONLY -> True only if tool.readonly is True
        """
```

**Key decisions:**
- Frozen dataclass — immutable after creation.
- `tool_overrides` allows escape hatches in any mode.
- `audit_fn` is optional. SUPERVISED mode without it falls back to structured logging.
- When `tool` is `None` (unknown tool), READONLY mode denies by default (safe fallback).

### 2.4 PermissionMiddleware

The enforcement mechanism is a standard `MiddlewareBase` subclass that hooks into `on_tool_call` and `on_llm_start`. `SimpleAgentLoop` requires zero changes.

**`on_tool_call` behavior:**

| Mode | Behavior |
|------|----------|
| `UNRESTRICTED` | Short-circuit to `call_next` immediately (no overhead) |
| `DEFAULT` | Pass through to `call_next` |
| `READONLY` | Check `can_execute()` → reject non-readonly tools with error message |
| `SUPERVISED` | Execute via `call_next`, then call `audit_fn` or log |

**`on_llm_start` behavior (READONLY only):**

Injects a permission constraint block into the system message:

```xml
<permission_constraint>
You are in READONLY mode. You may only use tools that read data.
Do NOT attempt to call any tool that modifies, creates, or deletes data.
If you need a write operation, describe what you would do instead.
</permission_constraint>
```

**Middleware ordering**: `PermissionMiddleware` should be the **first** middleware in the chain (outermost wrapper). Permission denial happens before any other middleware processes the call (before DedupMiddleware caches it, before TrajectoryMiddleware records it).

### 2.5 READONLY Mode: Defense in Depth

Two layers:

1. **Tool list filtering (build time)**: Non-readonly tools are excluded from the tools list passed to the LLM. The LLM never sees mutating tools in its function definitions.
2. **Middleware interception (runtime)**: `PermissionMiddleware.on_tool_call` rejects any non-readonly tool call that somehow reaches execution (hallucinated tool name, dynamically added tool).

Layer 1 is the primary defense. Layer 2 is the safety net.

### 2.6 SUPERVISED Mode: Audit Trail

All tools are available and executable. The middleware records every tool invocation after execution. The `audit_fn` callback allows SDK users to route audit records to their own backend (database, message queue, file).

**Note**: `TrajectoryMiddleware` already records tool calls. SUPERVISED audit is complementary — it is a **permission audit** (who was allowed to do what), while trajectory is an **execution trace** (what happened). They serve different purposes and both may coexist.

---

## 3. Integration

### 3.1 ScenarioWiring

`ScenarioWiring` gains optional permission fields:

```python
@dataclass
class ScenarioWiring:
    # ... existing fields ...
    orchestrator_permission: PermissionPolicy = field(default_factory=PermissionPolicy)
    worker_permission: PermissionPolicy = field(default_factory=PermissionPolicy)
```

Scenarios declare permission policies in their `setup()` return value.

### 3.2 WorkerLoopFactory

Reads `worker_permission` from `ScenarioWiring` and:
1. Filters the tool list if mode is READONLY (Layer 1).
2. Prepends `PermissionMiddleware` to the middleware stack (Layer 2).

### 3.3 Builder

`_build_orchestrator_loop` wires `PermissionMiddleware` for the orchestrator based on `ScenarioWiring.orchestrator_permission`.

---

## 4. Prompt Reinforcement — Three-Layer Defense

| Layer | Mechanism | When |
|-------|-----------|------|
| **Prompt** (soft) | `PermissionMiddleware.on_llm_start` injects `<permission_constraint>` | Every LLM call |
| **Tool filtering** (hard, build-time) | Non-readonly tools excluded from LLM's tool schema | Agent construction |
| **Middleware** (hard, runtime) | `PermissionMiddleware.on_tool_call` rejects disallowed calls | Every tool execution |

**Why three layers?**
- Prompt alone is unreliable — LLMs can ignore instructions.
- Tool filtering alone misses hallucinated tool names.
- Middleware alone means the LLM wastes a step attempting a denied call.
- Together: the LLM is told (prompt), cannot see (filter), and is blocked (middleware).

---

## 5. Impact Assessment

### New Files

| File | Content |
|------|---------|
| `agentm/harness/permission.py` | `PermissionMode`, `PermissionPolicy`, `PermissionMiddleware` |

### Modified Files

| File | Change |
|------|--------|
| `agentm/core/tool.py` | Add `readonly: bool = False` to `Tool` |
| `agentm/harness/tool.py` | Pass `readonly` through `@tool` decorator |
| `agentm/harness/scenario.py` | Add permission fields to `ScenarioWiring` |
| `agentm/harness/worker_factory.py` | Accept `PermissionPolicy`, filter tools, prepend middleware |
| `agentm/builder.py` | Wire `PermissionMiddleware` in `_build_orchestrator_loop` |

### No Breaking Changes

- `Tool.readonly` defaults to `False` — all existing tools work unchanged.
- `PermissionPolicy()` defaults to `DEFAULT` — no permission checks by default.
- `ScenarioWiring` new fields have defaults — existing scenarios unaffected.

### Design Documents Affected

| Document | Impact |
|----------|--------|
| [agent-harness.md](agent-harness.md) | Add PermissionMiddleware to middleware catalog |
| [sdk-consistency.md](sdk-consistency.md) | Note `readonly` addition to Tool dataclass |
| [sub-agent.md](sub-agent.md) | Document worker permission inheritance from ScenarioWiring |

---

## 6. Constraints & Decisions

| Decision | Rationale |
|----------|-----------|
| Middleware, not loop modification | Permission is a cross-cutting concern; fits `on_tool_call` hook naturally |
| Readonly annotation on Tool dataclass | Tool author declares intent; SDK enforces it |
| No interactive approval flow | AgentM is SDK, not CLI; no `input()` or terminal prompt |
| Audit vs. Trajectory separation | Authorization audit ≠ execution trace; both may coexist |
| Four modes (not five) | `acceptEdits` is UI-only; `bubble` is handled by orchestration |

---

## 7. Open Questions

1. Should READONLY mode return a descriptive error message or raise an exception? Current design: error string (LLM sees it and adjusts).
2. Should `PermissionPolicy` support a dedicated `denied_tools: set[str]`? Current `tool_overrides: dict[str, bool]` handles both allow and deny.
3. Per-tool audit granularity: include full `result` or just `result_length`? A `max_audit_result_length` config could cap it.

---

## 8. Related Concepts

- [Agent Harness](agent-harness.md) — Middleware pipeline
- [Middleware System](sdk-consistency.md) — MiddlewareBase pattern
- [Sub-Agent](sub-agent.md) — Worker permission from ScenarioWiring
- [Tool Filter](tool-filter.md) — Complementary: tool-filter removes tools from the pool; permission-mode controls execution of remaining tools