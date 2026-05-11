**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](../pluggable-architecture.md) and
[extension-as-scenario.md](../extension-as-scenario.md).

---

# Design: Tool Filter (Whitelist / Disallowed-list)

**Status**: DRAFT
**Last Updated**: 2026-03-31

---

## 1. Overview

A three-layer tool filtering mechanism that controls which tools an agent can access. Prevents workers from using orchestration tools (recursion guard), allows per-agent exclusion of specific tools, and supports whitelist semantics for restricting agents to a known subset.

---

## 2. Background & Motivation

### 2.1 Current State

Tool resolution for workers (`WorkerLoopFactory._build_tools()`) and the orchestrator (`_assemble_orchestrator_tools()`) follows a simple additive model:

1. Look up each name in `AgentConfig.tools` from the `ToolRegistry`.
2. Append `extra_tools` (from `ScenarioWiring`).
3. Optionally add the `think` tool.

There is **no subtraction step**. Every tool listed in config is unconditionally included.

### 2.2 Why This Is a Problem

| Problem | Impact |
|---------|--------|
| **Recursive dispatch** | A worker with access to `dispatch_agent` or `check_tasks` could spawn new workers, creating unbounded recursion. |
| **Accidental capability leak** | A scenario's `extra_tools` are appended to all workers regardless of task type. A `verify` worker should not have `update_hypothesis` if that tool is only meant for `scout`. |
| **No declarative exclusion** | To prevent a tool from being used, you must carefully omit it from `AgentConfig.tools` AND ensure no `extra_tools` source injects it. This is fragile and implicit. |

### 2.3 Inspiration

Production agent systems often use a three-layer filter model:

1. **Global disallowed list** — SDK-level, prevents sub-agents from using dispatch tools (recursion guard).
2. **Agent-level disallowed list** — Declarative exclusion of specific tools per agent definition.
3. **Whitelist** — `["*"]` means all available tools; otherwise only the listed tools are included.

This design adapts that model to AgentM's architecture.

---

## 3. Design

### 3.1 Three-Layer Filter Model

Tool resolution proceeds in three stages, applied in order:

```
All available tools (registry + extra_tools + think)
    │
    ▼
┌─────────────────────────────┐
│ Layer 1: Global Disallowed  │  Remove WORKER_DISALLOWED_TOOLS
│          (SDK-enforced)     │  (only applied to workers, not orchestrator)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Layer 2: Whitelist          │  If tools != ["*"], keep only listed names
│          (AgentConfig)      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Layer 3: Agent Disallowed   │  Remove names in disallowed_tools
│          (AgentConfig)      │
└─────────────────────────────┘
    │
    ▼
Final tool list for this agent
```

**Order rationale**: Global disallowed is checked first because it is a hard security boundary — no config override should be able to re-add `dispatch_agent` to a worker. Whitelist narrows second. Agent-level disallowed applies last as a fine-grained exclusion on top of the whitelist.

### 3.2 `WORKER_DISALLOWED_TOOLS` — Global Disallowed List

A module-level constant defining tools that workers must never have access to:

```python
WORKER_DISALLOWED_TOOLS: frozenset[str] = frozenset({
    "dispatch_agent",
    "check_tasks",
    "inject_instruction",
    "abort_task",
})
```

These are exactly the four orchestrator tools defined in `agentm/tools/orchestrator.py`. Workers must not have dispatch capabilities — all task coordination flows through the orchestrator.

**Design decision**: This is a `frozenset`, not a configurable list. The recursion guard is an SDK invariant, not a scenario-level choice. If a future scenario needs worker-to-worker dispatch, it should introduce a separate, bounded mechanism (e.g., a `request_peer` tool with depth limits), not remove the guard.

### 3.3 `AgentConfig` Field Extensions

Two new optional fields on `AgentConfig`:

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `tools` | `list[str]` | (existing) | Whitelist. `["*"]` = all available tools. `["a", "b"]` = only these. `[]` = no tools. |
| `disallowed_tools` | `list[str]` | `[]` | Exclusion list. Applied after whitelist. |

**`tools: ["*"]` semantics**: The whitelist layer is skipped — all tools that survive the global disallowed filter are included. Useful for orchestrator agents and development/debugging scenarios.

**`disallowed_tools` semantics**: Always applied after whitelist resolution. Allows excluding specific tools without enumerating all the others.

### 3.4 `resolve_tools()` Function

A pure function that implements the three-layer filter:

```python
def resolve_tools(
    available_tools: list[Tool],
    *,
    whitelist: list[str],
    disallowed_tools: list[str] | None = None,
    global_disallowed: frozenset[str] | None = None,
) -> list[Tool]:
    """Apply three-layer tool filtering.

    Args:
        available_tools: All tools from registry + extra_tools + think.
        whitelist: Tool names to include. ["*"] means all.
        disallowed_tools: Tool names to exclude (agent-level).
        global_disallowed: Tool names to exclude (SDK-level).

    Returns:
        Filtered list of Tool instances, preserving original order.

    Raises:
        ValueError: If a whitelisted tool name is not found in available_tools
                    (unless whitelist is ["*"]).
    """
```

**Implementation logic**:

```
1. Build a name->Tool index from available_tools
2. Layer 1: Remove all names in global_disallowed from the index
3. Layer 2: If whitelist != ["*"]:
     - For each name in whitelist:
         - If name not in index: raise ValueError
         - Collect the Tool
     - Result = collected tools (in whitelist order)
   Else:
     - Result = all remaining tools (in available_tools order)
4. Layer 3: Remove any tool whose name is in disallowed_tools
5. Return result
```

**Key behaviors**:
- Preserves ordering (whitelist order when explicit, original order when `["*"]`).
- Raises on unknown whitelist names to catch typos early (fail-fast at startup).
- Silently ignores unknown names in `disallowed_tools` (defensive — a tool may not exist in all scenarios).
- Logs a warning when `disallowed_tools` contains a name that was already filtered by global disallowed (redundant config).

### 3.5 Integration with `WorkerLoopFactory`

`WorkerLoopFactory._build_tools()` changes from the current additive model to use `resolve_tools()`:

1. **Collect all available tools** (unchanged): registry tools + `extra_tools` + optional `think` tool.
2. **Apply three-layer filter** (new): Call `resolve_tools()` with the worker's whitelist, disallowed list, and `WORKER_DISALLOWED_TOOLS`.

**Note on orchestrator**: The orchestrator builder does NOT apply `WORKER_DISALLOWED_TOOLS`. The orchestrator is supposed to have dispatch tools. However, `resolve_tools()` can still be used for the orchestrator with `global_disallowed=None` if orchestrator-level disallowed lists are needed in the future.

### 3.6 Module Placement

```
src/agentm/harness/tool_filter.py
```

The `resolve_tools()` function and `WORKER_DISALLOWED_TOOLS` constant live in a dedicated module. This keeps the filtering logic independent of `WorkerLoopFactory` (testable in isolation) and avoids circular imports since it only depends on `Tool` from `agentm.core.tool`.

---

## 4. Config Format

### 4.1 YAML (scenario.yaml)

**Explicit whitelist (current behavior, enhanced with disallowed_tools)**:

```yaml
agents:
  worker:
    model: "gpt-4o-mini"
    tools: [check_metrics, query_logs, query_traces]
    disallowed_tools: []  # optional, default empty
```

**Wildcard whitelist with exclusions**:

```yaml
agents:
  worker:
    model: "gpt-4o-mini"
    tools: ["*"]
    disallowed_tools: [update_hypothesis, dangerous_tool]
```

**No tools (pure reasoning agent)**:

```yaml
agents:
  synthesizer:
    model: "gpt-4o"
    tools: []
```

### 4.2 Markdown Frontmatter (Agent Definition Files)

For future agent definition files:

```yaml
---
name: worker
model: gpt-4o-mini
tools: ["*"]
disallowed_tools:
  - update_hypothesis
---
```

---

## 5. Edge Cases

| Case | Behavior | Rationale |
|------|----------|-----------|
| Tool in both whitelist and disallowed | Excluded (`disallowed` wins) | Disallowed is a safety mechanism |
| `["*"]` alongside other names | `ValueError` at startup | `"*"` must be sole element |
| Unknown name in whitelist | `ValueError` at startup | Fail-fast prevents silent misconfiguration |
| Unknown name in disallowed | Silently ignored | Defensive; shared lists across scenarios |
| All tools filtered out | Return empty list + warning log | Agent runs as pure LLM reasoning |
| `extra_tools` named `dispatch_agent` | Removed by global disallowed | Primary safety guarantee |
| `think` tool | Subject to filter like any other | Can be excluded via `disallowed_tools: ["think"]` |

---

## 6. Constraints and Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Global disallowed is a constant | `frozenset` in source | Recursion prevention is an SDK safety invariant |
| Filter order: global → whitelist → disallowed | Fixed, not configurable | Security layers applied in deterministic order |
| `resolve_tools()` is a pure function | No side effects beyond logging | Testable, predictable, no hidden state |
| Orchestrator skips global disallowed | By design | Orchestrator must have dispatch tools |

---

## 7. Impact on Related Designs

### [Sub-Agent](sub-agent.md)

- **Section 2 (WorkerLoopFactory)**: `_build_tools()` method description needs updating to reflect the three-layer filter.
- **Section 9 (Configuration)**: `AgentConfig` listing needs `disallowed_tools` field.

### [Agent Harness](agent-harness.md)

- Minimal impact. A brief mention of `tool_filter.py` as a harness utility is sufficient.

### [SDK Consistency](sdk-consistency.md)

- The `resolve_tools()` function aligns with the SDK consistency goal: a single canonical path for tool resolution.

---

## 8. Open Questions

1. **Should `_assemble_orchestrator_tools()` also use `resolve_tools()`?** Unifying through `resolve_tools()` (with `global_disallowed=None`) would improve consistency but may add complexity for a path that currently works fine.

2. **Per-task-type tool filtering**: Should `disallowed_tools` vary by `task_type`? This could be handled by extending `task_type_prompts` to `task_type_config` with per-type tool overrides, but is a separate design concern.

---

## 9. Related Documents

- [Sub-Agent](sub-agent.md) — Worker creation and tool assignment
- [Agent Harness](agent-harness.md) — SDK harness architecture
- [SDK Consistency](sdk-consistency.md) — Unified SDK patterns
- [Tool Dedup](tool-dedup.md) — Tool call deduplication (operates after tool filtering)