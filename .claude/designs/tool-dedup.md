# Design: Tool Call Deduplication

**Status**: DRAFT
**Last Updated**: 2026-03-09

---

## Overview

Sub-Agents in the ReAct loop frequently repeat identical tool calls (same tool name + same arguments), wasting tokens on duplicate API responses. This module provides a **two-layer deduplication** system:

1. **Pre-model Hook (soft limit)**: Before each LLM call, scans message history for past tool calls, injects a short SystemMessage telling the LLM not to repeat those calls (no result content included — to save tokens).
2. **Tool Wrapper (hard limit)**: Wraps each tool so that repeated `(tool_name, args)` calls are **blocked** with a short rejection message, without re-executing or returning cached results.

---

## Data Structures

### DedupTracker

Per-task in-memory cache using `OrderedDict` with FIFO eviction.

```python
from collections import OrderedDict

class DedupTracker:
    """Track tool call results for deduplication.

    - Key: (tool_name, json.dumps(args, sort_keys=True, default=str))
    - Value: tool result string
    - FIFO eviction when max_cache_size exceeded
    """

    def __init__(self, max_cache_size: int = 50):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_cache_size

    def make_key(self, tool_name: str, args: dict) -> str:
        """Deterministic cache key — arg order does not matter."""
        import json
        args_str = json.dumps(args, sort_keys=True, default=str)
        return f"{tool_name}:{args_str}"

    def lookup(self, key: str) -> str | None:
        """Return cached result or None."""
        return self._cache.get(key)

    def store(self, key: str, result: str) -> None:
        """Store result, evicting oldest entry if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = result
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
```

---

## Hook: build_dedup_hook

```python
def build_dedup_hook(tracker: DedupTracker) -> Callable:
    """Pre-model hook that:
    1. Scans message history for tool call/response pairs → populates tracker
    2. If tracker has entries matching recent AI tool calls, injects SystemMessage:
       "You already called X with args Y — you already have this result."
       No result content is included in the reminder (token savings).

    Excludes `think` tool from dedup tracking.
    """
```

**Hook position in chain**: `budget → compression → dedup → llm_input`

The hook scans all `AIMessage` tool calls and their corresponding `ToolMessage` responses. For each pair where `tool_name != "think"`, it records the key in the tracker. Then, for any tool calls in the most recent `AIMessage` that already exist in the tracker, it injects a short reminder SystemMessage (without result content).

---

## Wrapper: wrap_tool_with_dedup

```python
def wrap_tool_with_dedup(tool: StructuredTool, tracker: DedupTracker) -> StructuredTool:
    """Wrap a tool so repeated (name, args) calls are blocked.

    - On first call: execute tool, record key in tracker, return result
    - On repeat call: return short rejection message ("BLOCKED: ...")
    - Does NOT return cached results — the goal is token savings
    - Preserves tool name, description, and args_schema
    """
```

---

## Configuration

```python
class DedupConfig(BaseModel):
    enabled: bool = True
    max_cache_size: int = 50
```

Added as `Optional[DedupConfig]` on `ExecutionConfig`:

```yaml
# scenarios/rca_hypothesis/scenario.yaml
agents:
  worker:
    execution:
      dedup:
        enabled: true
        max_cache_size: 50
```

---

## Integration Points

### create_sub_agent (sub_agent.py)

When `config.execution.dedup` is set and enabled:

1. Create `DedupTracker(max_cache_size=config.execution.dedup.max_cache_size)`
2. Wrap all tools (except `think`) with `wrap_tool_with_dedup(tool, tracker)`
3. Create dedup hook via `build_dedup_hook(tracker)`
4. Chain into `pre_model_hook`: `budget → compression → dedup → llm_input`

### Exclusions

- `think` tool is excluded from both hook scanning and wrapper (it's a free scratchpad)
- Dedup is per-agent-invocation — tracker is not shared across agents

---

## Related Documents

- [Sub-Agent](sub-agent.md) — Agent creation and hook chain
- [System Design Overview](system-design-overview.md) — Configuration system

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two-layer (hook + wrapper) | Hook is soft (LLM may still try); wrapper is hard (blocks with short message). Belt and suspenders |
| OrderedDict + FIFO | Simple, bounded memory. Oldest entries evicted first |
| `sort_keys=True` in key | Arg order should not affect dedup — `{a:1, b:2}` == `{b:2, a:1}` |
| Exclude `think` tool | Think is a free scratchpad, no API cost, caching would break reasoning |
| Per-task tracker | No cross-agent cache pollution. Each sub-agent invocation is independent |
| SystemMessage injection | Consistent with budget_hook pattern — visible to LLM but not persisted in state |
| No cached result in responses | Neither hook nor wrapper returns cached data — the entire goal is token savings |
