# Design: Trajectory Collector

**Status**: DRAFT
**Last Updated**: 2026-03-08

---

## Overview

TrajectoryCollector captures structured execution events to JSONL files. It serves as the central event bus for recording agent system activity -- tool calls, task lifecycle, hypothesis updates, and LLM completions. Events are written to disk for post-hoc analysis, debugging, and RL training data export. Listeners can subscribe to the event stream for real-time consumption (e.g., the Debug Console).

---

## TrajectoryEvent Schema

Every recorded event conforms to the `TrajectoryEvent` Pydantic model:

```python
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime

class TrajectoryEvent(BaseModel):
    run_id: str                              # Unique identifier for the current execution run
    seq: int                                 # Monotonically increasing sequence number within the run
    timestamp: datetime                      # When the event was recorded
    agent_path: str                          # Dot-separated agent identifier (e.g., "orchestrator", "worker.scout")
    node_name: str                           # Graph node that produced the event (e.g., "agent", "tools")
    event_type: str                          # One of the defined event types (see below)
    data: dict[str, Any]                     # Event-type-specific payload
    task_id: Optional[str] = None            # AgentRuntime task_id, if event relates to a task
    hypothesis_id: Optional[str] = None      # Hypothesis ID, if event relates to hypothesis reasoning
    parent_seq: Optional[int] = None         # Sequence number of the parent event (for causal linking)
```

---

## Event Types

| Event Type | Producer | Payload (data field) |
|-----------|----------|---------------------|
| `tool_call` | Builder (orchestrator hooks) | `{tool_name, args}` |
| `tool_result` | Builder (orchestrator hooks) | `{tool_name, result_preview}` |
| `llm_end` | Builder (orchestrator hooks) | `{model, token_usage, content_preview}` |
| `task_dispatch` | AgentRuntime | `{task_id, agent_id, task_type, instruction_preview}` |
| `task_complete` | AgentRuntime | `{task_id, agent_id, duration_seconds, result_preview}` |
| `task_fail` | AgentRuntime | `{task_id, agent_id, error}` |
| `task_abort` | AgentRuntime | `{task_id, agent_id, reason}` |
| `hypothesis_update` | Orchestrator tools | `{hypothesis_id, status, description}` |

---

## API

### Core Methods

```python
class TrajectoryCollector:
    async def record(self, event: TrajectoryEvent) -> None:
        """Record an event asynchronously. Writes to JSONL file and notifies all listeners."""

    def record_sync(self, event: TrajectoryEvent) -> None:
        """Record an event from a synchronous context. Thread-safe.
        Schedules the async write via the running event loop if available,
        otherwise writes directly under a threading lock."""

    def add_listener(self, callback: Callable) -> None:
        """Register a listener callback. Supports both sync and async callables.
        Async listeners are awaited; sync listeners are called directly."""

    async def close(self) -> None:
        """Flush and close the JSONL file. Must be called for clean shutdown."""
```

---

## Listener Mechanism

Listeners are sync or async callables notified on every event:

- **Async context**: Async listeners are `await`-ed directly. Sync listeners are called directly (not scheduled as tasks).
- **Error isolation**: Listener exceptions are logged but do not prevent event recording or other listeners from being notified.
- **Registration**: `add_listener()` appends to an internal list. No removal API -- listeners live for the collector's lifetime.

Primary consumer: the [Debug Console](debug-console.md), which registers `on_trajectory_event` as a listener.

---

## Thread Safety

- **Async path** (`record`): Protected by `asyncio.Lock` to serialize file writes and listener notification.
- **Sync path** (`record_sync`): Protected by `threading.Lock` for direct file writes from non-async contexts (e.g., sync tool callbacks).
- Sequence number generation uses an `itertools.count()` or equivalent atomic counter.

---

## File Lifecycle

- **Lazy initialization** (`_ensure_file`): The JSONL output file is not opened until the first `record()` call. The file path is derived from `run_id` and an output directory.
- **Explicit cleanup** (`close`): Flushes the write buffer and closes the file handle. After `close()`, further `record()` calls are no-ops or raise.
- **Safety net** (`__del__`): Attempts to close the file if `close()` was not called explicitly. This is a best-effort fallback, not a substitute for proper lifecycle management.

---

## Integration Points

| Component | Events Produced | Mechanism |
|-----------|----------------|-----------|
| **TrajectoryMiddleware** | `tool_call`, `tool_result`, `llm_end` | Middleware hooks (on_llm_start, on_tool_call, on_llm_end) on SimpleAgentLoop |
| **AgentRuntime** | `task_dispatch`, `task_complete`, `task_fail`, `task_abort` | Calls `record()` / `record_sync()` at task lifecycle transitions |
| **Orchestrator tools** | `hypothesis_update` | `update_hypothesis` tool calls `record_sync()` after state update |

The `build_agent_system()` builder creates the TrajectoryCollector during build and injects it into the AgentRuntime. The AgentRuntime uses the collector to record all task lifecycle events. TrajectoryMiddleware is added to the middleware stack to capture LLM and tool events.

---

## Related Documents

- [System Architecture](system-design-overview.md) -- Overall system design
- [Orchestrator](orchestrator.md) -- Trajectory Registry (TaskTraceRef) for hierarchical trace linking
- [Debug Console](debug-console.md) -- Real-time event consumer
- [SDK Consistency](sdk-consistency.md) -- build_agent_system() wires TrajectoryCollector into the agent system

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| JSONL format | Append-only, streaming-friendly, easy to parse line-by-line for analysis |
| Pydantic model for events | Schema validation at creation time; serialization to dict/JSON is trivial |
| Dual async/sync recording | Tools and hooks may run in sync or async contexts; both paths must be supported |
| Listener fan-out | Decouples producers (TrajectoryMiddleware, AgentRuntime) from consumers (Debug Console, future analytics) |
| Lazy file init | Avoids creating empty trajectory files for runs that produce no events |
| Sequence numbering | Enables causal ordering and parent-child linking across events |
