# Design: Builder (AgentSystemBuilder)

**Status**: DRAFT
**Last Updated**: 2026-03-08

---

## Overview

`AgentSystemBuilder` is the unified entry point for constructing agent systems. It orchestrates the full build pipeline: config validation, dependency creation, tool wiring, graph compilation, and system assembly. The result is an `AgentSystem` that can be executed via a single `execute()` or `stream()` call.

---

## Build Flow

```
validate system_type
    |
    v
create ToolRegistry (load tool definitions from config)
    |
    v
resolve model configs (orchestrator + agents)
    |
    v
create AgentPool (compile sub-agent subgraphs per task_type)
    |
    v
create TaskManager (receives AgentPool)
    |
    v
create checkpointer (Memory or SQLite)
    |
    v
create TrajectoryCollector (JSONL event recorder)
    |
    v
create orchestrator tools (factory closures capture dependencies)
    |
    v
wire dependencies (trajectory -> TaskManager, broadcast callback)
    |
    v
compile orchestrator graph (create_react_agent)
    |
    v
return AgentSystem
```

---

## Dependency Injection

The Builder uses a **factory closure** pattern for orchestrator tools. Tool functions need access to `TaskManager`, `AgentPool`, `TrajectoryCollector`, and the `DiagnosticNotebook`, but these are created during `build()` -- they cannot be passed as constructor arguments to static tool definitions.

### Factory Closure Pattern

```python
def create_orchestrator_tools(
    task_manager: TaskManager,
    agent_pool: AgentPool,
    trajectory: TrajectoryCollector,
) -> list[Tool]:
    """Create orchestrator tools with dependencies captured in closures."""

    @tool
    async def dispatch_agent(agent_id: str, task: str, ...) -> Command:
        # task_manager, agent_pool, trajectory captured from enclosing scope
        managed = await task_manager.submit(agent_id, task, ...)
        trajectory.record_sync(TrajectoryEvent(...))
        ...

    @tool
    async def check_tasks(wait_seconds: int = 10, ...) -> Command:
        results = await task_manager.get_all_status(wait_seconds)
        ...

    return [dispatch_agent, check_tasks, ...]
```

### Public Wiring Methods

The Builder uses public setter methods on `TaskManager` to inject late-binding dependencies:

- `task_manager.set_trajectory(trajectory)` -- Enables task lifecycle event recording
- `task_manager.set_broadcast_callback(callback)` -- Enables event forwarding to external consumers (e.g., WebSocket, Debug Console)

These setters exist because the TrajectoryCollector and broadcast callback are created after the TaskManager, and circular dependencies would result from constructor injection.

---

## Checkpointer

Two checkpointer backends are supported:

| Backend | Context | Init | Use Case |
|---------|---------|------|----------|
| **MemorySaver** | Sync | Immediate | Testing, development, short-lived runs |
| **AsyncSqliteSaver** | Async | Lazy via `_ensure_checkpointer()` | Production, persistent checkpoint storage |

### Lazy Async Init

The SQLite checkpointer requires an async setup step (`await saver.setup()`). Since `build()` may be called from sync context, the actual initialization is deferred:

```python
async def _ensure_checkpointer(self) -> BaseCheckpointSaver:
    """Lazily initialize the async checkpointer on first use."""
    if self._checkpointer is None:
        self._checkpointer = AsyncSqliteSaver(self._db_path)
        await self._checkpointer.setup()
    return self._checkpointer
```

This is invoked by `AgentSystem.__aenter__()` during async context manager entry.

---

## AgentSystem

The unified interface returned by `build()`:

```python
class AgentSystem:
    async def execute(self, input_data: dict) -> dict:
        """Run the agent system to completion and return the final state."""

    async def stream(self, input_data: dict) -> AsyncIterator:
        """Stream execution events as they occur."""

    async def __aenter__(self) -> "AgentSystem":
        """Async context manager entry. Initializes async resources (checkpointer, trajectory)."""

    async def __aexit__(self, *exc) -> None:
        """Async context manager exit. Closes trajectory collector and checkpointer."""
```

### Usage

```python
async with builder.build() as system:
    result = await system.execute({"task": "Investigate API latency spike"})
```

The async context manager ensures proper resource cleanup (trajectory file closure, checkpointer shutdown) even on exceptions.

---

## Current Limitations

- **Only `react` mode** is implemented. The `graph` mode (for StateGraph-based systems like memory_extraction, sequential, decision_tree) raises `NotImplementedError`.
- **Single worker config**: The `AgentPool` currently uses one shared worker configuration with task_type prompts, not per-agent-id pools. See [sub-agent.md](sub-agent.md#agent-pool) for details.
- **Knowledge Store**: Not yet wired. Knowledge tools exist but the LangGraph Store backend is not injected by the Builder.

---

## Related Documents

- [System Architecture](system-design-overview.md) -- Overall system design
- [Orchestrator](orchestrator.md) -- Orchestrator tools and TaskManager
- [Sub-Agent](sub-agent.md) -- AgentPool and worker configuration
- [Generic State Wrapper](generic-state-wrapper.md) -- AgentSystemBuilder interface contract
- [Trajectory Collector](trajectory.md) -- Event recording wired by Builder
- [Debug Console](debug-console.md) -- Terminal UI wired as trajectory listener

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Factory closure for tools | Tools need runtime dependencies (TaskManager, etc.) that are created during build; closures capture them without global state |
| Lazy async checkpointer | `build()` can be called from sync context; async setup deferred to first use via context manager |
| Setter-based wiring | Avoids circular dependencies between TaskManager and TrajectoryCollector |
| Async context manager | Guarantees cleanup of file handles and database connections |
| Single `build()` entry point | Encapsulates all construction complexity; callers get a ready-to-use `AgentSystem` |
