# Task 3: Implement TaskManager

**Status**: PENDING
**Depends on**: 2B
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [orchestrator.md](../designs/orchestrator.md) § TaskManager and Orchestrator Tools
**Assignee**: implementer
**Risk**: HIGH — asyncio lifecycle management

## Objective

Implement all 6 methods of `TaskManager`. Most complex task — manages async Sub-Agent lifecycles.

## Implementation Order (sync first, async second)

### 1. `get_task(task_id) -> ManagedTask`
Dict lookup: `self._tasks[task_id]`. `KeyError` if not found.

### 2. `consume_instructions(task_id) -> list[str]`
Dequeue all pending: copy list, clear original, return copy.

### 3. `inject(task_id, instruction) -> None`
Check `task.status == RUNNING`, else `ValueError`. Append to `task.pending_instructions`.

### 4. `abort(task_id, reason) -> None`
Check RUNNING. Cancel `task.asyncio_task`. Set FAILED + error_summary.

### 5. `submit(agent_id, instruction, task_type, hypothesis_id, **kwargs) -> str`
Generate UUID. Create `ManagedTask`. Create `asyncio.Task` via `create_task`. Store and return `task_id`.

### 6. `_execute_agent(managed, subgraph, config) -> None`
Core async loop with retry:
```python
try:
    result = await subgraph.ainvoke(...)
    managed.status = COMPLETED
    managed.result = result
except CancelledError:
    raise  # must propagate
except Exception:
    managed.status = FAILED
    managed.error_summary = str(e)
```

## Verification

- `get_task` raises `KeyError` for unknown ID
- `consume_instructions` dequeues and clears
- `inject` raises `ValueError` for non-RUNNING
- `abort` cancels task, sets FAILED
- `test_managed_task.py` still passes
- `test_tool_signatures.py::TestTaskManagerSubmitSignature` passes

## Notes

- `CancelledError` must propagate — never catch silently
- Retry: exponential backoff with `max_attempts` from config
