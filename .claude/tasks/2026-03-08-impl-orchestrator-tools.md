# Task 5: Implement Orchestrator Tools + Knowledge Tools

**Status**: PENDING
**Depends on**: 3, 1A
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [orchestrator.md](../designs/orchestrator.md) § Orchestrator Tools, [generic-state-wrapper.md](../designs/generic-state-wrapper.md) § Knowledge Tools
**Assignee**: implementer
**Risk**: HIGH — LangGraph `Command` objects

## Objective

Implement all 7 tool functions in `tools/orchestrator.py` and 5 knowledge tools in `tools/knowledge.py`.

## Orchestrator Tools

All tools returning `Command` use `InjectedToolCallId` for `ToolMessage` routing. Access `TaskManager`/Notebook via closure or `InjectedState`.

### `dispatch_agent` → `Command`
Submit to TaskManager, update Notebook with ExplorationStep, return Command with notebook update + ToolMessage.

### `check_tasks` → `Command`
Get all status, for completed tasks add collected_data + ExplorationStep, return Command.

### `inject_instruction` → `str`
Call `task_manager.inject()`, return confirmation.

### `abort_task` → `str`
Call `task_manager.abort()`, return confirmation.

### `update_hypothesis` → `Command`
Validate transition, add/update hypothesis, return Command with notebook update.

### `remove_hypothesis` → `Command`
Remove from notebook dict, return Command.

### `recall_history` → `str`
Search pre-compression checkpoints. Phase 1: placeholder if no refs.

## Knowledge Tools

Access LangGraph `store` via `InjectedStore` or closure.

### `knowledge_search`, `knowledge_list`, `knowledge_read`, `knowledge_write`, `knowledge_delete`
Convert paths to namespaces (using existing `path_to_namespace*` utils), call corresponding `store.*` methods.

## Verification

- `test_tool_signatures.py` all pass
- `test_interface_consistency.py::TestTaskTypeLiteralConsistency` passes
- `update_hypothesis` rejects REJECTED → CONFIRMED (P7)
- `dispatch_agent` returns Command with notebook update

## Notes

- Tool factory pattern: `make_dispatch_agent(task_manager, agent_pool)` returns the tool function
- `InjectedToolCallId` already in signatures
