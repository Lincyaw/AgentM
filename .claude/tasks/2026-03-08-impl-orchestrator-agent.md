# Task 4B: Implement Orchestrator Agent

**Status**: PENDING
**Depends on**: 1A
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [orchestrator.md](../designs/orchestrator.md) § Orchestrator Creation
**Assignee**: implementer

## Objective

Implement `agents/orchestrator.py` (2 functions).

## Functions

### `build_orchestrator_prompt(system_prompt_template) -> Callable`
Return callable accepting `ExecutorState`. It:
1. Formats `state['notebook']` via `format_notebook_for_llm`
2. Renders template with formatted notebook
3. Returns `[SystemMessage(content=rendered), *state['messages']]`

### `create_orchestrator(config, tools, checkpointer, store) -> CompiledGraph`
1. Build `ChatOpenAI(model, temperature)`
2. Build prompt callable via `build_orchestrator_prompt`
3. Return `create_react_agent(model, tools, prompt=callable, state_schema=ExecutorState, checkpointer=checkpointer, store=store)`

## Verification

- `test_interface_consistency.py::TestOrchestratorCreationImports` passes
- `build_orchestrator_prompt` returns callable
- `create_orchestrator` returns compiled graph

## Notes

- Verify `create_react_agent` supports callable as `prompt` parameter
