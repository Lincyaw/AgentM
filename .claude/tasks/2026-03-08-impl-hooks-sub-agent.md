# Task 4A: Implement Hooks and Sub-Agent

**Status**: PENDING
**Depends on**: 3, 2B
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [sub-agent.md](../designs/sub-agent.md) § Implementation via create_react_agent, [orchestrator.md](../designs/orchestrator.md) § Instruction Injection
**Assignee**: implementer

## Objective

Implement `agents/hooks.py` (2 functions) and `agents/sub_agent.py` (`create_sub_agent` + `AgentPool`).

## Functions

### `build_instruction_hook(task_manager, task_id) -> Callable`
Closure: calls `task_manager.consume_instructions(task_id)`, wraps as `HumanMessage`, prepends to messages. Returns `{'messages': [...]}`.

### `build_combined_hook(instruction_hook, compression_hook) -> Callable`
Chain: apply instruction_hook first, then compression_hook on result. Return whichever key compression_hook chose.

### `create_sub_agent(agent_id, config, tool_registry, task_type="scout") -> CompiledGraph`
1. Build `ChatOpenAI(model, temperature)`
2. Build tools from registry
3. Load prompt, apply task_type overlay if configured
4. Build `pre_model_hook` via `build_combined_hook`
5. Return `create_react_agent(model, tools, prompt, state_schema=SubAgentState, pre_model_hook=hook)`

### `AgentPool.__init__(scenario_config, tool_registry)`
Build agents dict: `{agent_id: create_sub_agent(...)}`

### `AgentPool.get_agent(agent_id) -> Any`
Dict lookup with `KeyError` on miss.

## Verification

- `test_interface_consistency.py::TestHooksModuleExports` passes
- `build_instruction_hook` returns callable
- `build_combined_hook` chains both hooks
- `create_sub_agent` doesn't raise (with mocked registry)

## Notes

- `from langgraph.prebuilt import create_react_agent`
- Skip compression hook if `config.compression is None`
