# Task 6: Implement AgentSystemBuilder

**Status**: PENDING
**Depends on**: 4A, 4B, 5
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [generic-state-wrapper.md](../designs/generic-state-wrapper.md) § AgentSystemBuilder
**Assignee**: implementer

## Objective

Implement `builder.py` — `AgentSystemBuilder.build`, `AgentSystem.execute`, `AgentSystem.stream`.

## Functions

### `AgentSystemBuilder.build(system_type, config) -> AgentSystem`
1. `get_state_schema(system_type)`
2. If `config.orchestrator.orchestrator_mode == "react"`: build ReAct system via `create_orchestrator`
3. Other modes: `raise NotImplementedError` (only hypothesis_driven needed for Phase 1)
4. Return `AgentSystem` wrapping compiled graph + TaskManager

### `AgentSystem.execute(input_data) -> dict`
`await self.graph.ainvoke(input_data, config=self.langgraph_config)`

### `AgentSystem.stream(input_data) -> AsyncIterator`
`self.graph.astream(input_data, config=self.langgraph_config)`, yield events.

## Verification

- `build("hypothesis_driven", config)` doesn't raise (with minimal config)
- `execute` returns a dict

## Notes

- Builder needs `SystemConfig` for checkpointer/store, `ScenarioConfig` for agents
- Only `hypothesis_driven` ReAct path needed now
