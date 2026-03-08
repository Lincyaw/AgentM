"""AgentSystem and AgentSystemBuilder — unified entry point for all agent systems."""

from __future__ import annotations

from typing import Any, AsyncIterator

from agentm.config.schema import ScenarioConfig
from agentm.models.state_registry import get_state_schema


class AgentSystem:
    """Unified interface for all agent systems."""

    def __init__(self, graph: Any, config: dict[str, Any] | None = None) -> None:
        self.graph = graph
        self.langgraph_config = config or {}

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent system with the given input. Returns final state."""
        return await self.graph.ainvoke(input_data, config=self.langgraph_config)

    async def stream(self, input_data: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream events from the agent system execution."""
        async for event in self.graph.astream(input_data, config=self.langgraph_config):
            yield event


class AgentSystemBuilder:
    """Unified entry point for building any agent system.

    Internally selects the appropriate architecture based on system_type:
    - ReAct-based (create_react_agent): For exploratory, non-linear scenarios like RCA
    - StateGraph-based (custom graph with phase nodes): For linear, deterministic scenarios
    """

    @staticmethod
    def build(system_type: str, config: ScenarioConfig) -> AgentSystem:
        """Build an AgentSystem from a system type and scenario config."""
        from agentm.agents.orchestrator import create_orchestrator
        from agentm.agents.sub_agent import AgentPool
        from agentm.core.task_manager import TaskManager
        from agentm.core.tool_registry import ToolRegistry

        get_state_schema(system_type)  # validate system_type exists

        if config.orchestrator.orchestrator_mode == "react":
            tool_registry = ToolRegistry()
            agent_pool = AgentPool(config, tool_registry)
            task_manager = TaskManager()

            import agentm.tools.orchestrator as orch_tools
            orch_tools._task_manager = task_manager
            orch_tools._agent_pool = agent_pool

            tools: list[Any] = []
            graph = create_orchestrator(
                config=config.orchestrator,
                tools=tools,
                checkpointer=None,
                store=None,
            )
            return AgentSystem(graph)

        raise NotImplementedError(
            f"Orchestrator mode {config.orchestrator.orchestrator_mode!r} not yet implemented"
        )
