"""Sub-Agent creation and AgentPool management.

All functions/methods are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any, Literal

from agentm.config.schema import AgentConfig, ScenarioConfig
from agentm.core.tool_registry import ToolRegistry


def create_sub_agent(
    agent_id: str,
    config: AgentConfig,
    tool_registry: ToolRegistry,
    task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
) -> Any:
    """Create a Sub-Agent subgraph via create_react_agent. Returns a CompiledGraph.

    The task_type parameter selects a prompt overlay from config.task_type_prompts
    (if configured) to specialize the agent's system prompt for the given task type.
    """
    raise NotImplementedError


class AgentPool:
    """Collection of independently compiled Sub-Agent subgraphs.

    Sub-Agents are NOT added as graph nodes. They are compiled independently
    and managed by the TaskManager, which launches them as asyncio.Tasks.
    """

    def __init__(self, scenario_config: ScenarioConfig, tool_registry: ToolRegistry) -> None:
        self.agents: dict[str, Any] = {}
        raise NotImplementedError

    def get_agent(self, agent_id: str) -> Any:
        """Get a compiled sub-agent subgraph by ID."""
        raise NotImplementedError
