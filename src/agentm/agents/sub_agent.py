"""Sub-Agent creation and AgentPool management."""

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
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from agentm.core.prompt import load_prompt_template
    from agentm.models.state import SubAgentState

    model = ChatOpenAI(model=config.model, temperature=config.temperature)

    tools = [
        tool_registry.get(name).create_with_config(**config.tool_settings.get(name, {}))
        for name in config.tools
    ]

    prompt = load_prompt_template(config.prompt, agent_id=agent_id)

    if config.task_type_prompts and task_type in config.task_type_prompts:
        overlay = load_prompt_template(config.task_type_prompts[task_type], agent_id=agent_id)
        prompt = prompt + "\n\n" + overlay

    if config.compression is not None:
        from agentm.core.compression import build_compression_hook
        pre_model_hook = build_compression_hook(config.compression)
    else:
        pre_model_hook = None

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name=agent_id,
        state_schema=SubAgentState,
        pre_model_hook=pre_model_hook,
    )


class AgentPool:
    """Collection of independently compiled Sub-Agent subgraphs.

    Sub-Agents are NOT added as graph nodes. They are compiled independently
    and managed by the TaskManager, which launches them as asyncio.Tasks.
    """

    def __init__(self, scenario_config: ScenarioConfig, tool_registry: ToolRegistry) -> None:
        self.agents: dict[str, Any] = {}
        for agent_id, agent_config in scenario_config.agents.items():
            self.agents[agent_id] = create_sub_agent(agent_id, agent_config, tool_registry)

    def get_agent(self, agent_id: str) -> Any:
        """Get a compiled sub-agent subgraph by ID."""
        return self.agents[agent_id]
