"""Sub-Agent creation and AgentPool management."""

from __future__ import annotations

from typing import Any, Literal

from agentm.config.schema import AgentConfig, ModelConfig, ScenarioConfig
from agentm.core.tool_registry import ToolRegistry


def create_sub_agent(
    agent_id: str,
    config: AgentConfig,
    tool_registry: ToolRegistry,
    task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
    model_config: ModelConfig | None = None,
) -> Any:
    """Create a Sub-Agent subgraph via create_react_agent. Returns a CompiledGraph.

    When config.prompt is None, uses task_type_prompts[task_type] directly as the
    full system prompt (not an overlay). When config.prompt is set, the task_type
    prompt is appended as an overlay.
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from agentm.core.prompt import load_prompt_template

    llm_kwargs: dict[str, Any] = {"model": config.model, "temperature": config.temperature}
    if model_config is not None:
        llm_kwargs["api_key"] = model_config.api_key
        if model_config.base_url:
            llm_kwargs["base_url"] = model_config.base_url
    model = ChatOpenAI(**llm_kwargs)

    tools = [
        tool_registry.get(name).create_with_config(**config.tool_settings.get(name, {}))
        for name in config.tools
    ]

    if config.prompt is None:
        if config.task_type_prompts and task_type in config.task_type_prompts:
            prompt = load_prompt_template(config.task_type_prompts[task_type], agent_id=agent_id)
        else:
            prompt = ""
    else:
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
        pre_model_hook=pre_model_hook,
    )


class AgentPool:
    """Lazy-init pool of worker agents keyed by task_type.

    Sub-Agents are NOT added as graph nodes. They are compiled independently
    and managed by the TaskManager, which launches them as asyncio.Tasks.
    """

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        tool_registry: ToolRegistry,
        model_config: ModelConfig | None = None,
    ) -> None:
        self._worker_config = scenario_config.agents["worker"]
        self._tool_registry = tool_registry
        self._model_config = model_config
        self._workers: dict[str, Any] = {}

    def get_worker(self, task_type: str) -> Any:
        """Get or create a compiled worker agent subgraph for the given task_type."""
        if task_type not in self._workers:
            self._workers[task_type] = create_sub_agent(
                f"worker-{task_type}",
                self._worker_config,
                self._tool_registry,
                task_type,
                self._model_config,
            )
        return self._workers[task_type]

    def get_agent(self, agent_id: str) -> Any:
        """Get a compiled sub-agent subgraph by ID (legacy compat)."""
        return self._workers.get(agent_id)
