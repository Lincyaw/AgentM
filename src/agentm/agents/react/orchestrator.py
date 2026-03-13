"""Orchestrator graph creation.

Builds the Orchestrator's CompiledGraph via create_react_agent with
a prompt callable that injects working memory into LLM context.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentm.config.schema import OrchestratorConfig
from agentm.middleware.compression import build_compression_hook
from agentm.core.prompt import load_prompt_template
from agentm.core.trajectory import TrajectoryCollector
from agentm.models.output import get_output_schema


def build_orchestrator_prompt(
    system_prompt_template: str,
    format_context: Callable[[dict], str] | None = None,
) -> Callable:
    """Build a prompt callable that injects working memory into LLM context before each call.

    The returned callable receives the current state and returns a list of
    messages with the system prompt populated from the format_context callable.

    Args:
        system_prompt_template: Template string for the system prompt.
        format_context: Optional callable ``(state: dict) -> str`` producing
            the working-memory text. If not provided, the system prompt
            receives empty context.
    """

    def prompt(state: dict) -> list:
        if format_context is not None:
            context_text = format_context(state)
        else:
            context_text = ""

        system_prompt = load_prompt_template(
            system_prompt_template,
            notebook=context_text,
            context=context_text,
        )
        return [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]

    return prompt


def create_orchestrator(
    config: OrchestratorConfig,
    tools: list,
    checkpointer: Any,
    store: Any,
    format_context: Callable[[dict], str] | None = None,
    model_config: Any | None = None,
    trajectory: TrajectoryCollector | None = None,
) -> Any:
    """Create the Orchestrator via create_react_agent.

    Uses build_orchestrator_prompt to create the prompt callable.
    Does NOT pass state_schema — create_react_agent's default AgentState
    includes 'remaining_steps' which is required by the framework.

    Args:
        config: OrchestratorConfig from the scenario YAML.
        tools: List of tools available to the orchestrator.
        checkpointer: LangGraph checkpointer (or None).
        store: LangGraph store backend (or None).
        format_context: Optional callable for working-memory formatting.
        model_config: Optional model configuration (API key, base_url).
        trajectory: Optional TrajectoryCollector for event recording.

    Returns a CompiledGraph (langgraph).
    """
    llm_kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }
    if model_config is not None:
        if hasattr(model_config, "api_key") and model_config.api_key:
            llm_kwargs["api_key"] = model_config.api_key
        if hasattr(model_config, "base_url") and model_config.base_url:
            llm_kwargs["base_url"] = model_config.base_url
    model = ChatOpenAI(**llm_kwargs)

    prompt_callable = build_orchestrator_prompt(
        config.prompts.get("system", ""),
        format_context=format_context,
    )

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "prompt": prompt_callable,
    }
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    if store is not None:
        agent_kwargs["store"] = store

    # Orchestrator message compression via pre_model_hook
    hooks: list[Any] = []
    if config.compression is not None and config.compression.enabled:
        hooks.append(build_compression_hook(config.compression))
    if trajectory is not None:
        from agentm.middleware.trajectory import build_llm_input_hook

        hooks.append(build_llm_input_hook(trajectory, ["orchestrator"]))
    if hooks:
        if len(hooks) == 1:
            agent_kwargs["pre_model_hook"] = hooks[0]
        else:

            def _chain(*fns: Any) -> Any:
                def chained(state: dict) -> dict:
                    result = state
                    for fn in fns:
                        result = fn(result)
                    return result

                return chained

            agent_kwargs["pre_model_hook"] = _chain(*hooks)

    # Structured output via response_format=(extraction_prompt, schema)
    if config.output is not None:
        output_prompt = load_prompt_template(config.output.prompt)
        output_schema = get_output_schema(config.output.schema_name)
        agent_kwargs["response_format"] = (output_prompt, output_schema)

    return create_react_agent(**agent_kwargs)
