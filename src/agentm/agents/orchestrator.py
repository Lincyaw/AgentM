"""Orchestrator graph creation.

Builds the Orchestrator's CompiledGraph via create_react_agent with
a prompt callable that injects the DiagnosticNotebook into LLM context.
"""

from __future__ import annotations

from typing import Any, Callable

from agentm.config.schema import ModelConfig, OrchestratorConfig
from langchain_core.messages import SystemMessage

from agentm.core.notebook import format_notebook_for_llm
from agentm.core.prompt import load_prompt_template
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def build_orchestrator_prompt(system_prompt_template: str) -> Callable:
    """Build a prompt callable that injects Notebook into LLM context before each call.

    The returned callable receives the current ExecutorState and returns a list of
    messages with the system prompt populated from the DiagnosticNotebook.
    """

    def prompt(state: dict) -> list:
        notebook_data = state.get("notebook")
        if notebook_data is not None:
            notebook_text = format_notebook_for_llm(notebook_data)
        else:
            notebook_text = "(Investigation starting — no data collected yet)"
        system_prompt = load_prompt_template(
            system_prompt_template, notebook=notebook_text
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
    model_config: ModelConfig | None = None,
) -> Any:
    """Create the Orchestrator via create_react_agent.

    Uses build_orchestrator_prompt to create the prompt callable,
    passes tools, and sets state_schema=ExecutorState.

    Returns a CompiledGraph (langgraph).
    """

    llm_kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }
    if model_config is not None:
        llm_kwargs["api_key"] = model_config.api_key
        if model_config.base_url:
            llm_kwargs["base_url"] = model_config.base_url
    model = ChatOpenAI(**llm_kwargs)

    prompt_callable = build_orchestrator_prompt(config.prompts.get("system", ""))

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "prompt": prompt_callable,
    }
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    if store is not None:
        agent_kwargs["store"] = store

    return create_react_agent(**agent_kwargs)
