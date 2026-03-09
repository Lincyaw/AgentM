"""Orchestrator graph creation.

Builds the Orchestrator's CompiledGraph via create_react_agent with
a prompt callable that injects the DiagnosticNotebook into LLM context.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentm.config.schema import OrchestratorConfig
from agentm.core.compression import (
    build_compression_hook,
    compress_completed_phase,
    count_tokens,
)
from agentm.core.notebook import format_notebook_for_llm, should_compress_phase
from agentm.core.prompt import load_prompt_template


def build_orchestrator_prompt(system_prompt_template: str) -> Callable:
    """Build a prompt callable that injects Notebook into LLM context before each call.

    The returned callable receives the current HypothesisDrivenState and returns a list of
    messages with the system prompt populated from the DiagnosticNotebook.
    """

    def prompt(state: dict) -> list:
        notebook_data = state.get("notebook")
        if notebook_data is not None:
            # Compress completed phases for LLM input only (state is not mutated)
            notebook_for_llm = notebook_data
            for phase in ("exploration", "generation", "verification"):
                if should_compress_phase(notebook_for_llm, phase):
                    notebook_for_llm = compress_completed_phase(notebook_for_llm, phase)
            notebook_text = format_notebook_for_llm(notebook_for_llm)
        else:
            notebook_text = "(Investigation starting — no data collected yet)"
        system_prompt = load_prompt_template(system_prompt_template, notebook=notebook_text)
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
    model_config: Any | None = None,
) -> Any:
    """Create the Orchestrator via create_react_agent.

    Uses build_orchestrator_prompt to create the prompt callable.
    Does NOT pass state_schema — create_react_agent's default AgentState
    includes 'remaining_steps' which is required by the framework.

    Returns a CompiledGraph (langgraph).
    """
    llm_kwargs: dict[str, Any] = {"model": config.model, "temperature": config.temperature}
    if model_config is not None:
        if hasattr(model_config, "api_key") and model_config.api_key:
            llm_kwargs["api_key"] = model_config.api_key
        if hasattr(model_config, "base_url") and model_config.base_url:
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

    # Orchestrator message compression via pre_model_hook
    if config.compression is not None and config.compression.enabled:
        agent_kwargs["pre_model_hook"] = build_compression_hook(config.compression)

    return create_react_agent(**agent_kwargs)
