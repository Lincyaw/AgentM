"""Orchestrator graph creation.

Builds the Orchestrator's CompiledGraph via create_react_agent with
a prompt callable that injects the DiagnosticNotebook into LLM context.
"""

from __future__ import annotations

from typing import Any, Callable

from agentm.config.schema import OrchestratorConfig


def build_orchestrator_prompt(system_prompt_template: str) -> Callable:
    """Build a prompt callable that injects Notebook into LLM context before each call.

    The returned callable receives the current ExecutorState and returns a list of
    messages with the system prompt populated from the DiagnosticNotebook.
    """
    from langchain_core.messages import SystemMessage

    from agentm.core.notebook import format_notebook_for_llm
    from agentm.core.prompt import load_prompt_template

    def prompt(state: dict) -> list:
        notebook_text = format_notebook_for_llm(state["notebook"])
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
) -> Any:
    """Create the Orchestrator via create_react_agent.

    Uses build_orchestrator_prompt to create the prompt callable,
    passes tools, and sets state_schema=ExecutorState.

    Returns a CompiledGraph (langgraph).
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from agentm.models.state import ExecutorState

    model = ChatOpenAI(model=config.model, temperature=config.temperature)
    prompt_callable = build_orchestrator_prompt(config.prompts.get("system", ""))

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt_callable,
        state_schema=ExecutorState,
        checkpointer=checkpointer,
        store=store,
    )
