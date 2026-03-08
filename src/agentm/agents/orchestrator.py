"""Orchestrator graph creation.

Builds the Orchestrator's CompiledGraph via create_react_agent with
a prompt callable that injects the DiagnosticNotebook into LLM context.

All functions are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any, Callable

from agentm.config.schema import OrchestratorConfig


def build_orchestrator_prompt(system_prompt_template: str) -> Callable:
    """Build a prompt callable that injects Notebook into LLM context before each call.

    The returned callable receives the current ExecutorState and returns a list of
    messages with the system prompt populated from the DiagnosticNotebook.
    """
    raise NotImplementedError


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
    raise NotImplementedError
