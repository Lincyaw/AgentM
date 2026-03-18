"""ReasoningStrategy protocol — the central abstraction for pluggable domain logic.

A ``ReasoningStrategy[S]`` encapsulates everything that makes one agent system
different from another: state initialization, phase definitions, context
formatting, termination logic, answer schemas, and scenario-specific tools.

Framework components (builder, middleware, task manager) are parameterized over
``S`` and delegate domain-specific behavior to the strategy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agentm.models.data import OrchestratorHooks, PhaseDefinition, ScenarioToolBundle
from agentm.models.state import S


@runtime_checkable
class ReasoningStrategy(Protocol[S]):
    """Protocol for pluggable reasoning strategies.

    Implementations provide domain-specific logic for an agent system type.
    The framework calls these methods at well-defined points during execution.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this strategy (e.g. 'hypothesis_driven')."""
        ...

    def initial_state(self, task_id: str, task_description: str) -> S:
        """Create the initial state for a new execution run."""
        ...

    def format_context(self, state: S) -> str:
        """Format state into a text block for the LLM system prompt.

        Replaces the per-system-type entries in FORMAT_CONTEXT_REGISTRY.
        Return an empty string if no context injection is needed.
        """
        ...

    def phase_definitions(self) -> dict[str, PhaseDefinition]:
        """Return the phase graph for this strategy.

        Keys are phase names (str), values are ``PhaseDefinition`` with
        allowed ``next_phases`` transitions.
        """
        ...

    def should_terminate(self, state: S) -> bool:
        """Check whether the execution should terminate.

        Called after each orchestrator step.  Returns ``True`` to stop.
        """
        ...

    def compress_state(self, state: S, completed_phase: str) -> S:
        """Compress state after a phase completes.

        Returns a new state with the completed phase's detailed records
        replaced by a summary.  Implementors should maintain immutability.
        """
        ...

    def get_answer_schemas(self) -> dict[str, type[BaseModel]]:
        """Map task-type names to Pydantic models for sub-agent structured output.

        Keys are task-type strings (e.g. 'scout', 'verify'); values are
        Pydantic model classes passed to ``create_react_agent(response_format=...)``.
        """
        ...

    def get_output_schema(self) -> type[BaseModel] | None:
        """Return the orchestrator-level output schema, if any.

        ``None`` means the orchestrator does not produce structured output.
        """
        ...

    def state_schema(self) -> type[S]:
        """Return the TypedDict class used as the LangGraph state schema."""
        ...

    def orchestrator_hooks(self) -> OrchestratorHooks:
        """Return orchestrator behavior customizations.

        Default: generic hooks suitable for most scenarios.
        Override to customize think-stall detection, context injection, etc.
        """
        ...


def get_scenario_tools(strategy: ReasoningStrategy[Any], **kwargs: Any) -> ScenarioToolBundle:
    """Call ``create_scenario_tools`` on a strategy if it implements the hook.

    This is an optional extension point: strategies that need scenario-specific
    tools (e.g. RCA's hypothesis tools, GP's skill tools) implement
    ``create_scenario_tools(**kwargs) -> ScenarioToolBundle``.  Strategies that
    don't need custom tools simply omit the method and get an empty bundle.

    The builder calls this function instead of branching on ``system_type``.
    """
    factory = getattr(strategy, "create_scenario_tools", None)
    if factory is not None:
        return factory(**kwargs)
    return ScenarioToolBundle()
