"""Hypothesis-driven RCA strategy.

Consolidates logic previously spread across:
- ``core/context_formatters.py``  (format_rca_context)
- ``core/notebook.py``            (should_compress_phase)
- ``core/compression.py``         (compress_completed_phase)
- ``models/answer_schemas.py``    (ANSWER_SCHEMA subset)
- ``core/state_registry.py``      (STATE_SCHEMAS["hypothesis_driven"])
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agentm.models.data import OrchestratorHooks, PhaseDefinition, ScenarioToolBundle
from agentm.scenarios.rca.answer_schemas import (
    DeepAnalyzeAnswer,
    ScoutAnswer,
    VerifyAnswer,
)
from agentm.scenarios.rca.compression import compress_completed_phase
from agentm.scenarios.rca.data import DiagnosticNotebook
from agentm.scenarios.rca.enums import Phase
from agentm.scenarios.rca.notebook import format_notebook_for_llm, should_compress_phase
from agentm.scenarios.rca.state import HypothesisDrivenState


_RCA_PHASES: dict[str, PhaseDefinition] = {
    "exploration": PhaseDefinition(
        name="exploration",
        description="Initial data collection and incident scoping",
        handler=None,
        next_phases=["generation"],
    ),
    "generation": PhaseDefinition(
        name="generation",
        description="Hypothesis generation from collected evidence",
        handler=None,
        next_phases=["verification"],
    ),
    "verification": PhaseDefinition(
        name="verification",
        description="Hypothesis testing and evidence evaluation",
        handler=None,
        next_phases=["confirmation", "generation"],
    ),
    "confirmation": PhaseDefinition(
        name="confirmation",
        description="Root cause confirmed; final reporting",
        handler=None,
        next_phases=[],
    ),
}


class HypothesisDrivenStrategy:
    """ReasoningStrategy implementation for hypothesis-driven RCA."""

    @property
    def name(self) -> str:
        return "hypothesis_driven"

    def initial_state(
        self, task_id: str, task_description: str
    ) -> HypothesisDrivenState:
        return HypothesisDrivenState(
            messages=[HumanMessage(content=task_description)],
            task_id=task_id,
            task_description=task_description,
            current_phase=Phase.EXPLORATION.value,
            notebook=DiagnosticNotebook(
                task_id=task_id,
                task_description=task_description,
                start_time="",
            ),
            current_hypothesis=None,
            compression_refs=[],
            structured_response=None,
        )

    def format_context(self, state: HypothesisDrivenState) -> str:
        """Format notebook for LLM system prompt with phase compression."""
        notebook = state.get("notebook")
        if notebook is None:
            return "(Investigation starting — no data collected yet)"

        notebook_for_llm = notebook
        for phase in ("exploration", "generation", "verification"):
            if should_compress_phase(notebook_for_llm, phase):
                notebook_for_llm = compress_completed_phase(notebook_for_llm, phase)

        return format_notebook_for_llm(notebook_for_llm)

    def phase_definitions(self) -> dict[str, PhaseDefinition]:
        return dict(_RCA_PHASES)

    def should_terminate(self, state: HypothesisDrivenState) -> bool:
        notebook = state.get("notebook")
        if notebook is None:
            return False
        return notebook.confirmed_hypothesis is not None

    def compress_state(
        self, state: HypothesisDrivenState, completed_phase: str
    ) -> HypothesisDrivenState:
        notebook = state.get("notebook")
        if notebook is None:
            return state
        compressed = compress_completed_phase(notebook, completed_phase)
        return {**state, "notebook": compressed}

    def get_answer_schemas(self) -> dict[str, type[BaseModel]]:
        return {
            "scout": ScoutAnswer,
            "deep_analyze": DeepAnalyzeAnswer,
            "verify": VerifyAnswer,
        }

    def get_output_schema(self) -> type[BaseModel] | None:
        return None

    def state_schema(self) -> type[HypothesisDrivenState]:
        return HypothesisDrivenState

    def orchestrator_hooks(self) -> OrchestratorHooks:
        return OrchestratorHooks(
            think_stall_enabled=True,
            think_stall_limit=3,
            think_stall_warning=(
                "THINK-STALL WARNING: You have called only `think` for the "
                "last {streak} rounds without taking any action. "
                "Thinking does not advance the investigation.\n\n"
                "You MUST call an action tool NOW — dispatch_agent, "
                "update_hypothesis, or finalize with "
                "<decision>finalize</decision>.\n"
                "Do NOT call think again until you have taken an action."
            ),
            skip_context_on_think_only=True,
        )

    def create_scenario_tools(self, **kwargs: Any) -> ScenarioToolBundle:
        """Create RCA-specific tools: hypothesis management + service profiles.

        Accepts ``trajectory`` and ``vault`` keyword arguments from the builder.
        """
        from agentm.scenarios.rca.formatters import format_rca_context
        from agentm.scenarios.rca.hypothesis_store import HypothesisStore
        from agentm.scenarios.rca.service_profile import ServiceProfileStore
        from agentm.scenarios.rca.tools import create_rca_tools

        trajectory = kwargs.get("trajectory")
        profile_store = ServiceProfileStore()
        hypothesis_store = HypothesisStore()
        rca_tools = create_rca_tools(
            trajectory=trajectory,
            profile_store=profile_store,
            hypothesis_store=hypothesis_store,
        )

        worker_profile_tools = rca_tools.pop("_worker_profile_tools", [])
        format_context_fn = partial(
            format_rca_context,
            profile_store=profile_store,
            hypothesis_store=hypothesis_store,
        )

        return ScenarioToolBundle(
            orchestrator_tools=rca_tools,
            worker_tools=worker_profile_tools,
            format_context_override=format_context_fn,
        )
