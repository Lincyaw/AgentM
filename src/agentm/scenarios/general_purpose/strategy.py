"""General-purpose task execution strategy.

Provides a single self-looping 'execute' phase suitable for open-ended
tasks that do not follow a fixed multi-phase pipeline.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agentm.models.data import PhaseDefinition, ScenarioToolBundle
from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer
from agentm.scenarios.general_purpose.state import GeneralPurposeState

_MAX_FACTS = 50
_KEEP_FACTS = 30
_MAX_SKILL_CONTEXT_CHARS = 8000
_MAX_DISPLAYED_FACTS = 20

_GP_PHASES: dict[str, PhaseDefinition] = {
    "execute": PhaseDefinition(
        name="execute",
        description="Open-ended task execution with optional skill augmentation",
        handler=None,
        next_phases=["execute"],
    ),
}


class GeneralPurposeStrategy:
    """ReasoningStrategy implementation for general-purpose task execution."""

    @property
    def name(self) -> str:
        return "general_purpose"

    def initial_state(
        self, task_id: str, task_description: str
    ) -> GeneralPurposeState:
        return GeneralPurposeState(
            messages=[HumanMessage(content=task_description)],
            task_id=task_id,
            task_description=task_description,
            current_phase="execute",
            active_skills=[],
            skill_cache={},
            conversation_facts=[],
            structured_response=None,
        )

    def format_context(self, state: GeneralPurposeState) -> str:
        """Render active skills and conversation facts for the system prompt.

        Active skills are shown most-recently-loaded first, capped at 8000
        chars total.  Conversation facts show the last 20 entries.
        """
        lines: list[str] = []

        active_skills: list[str] = state.get("active_skills", [])
        skill_cache: dict[str, str] = state.get("skill_cache", {})

        if active_skills and skill_cache:
            lines.append("## Active Skills")
            total_chars = 0
            for path in reversed(active_skills):
                body = skill_cache.get(path, "")
                if not body:
                    continue
                remaining = _MAX_SKILL_CONTEXT_CHARS - total_chars
                if remaining <= 0:
                    lines.append("  (skill context limit reached)")
                    break
                truncated = body[:remaining]
                lines.append(f"### {path}")
                lines.append(truncated)
                total_chars += len(truncated)
            lines.append("")

        conversation_facts: list[str] = state.get("conversation_facts", [])
        if conversation_facts:
            displayed = conversation_facts[-_MAX_DISPLAYED_FACTS:]
            lines.append(f"## Conversation Facts ({len(conversation_facts)} total)")
            for i, fact in enumerate(displayed, 1):
                lines.append(f"  {i}. {fact}")
            if len(conversation_facts) > _MAX_DISPLAYED_FACTS:
                lines.append(
                    f"  ... {len(conversation_facts) - _MAX_DISPLAYED_FACTS} earlier facts omitted"
                )
            lines.append("")

        if not lines:
            return "(General-purpose execution starting -- no context yet)"

        return "\n".join(lines)

    def phase_definitions(self) -> dict[str, PhaseDefinition]:
        return dict(_GP_PHASES)

    def should_terminate(self, state: GeneralPurposeState) -> bool:
        # General-purpose tasks do not self-terminate; the orchestrator
        # decides when to stop via max_rounds or explicit completion.
        return False

    def compress_state(
        self, state: GeneralPurposeState, completed_phase: str
    ) -> GeneralPurposeState:
        """Trim conversation_facts if they exceed the threshold."""
        facts: list[str] = state.get("conversation_facts", [])
        if len(facts) <= _MAX_FACTS:
            return state
        trimmed = facts[-_KEEP_FACTS:]
        return {**state, "conversation_facts": trimmed}

    def get_answer_schemas(self) -> dict[str, type[BaseModel]]:
        return {"execute": GeneralAnswer}

    def get_output_schema(self) -> type[BaseModel] | None:
        return None

    def state_schema(self) -> type[GeneralPurposeState]:
        return GeneralPurposeState

    def create_scenario_tools(self, **kwargs: Any) -> ScenarioToolBundle:
        """Create skill management tools with closure over the vault instance."""
        vault = kwargs.get("vault")
        if vault is None:
            return ScenarioToolBundle()

        from agentm.scenarios.general_purpose.skill_tools import create_skill_tools

        skill_tools = create_skill_tools(vault)
        return ScenarioToolBundle(orchestrator_tools=skill_tools)
