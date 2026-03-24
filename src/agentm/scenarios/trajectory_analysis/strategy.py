"""Skill-driven trajectory analysis strategy.

Provides a two-phase pipeline (analyze -> synthesize) where analysis
behavior is determined by vault-hosted skills rather than code.
Supports both model-driven (catalog) and config-driven (pre-loaded)
skill activation modes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agentm.models.data import OrchestratorHooks, PhaseDefinition
from agentm.scenarios.trajectory_analysis.data import SkillCatalogEntry
from agentm.scenarios.trajectory_analysis.state import TrajectoryAnalysisState

if TYPE_CHECKING:
    from agentm.tools.vault.store import MarkdownVault

logger = logging.getLogger(__name__)

_SKILL_BASE_PATH = "skill/trajectory-analysis"

_TA_PHASES: dict[str, PhaseDefinition] = {
    "analyze": PhaseDefinition(
        name="analyze",
        description=(
            "Activate skill, read trajectories, dispatch workers, "
            "collect findings"
        ),
        handler=None,
        next_phases=["synthesize"],
    ),
    "synthesize": PhaseDefinition(
        name="synthesize",
        description="Aggregate findings, produce output, write artifacts",
        handler=None,
        next_phases=[],
    ),
}


class TrajectoryAnalysisStrategy:
    """ReasoningStrategy for skill-driven trajectory analysis.

    At init time:
    - Scans vault for skills under skill/trajectory-analysis/
    - Reads SKILL.md frontmatter (name + description) for the catalog
    - If a skill name is provided, pre-loads its body (config-driven mode)

    At runtime:
    - format_context() injects either the catalog or the pre-loaded skill
    - should_terminate() returns True when current_phase == "synthesize"
    """

    def __init__(
        self,
        vault: MarkdownVault | None = None,
        skill_name: str | None = None,
    ) -> None:
        self._vault = vault
        self._skill_catalog: list[SkillCatalogEntry] = []
        self._preloaded_skill_name: str | None = None
        self._preloaded_skill_body: str | None = None

        if vault is not None:
            self._skill_catalog = self._discover_skills(vault)

            if skill_name is not None:
                self._preload_skill(vault, skill_name)

    @property
    def name(self) -> str:
        return "trajectory_analysis"

    def initial_state(
        self, task_id: str, task_description: str
    ) -> TrajectoryAnalysisState:
        return TrajectoryAnalysisState(
            messages=[HumanMessage(content=task_description)],
            task_id=task_id,
            task_description=task_description,
            current_phase="analyze",
            source_trajectories=[],
            skill_name=self._preloaded_skill_name or "",
            analysis_results=[],
            structured_output=None,
        )

    def format_context(self, state: TrajectoryAnalysisState) -> str:
        """Render skill context, feedback, and source trajectories for the system prompt.

        Two modes:
        - Config-driven (pre-loaded skill): renders <active_skill> XML
          with the skill body.
        - Model-driven (no pre-loaded skill): renders <available_skills>
          XML catalog for model selection.

        Appends source trajectories and evaluation feedback.
        """
        lines: list[str] = []

        if self._preloaded_skill_body is not None:
            lines.append(self._format_config_driven_section())
        elif self._skill_catalog:
            lines.append(self._format_model_driven_section())
        else:
            lines.append(
                "(No skills discovered -- ensure vault contains "
                f"skills under {_SKILL_BASE_PATH}/)"
            )

        source_trajectories: list[str] = state.get(
            "source_trajectories", []
        )
        if source_trajectories:
            lines.append("")
            lines.append("<source_trajectories>")
            for tid in source_trajectories:
                lines.append(f"  <thread_id>{tid}</thread_id>")
            lines.append("</source_trajectories>")
        else:
            lines.append("")
            lines.append("(No source trajectories provided yet)")

        feedback: str = state.get("feedback", "")
        if feedback:
            lines.append("")
            lines.append("<evaluation_feedback>")
            lines.append(feedback)
            lines.append("</evaluation_feedback>")

        return "\n".join(lines)

    def phase_definitions(self) -> dict[str, PhaseDefinition]:
        return dict(_TA_PHASES)

    def should_terminate(self, state: TrajectoryAnalysisState) -> bool:
        return state.get("current_phase") == "synthesize"

    def compress_state(
        self, state: TrajectoryAnalysisState, completed_phase: str
    ) -> TrajectoryAnalysisState:
        """No compression -- pass through unchanged."""
        return state

    def get_answer_schemas(self) -> dict[str, type[BaseModel]]:
        from agentm.scenarios.trajectory_analysis.answer_schemas import (
            AnalyzeAnswer,
        )

        return {"analyze": AnalyzeAnswer}

    def get_output_schema(self) -> type[BaseModel] | None:
        return None

    def state_schema(self) -> type[TrajectoryAnalysisState]:
        return TrajectoryAnalysisState

    def orchestrator_hooks(self) -> OrchestratorHooks:
        return OrchestratorHooks()

    # ------------------------------------------------------------------
    # Skill discovery and pre-loading
    # ------------------------------------------------------------------

    def _discover_skills(
        self, vault: MarkdownVault
    ) -> list[SkillCatalogEntry]:
        """Scan vault for trajectory analysis skills.

        Reads SKILL.md frontmatter (name + description) for each skill
        directory under skill/trajectory-analysis/.
        """
        entries: list[SkillCatalogEntry] = []
        notes = vault.list_notes(path=_SKILL_BASE_PATH, depth=1)
        for note in notes:
            note_path = note["path"]
            skill_md_path = f"{note_path}/SKILL.md"
            skill_md = vault.read(skill_md_path)
            if skill_md is None:
                continue
            fm: dict[str, Any] = skill_md.get("frontmatter", {})
            skill_name = fm.get("name", "")
            description = fm.get("description", "")
            if not skill_name:
                logger.warning(
                    "Skipping skill at %s: missing 'name' in frontmatter",
                    skill_md_path,
                )
                continue
            entries.append(
                SkillCatalogEntry(
                    name=skill_name,
                    description=description,
                    path=note_path,
                )
            )
        return entries

    def _preload_skill(self, vault: MarkdownVault, skill_name: str) -> None:
        """Pre-load a specific skill's body for config-driven mode."""
        skill_md_path = f"{_SKILL_BASE_PATH}/{skill_name}/SKILL.md"
        skill_md = vault.read(skill_md_path)
        if skill_md is None:
            logger.warning(
                "Config-driven skill '%s' not found at %s",
                skill_name,
                skill_md_path,
            )
            return
        self._preloaded_skill_name = skill_name
        self._preloaded_skill_body = skill_md.get("body", "")

    # ------------------------------------------------------------------
    # Context rendering helpers
    # ------------------------------------------------------------------

    def _format_model_driven_section(self) -> str:
        """Render <available_skills> XML catalog for model-driven mode."""
        lines: list[str] = []
        lines.append("<skill_usage>")
        lines.append(
            "1. Read the task description to understand what analysis "
            "is needed."
        )
        lines.append(
            "2. Review the available skills and pick the best match."
        )
        lines.append(
            '3. Load the skill: vault_read(path="<skill_path>/SKILL.md")'
        )
        lines.append(
            "4. Follow the skill's workflow. Load references via "
            "vault_read as directed."
        )
        lines.append("</skill_usage>")
        lines.append("")
        lines.append("<available_skills>")
        for entry in self._skill_catalog:
            lines.append("<skill>")
            lines.append(f"  <name>{entry.name}</name>")
            lines.append(f"  <description>{entry.description}</description>")
            lines.append(f"  <path>{entry.path}</path>")
            lines.append("</skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def _format_config_driven_section(self) -> str:
        """Render <active_skill> XML for config-driven mode."""
        assert self._preloaded_skill_name is not None
        assert self._preloaded_skill_body is not None
        lines: list[str] = []
        lines.append(
            f'<active_skill name="{self._preloaded_skill_name}">'
        )
        lines.append(self._preloaded_skill_body)
        lines.append("")
        lines.append(
            f"Skill directory: {_SKILL_BASE_PATH}/"
            f"{self._preloaded_skill_name}"
        )
        lines.append(
            "Relative paths in this skill resolve against the skill "
            "directory."
        )
        lines.append("</active_skill>")
        return "\n".join(lines)
