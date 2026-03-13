"""Memory-extraction strategy.

Consolidates logic from:
- ``core/context_formatters.py``  (format_memory_extraction_context)
- ``models/answer_schemas.py``    (ANSWER_SCHEMA subset)
- ``core/state_registry.py``      (STATE_SCHEMAS["memory_extraction"])
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agentm.models.data import PhaseDefinition
from agentm.scenarios.memory_extraction.answer_schemas import (
    AnalyzeAnswer,
    CollectAnswer,
    ExtractAnswer,
    RefineAnswer,
)
from agentm.scenarios.memory_extraction.state import MemoryExtractionState


_EXTRACTION_PHASES: dict[str, PhaseDefinition] = {
    "collect": PhaseDefinition(
        name="collect",
        description="Read and summarize source trajectories",
        handler=None,
        next_phases=["analyze"],
    ),
    "analyze": PhaseDefinition(
        name="analyze",
        description="Extract patterns from collected trajectories",
        handler=None,
        next_phases=["extract"],
    ),
    "extract": PhaseDefinition(
        name="extract",
        description="Produce knowledge entries from patterns",
        handler=None,
        next_phases=["refine"],
    ),
    "refine": PhaseDefinition(
        name="refine",
        description="Deduplicate, merge, and finalize knowledge entries",
        handler=None,
        next_phases=[],
    ),
}


class MemoryExtractionStrategy:
    """ReasoningStrategy implementation for cross-task knowledge extraction."""

    @property
    def name(self) -> str:
        return "memory_extraction"

    def initial_state(
        self, task_id: str, task_description: str
    ) -> MemoryExtractionState:
        return MemoryExtractionState(
            messages=[HumanMessage(content=task_description)],
            task_id=task_id,
            task_description=task_description,
            current_phase="collect",
            source_trajectories=[],
            extracted_patterns=[],
            knowledge_entries=[],
            existing_knowledge=[],
        )

    def format_context(self, state: MemoryExtractionState) -> str:
        """Render source trajectories, patterns, and knowledge as text."""
        lines: list[str] = []

        source_trajectories: list[str] = state.get("source_trajectories", [])
        if source_trajectories:
            lines.append("## Source Trajectories")
            for t in source_trajectories:
                lines.append(f"  - {t}")
            lines.append("")

        extracted_patterns: list[dict] = state.get("extracted_patterns", [])
        if extracted_patterns:
            lines.append(
                f"## Extracted Patterns ({len(extracted_patterns)} total)"
            )
            for i, p in enumerate(extracted_patterns[:20], 1):
                ptype = p.get("pattern_type", "unknown")
                desc = p.get("description", "")
                lines.append(f"  {i}. [{ptype}] {desc}")
            if len(extracted_patterns) > 20:
                lines.append(
                    f"  ... and {len(extracted_patterns) - 20} more"
                )
            lines.append("")

        knowledge_entries: list = state.get("knowledge_entries", [])
        if knowledge_entries:
            lines.append(
                f"## Knowledge Entries Queued ({len(knowledge_entries)} total)"
            )
            for entry in knowledge_entries[:10]:
                if hasattr(entry, "title"):
                    title = entry.title
                elif isinstance(entry, dict):
                    title = entry.get("title", str(entry))
                else:
                    title = str(entry)
                lines.append(f"  - {title}")
            if len(knowledge_entries) > 10:
                lines.append(
                    f"  ... and {len(knowledge_entries) - 10} more"
                )
            lines.append("")

        existing_knowledge: list = state.get("existing_knowledge", [])
        if existing_knowledge:
            lines.append(
                f"## Existing Knowledge ({len(existing_knowledge)} entries)"
            )
            for entry in existing_knowledge[:5]:
                if hasattr(entry, "title"):
                    title = entry.title
                elif isinstance(entry, dict):
                    title = entry.get("title", str(entry))
                else:
                    title = str(entry)
                lines.append(f"  - {title}")
            if len(existing_knowledge) > 5:
                lines.append(
                    f"  ... and {len(existing_knowledge) - 5} more"
                )
            lines.append("")

        if not lines:
            return "(Memory extraction starting — no data yet)"

        return "\n".join(lines)

    def phase_definitions(self) -> dict[str, PhaseDefinition]:
        return dict(_EXTRACTION_PHASES)

    def should_terminate(self, state: MemoryExtractionState) -> bool:
        return state.get("current_phase", "") == "refine"

    def compress_state(
        self, state: MemoryExtractionState, completed_phase: str
    ) -> MemoryExtractionState:
        # Memory extraction does not use phase compression.
        return state

    def get_answer_schemas(self) -> dict[str, type[BaseModel]]:
        return {
            "collect": CollectAnswer,
            "analyze": AnalyzeAnswer,
            "extract": ExtractAnswer,
            "refine": RefineAnswer,
        }

    def get_output_schema(self) -> type[BaseModel] | None:
        return None

    def state_schema(self) -> type[MemoryExtractionState]:
        return MemoryExtractionState
