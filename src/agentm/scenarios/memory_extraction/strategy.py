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
from agentm.scenarios.memory_extraction.formatters import format_memory_extraction_context
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
        return format_memory_extraction_context(state)

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
