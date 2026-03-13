"""Memory-extraction-specific state schema."""

from __future__ import annotations

import operator
from typing import Annotated

from agentm.scenarios.memory_extraction.data import KnowledgeEntry
from agentm.models.state import BaseExecutorState


class MemoryExtractionState(BaseExecutorState):
    """Cross-task knowledge extraction state."""

    source_trajectories: list[str]
    extracted_patterns: Annotated[list[dict], operator.add]
    knowledge_entries: list[KnowledgeEntry]
    existing_knowledge: list[KnowledgeEntry]
