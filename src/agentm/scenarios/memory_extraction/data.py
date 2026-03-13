"""Memory-extraction-specific data structures (dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agentm.scenarios.memory_extraction.enums import (
    KnowledgeCategory,
    KnowledgeConfidence,
)


@dataclass
class KnowledgeEvidence:
    """Evidence supporting a knowledge entry, from an RCA trajectory."""

    source_thread_id: str
    source_checkpoint_range: Optional[tuple[str, str]] = None
    relevant_data: dict = field(default_factory=dict)
    summary: str = ""


@dataclass
class KnowledgeEntry:
    """A single entry in the Knowledge Store."""

    id: str
    path: str
    category: KnowledgeCategory
    confidence: KnowledgeConfidence
    domain: str
    title: str
    description: str
    evidence: list[KnowledgeEvidence] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related_entries: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    frequency: int = 0
