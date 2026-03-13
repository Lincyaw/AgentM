"""Memory-extraction-specific enumeration types."""

from __future__ import annotations

from enum import Enum


class KnowledgeCategory(str, Enum):
    """Top-level classification of knowledge entries."""

    FAILURE_PATTERN = "failure_pattern"
    DIAGNOSTIC_SKILL = "diagnostic_skill"
    SYSTEM_KNOWLEDGE = "system_knowledge"


class KnowledgeConfidence(str, Enum):
    """Confidence level of a knowledge entry, determined by evidence strength.

    Memory Agent assigns this during extraction based on how the knowledge
    was established:
    - FACT: Verified causal relationship (hypothesis confirmed through RCA)
    - PATTERN: Observed correlation across multiple trajectories
    - HEURISTIC: Inferred strategy or rule of thumb from diagnostic experience
    """

    FACT = "fact"
    PATTERN = "pattern"
    HEURISTIC = "heuristic"
