"""Enumeration types for AgentM.

All enums use (str, Enum) pattern for JSON serialization compatibility.
These are normative definitions — field names and values are binding.
"""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """Diagnostic phase markers for the Orchestrator's Notebook."""

    EXPLORATION = "exploration"
    GENERATION = "generation"
    VERIFICATION = "verification"
    CONFIRMATION = "confirmation"


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis in the DiagnosticNotebook."""

    FORMED = "formed"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    REFINED = "refined"
    INCONCLUSIVE = "inconclusive"


class Verdict(str, Enum):
    """Three-value verdict for hypothesis verification results."""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    PARTIAL = "partial"


class AgentRunStatus(str, Enum):
    """Runtime status of a Sub-Agent execution (used by TaskManager)."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status of a dispatched task in PendingTask."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"


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
