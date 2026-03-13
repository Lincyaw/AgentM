"""RCA-specific enumeration types."""

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
