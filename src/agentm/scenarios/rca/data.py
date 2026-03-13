"""RCA-specific data structures (dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from agentm.models.enums import AgentRunStatus
from agentm.scenarios.rca.enums import HypothesisStatus, Phase, Verdict


# --- Hypothesis & Verification ---


@dataclass
class Hypothesis:
    """A diagnostic hypothesis tracked in the DiagnosticNotebook."""

    id: str
    description: str
    evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.FORMED
    created_at: str = ""
    last_updated: str = ""


@dataclass
class VerificationResult:
    """Structured result from a Sub-Agent hypothesis verification."""

    verdict: Verdict
    report: str
    refined_description: Optional[str] = None


# --- Exploration & Agent Outcomes ---


@dataclass
class AgentOutcome:
    """Execution metadata for a Sub-Agent within an ExplorationStep."""

    agent_id: str
    task_id: str
    status: AgentRunStatus
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ExplorationStep:
    """Records each step across all phases in the DiagnosticNotebook."""

    step_number: int
    phase: Phase
    action: str
    timestamp: str
    content: str
    target_agents: Optional[list[str]] = None
    target_hypothesis_id: Optional[str] = None
    verdict: Optional[Verdict] = None
    confirmed_root_cause: Optional[str] = None
    agent_outcomes: Optional[dict[str, AgentOutcome]] = None


# --- PhaseSummary ---


@dataclass
class PhaseSummary:
    """Compressed summary of a completed diagnostic phase."""

    phase: str
    started_at: str
    completed_at: str
    key_data_collected: dict = field(default_factory=dict)
    actions_taken: list[str] = field(default_factory=list)
    decisions_made: list[str] = field(default_factory=list)
    hypotheses_affected: list[str] = field(default_factory=list)
    anomalies_noted: list[str] = field(default_factory=list)
    critical_evidence: list[dict] = field(default_factory=list)


# --- DiagnosticNotebook ---


@dataclass
class DiagnosticNotebook:
    """The Orchestrator's complete working memory."""

    task_id: str
    task_description: str
    start_time: str
    collected_data: dict[str, dict] = field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    hypothesis_verification_order: list[str] = field(default_factory=list)
    confirmed_hypothesis: Optional[str] = None
    exploration_history: list[ExplorationStep] = field(default_factory=list)
    current_phase: Phase = Phase.EXPLORATION
    current_step: int = 0
    phase_summaries: list[PhaseSummary] = field(default_factory=list)


# --- Feature Gates: Adversarial Review ---


@dataclass
class ChallengeResult:
    """Result from a Devil's Advocate adversarial review."""

    counter_arguments: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    overlooked_evidence: list[str] = field(default_factory=list)
    challenge_strength: Literal["weak", "moderate", "strong"] = "weak"
