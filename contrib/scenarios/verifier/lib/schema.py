"""Data contracts for the propagation workflow.

Discovery results (seed/hop), gate decisions, audit reports, and the
parallel-result type guards live here so the workflow module is left with
orchestration logic only.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Required, TypedDict

from pydantic import BaseModel, ConfigDict, Field


# -- Discovery results -----------------------------------------------------
class Injection(TypedDict, total=False):
    target: Required[str]
    chaos_type: Required[str]
    params: str
    node_id: str
    subject: str
    target_entity: str
    effect_target: str
    edge_source: str
    edge_target: str


HopLogEntry = TypedDict(
    "HopLogEntry",
    {
        "round": Required[int],
        "from": Required[str],
        "to": Required[str],
        "verdict": Required[str],
        "source_seed": str,
        "obligation_id": str,
    },
    total=False,
)


class CandidateEdge(TypedDict, total=False):
    source_seed: Required[str]
    from_service: Required[str]
    to_service: Required[str]
    rel_type: Required[str]
    source: str
    reason: str
    anomaly_ids: list[str]


class SeedResult(TypedDict, total=False):
    verdict: Required[str]
    effect_target: str | None
    predicate: str | None
    rationale: str
    evidence: list[dict[str, Any]]
    claim: str
    gate: dict[str, Any]
    _error: str


class HopResult(TypedDict, total=False):
    verdict: Required[str]
    predicate: str | None
    rationale: str
    evidence: list[dict[str, Any]]
    relationship: dict[str, Any] | None
    claim: str
    gate: dict[str, Any]
    _error: str


class ExecutionError(TypedDict):
    stage: str
    item: str
    reason: str


class FinalCheckIssue(TypedDict, total=False):
    check: Required[str]
    item: Required[str]
    reason: Required[str]
    details: dict[str, Any]


class FinalCheckReport(TypedDict):
    passed: bool
    entry_services: list[str]
    frontend_services: list[str]
    seed_reachability: dict[str, list[list[str]]]
    frontend_anomalies: list[dict[str, Any]]
    resolved_frontend_anomalies: list[dict[str, Any]]
    unexplained_frontend_anomalies: list[dict[str, Any]]
    issues: list[FinalCheckIssue]


# -- Gate ------------------------------------------------------------------
class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool = Field(
        description="True when the seed/hop investigation is complete enough "
        "for reduction."
    )
    retryable: bool = Field(
        default=False,
        description="True when rerunning the same seed/hop with focused "
        "feedback can plausibly repair the missing investigation.",
    )
    missing_checks: list[str] = Field(default_factory=list)
    retry_prompt: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    rationale: str = ""


# -- Audit rework / coverage ----------------------------------------------
class SeedRecheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["seed_recheck"]
    seed: str
    context: str = ""


class HopRecheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["hop_recheck"]
    from_service: str
    to_service: str
    rel_type: str | None = None
    source_seed: str | None = None
    context: str = ""
    obligation_id: str | None = None
    obligation_kind: Literal["seed_reachability", "frontend_anomaly"] | None = None
    target_frontend: str | None = None
    anomaly_id: str | None = None
    anomaly_component: str | None = None


ReworkRequest = Annotated[
    SeedRecheckRequest | HopRecheckRequest,
    Field(discriminator="kind"),
]


SeedCoverageStatus = Literal[
    "explains_entry",
    "local_only",
    "benign_or_no_effect",
    "needs_recheck",
    "invalid_path",
]


# -- Pass 1: seed -> entry reachability -----------------------------------
class SeedReachabilityReport(BaseModel):
    """One confirmed seed: does it have a valid causal path to entry?

    The reachability pass never drops edges itself — when a path is
    invalid or unprovable it emits a ``hop_recheck`` for the weakest edge,
    and the re-dispatched (gated) hop verdict decides whether the edge
    survives.
    """

    model_config = ConfigDict(extra="forbid")

    seed: str
    coverage: SeedCoverageStatus
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


# -- Pass 2: entry alarm -> seed coverage ---------------------------------
class AnomalyCoverageReport(BaseModel):
    """One entry/SLO scope: are its meaningful anomalies explained by the
    candidate graph? Gaps become re-dispatch requests, not edge drops."""

    model_config = ConfigDict(extra="forbid")

    scope: str
    meaningful_anomalies: list[str] = Field(default_factory=list)
    explained: list[str] = Field(default_factory=list)
    unexplained: list[str] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


# -- Free exploration (last step of each audit round) ---------------------
class ExploreReport(BaseModel):
    """Free completeness sweep over the whole dashboard. May surface
    services/anomalies the targeted passes never investigated. Its proposed
    graph changes are expressed as re-dispatch requests (incl. exploratory
    edges) so every addition is still verified by a gated discovery agent."""

    model_config = ConfigDict(extra="forbid")

    findings: list[str] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


# -- Deterministic audit outcome (no LLM reducer) -------------------------
class AuditOutcome(BaseModel):
    """Harness-computed summary of the audit loop. ``accepted`` is the
    fixpoint decision (a full round produced no re-dispatch and no
    unexplained anomalies), not an LLM verdict."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool = False
    rounds: int = 0
    seed_coverage: dict[str, SeedCoverageStatus] = Field(default_factory=dict)
    unexplained_anomalies: list[str] = Field(default_factory=list)
    rationale: str = ""


# -- Workflow output -------------------------------------------------------
class PropagationResult(TypedDict, total=False):
    nodes: Required[list[dict[str, Any]]]
    edges: Required[list[dict[str, Any]]]
    verdicts: Required[dict[str, HopResult]]
    hop_log: Required[list[HopLogEntry]]
    rounds: Required[int]
    unreachable_seeds: list[str]
    reachability_warnings: list[str]
    seed_verdicts: dict[str, SeedResult]
    confirmed_seeds: list[str]
    gate_log: list[dict[str, Any]]
    audit: dict[str, Any]
    audit_rounds: list[dict[str, Any]]
    execution_errors: list[ExecutionError]
    anomaly_inventory: list[dict[str, Any]]
    candidate_edges: list[CandidateEdge]
    node_attribution: dict[str, list[dict[str, Any]]]
    edge_attribution: list[dict[str, Any]]
    review_notes: list[dict[str, Any]]
    final_checks: FinalCheckReport
    final_anomaly_resolutions: dict[str, dict[str, Any]]
