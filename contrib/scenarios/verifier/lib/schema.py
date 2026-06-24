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
    {"round": int, "from": str, "to": str, "verdict": str},
)


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
    context: str = ""


ReworkRequest = Annotated[
    SeedRecheckRequest | HopRecheckRequest,
    Field(discriminator="kind"),
]


class EdgeDrop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    seed: str | None = Field(
        default=None,
        description="Seed lineage for the invalid path, when known.",
    )
    path_id: str | None = Field(
        default=None,
        description="Candidate path id that contains this edge, when known.",
    )
    reason: str = ""


SeedCoverageStatus = Literal[
    "explains_entry",
    "local_only",
    "benign_or_no_effect",
    "needs_recheck",
    "invalid_path",
]


class AnomalyAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: str
    meaningful_anomalies: list[str] = Field(default_factory=list)
    explained: list[str] = Field(default_factory=list)
    unexplained: list[str] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


class CausalAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str
    path: list[str] = Field(default_factory=list)
    verdict: Literal["valid", "invalid", "needs_recheck"] = "needs_recheck"
    invalid_reason: str | None = None
    weakest_edge: EdgeDrop | None = None
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


class SeedCoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: str
    coverage: SeedCoverageStatus
    explained_entry_observations: list[str] = Field(default_factory=list)
    local_effect_observations: list[str] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    rationale: str = ""


class GlobalAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    seed_coverage: dict[str, SeedCoverageStatus] = Field(default_factory=dict)
    unexplained_anomalies: list[str] = Field(default_factory=list)
    invalid_causal_paths: list[str] = Field(default_factory=list)
    drop_edges: list[EdgeDrop] = Field(default_factory=list)
    rework_requests: list[ReworkRequest] = Field(default_factory=list)
    stop_reason: str | None = None
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
