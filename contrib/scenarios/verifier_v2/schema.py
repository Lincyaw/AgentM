"""Data contracts for the convergence loop verifier.

All typed structures that cross module boundaries live here:
gaps, tasks, evidence, verdicts, and the final output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# -- Gaps (output of evaluate_gaps) -------------------------------------------

GapKind = Literal[
    "unconfirmed_seed",
    "unreachable_seed",
    "unexplained_anomaly",
]


@dataclass(frozen=True, slots=True)
class Gap:
    """A structural invariant that the current graph does not satisfy."""

    kind: GapKind
    id: str
    source_seed: str | None
    target: str
    anomaly_id: str | None = None
    context: str = ""


@dataclass(slots=True)
class GapReport:
    """Result of evaluate_gaps: all unsatisfied invariants."""

    gaps: list[Gap]

    @property
    def satisfied(self) -> bool:
        return not self.gaps


# -- Verification tasks (output of plan_work) ---------------------------------

@dataclass(slots=True)
class VerificationTask:
    """A single unit of work: verify a seed or a hop."""

    kind: Literal["seed", "hop"]
    source_seed: str
    from_entity: str
    to_entity: str
    rel_type: str
    gap_ids: list[str]
    priority: tuple[Any, ...]
    context: str = ""
    max_retries: int = 3

    @property
    def edge_key(self) -> str:
        return f"{self.source_seed}::{self.from_entity}->{self.to_entity}"

    @property
    def is_critical(self) -> bool:
        """Seed confirmations and final hops to entry use majority voting."""
        return self.kind == "seed"


# -- Evidence dossier (searcher output) ----------------------------------------

class EvidenceItem(BaseModel):
    """One SQL query + explanation, re-executable against case data."""

    model_config = ConfigDict(extra="forbid")
    sql: str = Field(description="DuckDB SQL statement against case parquet views.")
    explanation: str = Field(description="What this query checks and what the result means.")


class EvidenceDossier(BaseModel):
    """Structured evidence submitted by the searcher. No verdict included."""

    model_config = ConfigDict(extra="forbid")

    observed_relationship: str = Field(
        description="Describe the observed relationship between from_entity and "
        "to_entity. Can be: direct call, co-deployment, async messaging, "
        "shared infrastructure, temporal correlation, or other. Free-form."
    )
    relationship_queries: list[EvidenceItem] = Field(
        default_factory=list,
        description="SQL evidence supporting the claimed relationship.",
    )

    target_observations: list[EvidenceItem] = Field(
        default_factory=list,
        description="All SQL queries examining target entity behavior "
        "(normal vs abnormal window).",
    )

    control_observations: list[EvidenceItem] = Field(
        default_factory=list,
        description="SQL queries examining control/comparison paths that "
        "are NOT on the fault propagation chain.",
    )

    counter_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence that could REFUTE propagation. Required field. "
        "Actively search for reasons the change might not be caused by the "
        "upstream fault (workload shift, other faults, proportional control "
        "change, timing mismatch).",
    )

    modalities_checked: list[str] = Field(
        default_factory=list,
        description="Which data sources were examined: traces, metrics, logs.",
    )
    modalities_unavailable: list[str] = Field(
        default_factory=list,
        description="Data sources that were attempted but not available.",
    )

    affected_endpoints: list[str] = Field(
        default_factory=list,
        description="Specific endpoints/spans affected, if the fault is "
        "endpoint-scoped rather than service-wide.",
    )


# -- Compiled dossier (compiler output, input to judge) ------------------------

@dataclass(slots=True)
class SQLResult:
    """A re-executed SQL query with verified results."""

    location: str
    sql: str
    explanation: str
    success: bool
    row_count: int = 0
    sample_values: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass(slots=True)
class CoverageGap:
    """A specific deficiency in the evidence dossier."""

    category: str  # "modality", "counter_evidence", "control", "sql_error"
    description: str


@dataclass(slots=True)
class CompiledDossier:
    """System-verified evidence ready for the judge. All SQL re-executed."""

    task: VerificationTask
    relationship_results: list[SQLResult]
    target_results: list[SQLResult]
    control_results: list[SQLResult]
    counter_results: list[SQLResult]
    observed_relationship: str
    affected_endpoints: list[str]
    modalities_checked: list[str]
    coverage_gaps: list[CoverageGap]

    @property
    def has_critical_gaps(self) -> bool:
        """Critical only when there's essentially no usable evidence at all."""
        successful_target = [r for r in self.target_results if r.success and r.row_count > 0]
        if not successful_target:
            return True
        return False

    @property
    def gap_feedback(self) -> str:
        if not self.coverage_gaps:
            return ""
        lines = ["Coverage gaps found in your evidence:"]
        for gap in self.coverage_gaps:
            lines.append(f"- [{gap.category}] {gap.description}")
        return "\n".join(lines)


# -- Judge verdict -------------------------------------------------------------

SubAnswer = Literal["yes", "no", "insufficient_evidence"]


@dataclass(slots=True)
class JudgeAnswer:
    """Answer to one sub-question."""

    answer: SubAnswer
    rationale: str


class JudgeVerdict(BaseModel):
    """Structured output from the judge agent."""

    model_config = ConfigDict(extra="forbid")

    causal_path: Literal["yes", "no", "insufficient_evidence"] = Field(
        description="Does a causal relationship exist between from and to?"
    )
    causal_path_rationale: str = Field(
        description="Why: what evidence supports or refutes the causal link."
    )
    causal_path_type: str = Field(
        default="",
        description="Type of relationship if yes: direct_call, co_deployment, "
        "async, shared_infra, temporal_correlation, other.",
    )

    effect_aligned: Literal["yes", "no", "insufficient_evidence"] = Field(
        description="Did the target change in the abnormal window in a way "
        "aligned with the upstream fault?"
    )
    effect_rationale: str = Field(
        description="Why: what changes were observed (or not) on the target."
    )

    selective: Literal["yes", "no", "insufficient_evidence"] = Field(
        description="Is the target's change disproportionate relative to "
        "control paths?"
    )
    selective_rationale: str = Field(
        description="Why: how does target change compare to control change."
    )

    predicate: str = Field(
        default="",
        description="If confirmed: the failure mode classification "
        "(latency_degraded, error_rate_elevated, flow_interrupted, etc.).",
    )


# -- Computed verdict ----------------------------------------------------------

VerdictKind = Literal["confirmed", "rejected", "inconclusive"]


@dataclass(slots=True)
class Verdict:
    """Final computed verdict from judge answers."""

    kind: VerdictKind
    predicate: str = ""
    relationship_type: str = ""
    rationale: str = ""
    judge_verdict: JudgeVerdict | None = None
    affected_endpoints: list[str] = field(default_factory=list)

    @staticmethod
    def from_judge(jv: JudgeVerdict, affected_endpoints: list[str]) -> Verdict:
        answers = [jv.causal_path, jv.effect_aligned, jv.selective]
        if all(a == "yes" for a in answers):
            kind: VerdictKind = "confirmed"
        elif any(a == "no" for a in answers):
            kind = "rejected"
        else:
            kind = "inconclusive"

        rationale_parts = []
        if jv.causal_path_rationale:
            rationale_parts.append(f"Relationship: {jv.causal_path_rationale}")
        if jv.effect_rationale:
            rationale_parts.append(f"Effect: {jv.effect_rationale}")
        if jv.selective_rationale:
            rationale_parts.append(f"Selectivity: {jv.selective_rationale}")

        return Verdict(
            kind=kind,
            predicate=jv.predicate if kind == "confirmed" else "",
            relationship_type=jv.causal_path_type if kind == "confirmed" else "",
            rationale=" | ".join(rationale_parts),
            judge_verdict=jv,
            affected_endpoints=affected_endpoints,
        )


# -- Task attempt history (for retry feedback) ---------------------------------

@dataclass(slots=True)
class TaskAttempt:
    """Record of one search-compile-judge cycle."""

    attempt_n: int
    coverage_gaps: list[str]
    verdict_kind: VerdictKind | None
    judge_rationale: str = ""
    sql_summary: list[str] = field(default_factory=list)


# -- Final output --------------------------------------------------------------

class PropagationResult(BaseModel):
    """Output of the entire verification workflow for one case."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    confirmed_seeds: list[str]
    gaps_satisfied: bool
    total_rounds: int
    total_agent_calls: int
    exhausted_edges: list[str] = Field(default_factory=list)
    rejected_edges: list[str] = Field(default_factory=list)
    remaining_gaps: list[dict[str, Any]] = Field(default_factory=list)
