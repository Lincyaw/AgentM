"""Pass 3 — claim status fold: pure code over Pass 2 edges + coverage.

The model's contribution ended at Pass 2: local pairwise supports/
conflicts edges, each with a code-verified verbatim quote. This pass is
deterministic algebra (P6 — code owns the decidable):

* ``conflicted``  — a conflicts edge exists (dominates: one witnessed
  contradiction outweighs any number of supports);
* ``supported``   — otherwise a supports edge exists;
* ``unsourced``   — neither, AND the per-claim closed-world condition of
  SCHEMA §2.4 holds for THIS claim: content coverage (all partitions
  swept, none degraded), attention coverage (an explicit retrieval row
  in every partition), judgment coverage (every nomination verified).
  Still sound only relative to retrieval recall — a missed nomination
  fails toward a false negative on support, never toward a fabricated
  edge. ``evidence_empty`` marks the degenerate sweep over a trajectory
  whose serialization carries no observation content at all;
* ``unknown``     — neither edge, and some coverage condition failed for
  this claim — never escalates (P5): a failed verification call demotes
  exactly the claims it touched, never entitles a negative.

Statuses are totally ordered by information (SCHEMA §2.6):
unknown < unsourced < supported < conflicted. The fold is monotone in
(verified edges at c, coverage bits of c); every upstream failure
shrinks those inputs, so failures under-alarm, never over-alarm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .edges import EdgePassResult
from .index import ClaimFinding, TrajectoryIndex

_STATUS_ORDER = ("conflicted", "supported", "unsourced", "unknown")


@dataclass(slots=True)
class ClaimStatusAnalysis:
    findings: list[ClaimFinding] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.status] = out.get(f.status, 0) + 1
        return out

    def to_artifact(self) -> dict[str, Any]:
        return {
            "findings": [
                {
                    "claim_id": f.claim_id, "step_id": f.step_id,
                    "status": f.status, "edge_ids": list(f.edge_ids),
                    "evidence_empty": f.evidence_empty,
                }
                for f in self.findings
            ],
            "counts": self.counts(),
            "coverage": self.coverage,
        }


def fold_claim_statuses(
    index: TrajectoryIndex,
    edge_result: EdgePassResult,
    *,
    run_id: str = "",
) -> ClaimStatusAnalysis:
    """Fold per-claim evidence status from edges + the sweep's coverage.

    Deterministic; idempotent per run (``index.claim_findings`` for this
    run is replaced wholesale).
    """
    analysis = ClaimStatusAnalysis(coverage=edge_result.coverage.to_dict())

    edges_by_claim: dict[str, list[Any]] = {}
    for e in index.edges.values():
        if e.kind in ("supports", "conflicts") and (not run_id or e.run_id == run_id):
            edges_by_claim.setdefault(e.src, []).append(e)

    evidence_empty = edge_result.coverage.n_observation_steps == 0
    content_complete = edge_result.coverage.complete

    for claim in sorted(
        (c for c in index.claims.values() if not run_id or c.run_id == run_id),
        key=lambda c: c.step_id,
    ):
        edges = edges_by_claim.get(claim.id, [])
        conflict_ids = tuple(e.id for e in edges if e.kind == "conflicts")
        support_ids = tuple(e.id for e in edges if e.kind == "supports")
        # per-claim closed world: an empty sweep attends/judges vacuously;
        # a claim the edge pass never saw (no coverage record) is unknown
        cc = edge_result.claim_coverage.get(claim.id)
        attended = cc.attended if cc is not None else evidence_empty
        judged = cc.judged if cc is not None else evidence_empty
        if conflict_ids:
            status, edge_ids = "conflicted", conflict_ids + support_ids
        elif support_ids:
            status, edge_ids = "supported", support_ids
        elif content_complete and attended and judged:
            status, edge_ids = "unsourced", ()
        else:
            status, edge_ids = "unknown", ()
        analysis.findings.append(ClaimFinding(
            claim_id=claim.id,
            run_id=claim.run_id,
            step_id=claim.step_id,
            status=status,
            edge_ids=edge_ids,
            evidence_empty=evidence_empty,
        ))

    index.claim_findings = [
        f for f in index.claim_findings if run_id and f.run_id != run_id
    ] + analysis.findings

    logger.info("claim statuses: {} (evidence_empty={}, content complete={})",
                analysis.counts(), evidence_empty, content_complete)
    return analysis
