"""Pass 2 — edge construction over Pass 1 nodes.

Pass 1 emits nodes (symbols, claims, provenance regions); this pass
connects claims to the observation evidence they relate to. Generic over
trajectory types: it presumes only the node kinds, never a benchmark or
domain. The formal contract is SCHEMA.md §2; the shape is FEVER-like —
retrieval, then verification, then a pure-code fold (verification.py).

    partition (code)    deterministic char-budget partitions over ALL
                        observation steps, whole excerpts only; an
                        oversized step splits at observation-REGION
                        boundaries, and a region larger than the budget
                        marks its partition coverage-degraded (§2.2) —
                        a "successful" call over content the model may
                        not have seen must not entitle a negative
    retrieval (oracle)  per partition, nominate candidate evidence steps
                        for EVERY claim — no polarity, no quotes, an
                        easy listing task; the explicit per-claim row
                        (empty allowed) is the attestation of attention
                        (§2.4); sampled and unioned — recall lives here,
                        so sampling does too
    verify    (oracle)  per claim, one focused adversarial call over its
                        nominated excerpts only: supports / conflicts /
                        neutral, plus a verbatim witness quote for the
                        non-neutral verdicts
    gates     (code)    endpoints exist, relation valid, FULL quote
                        verbatim within a SINGLE observation region
                        (never spliced across region seams), and not the
                        claim's own words; failures are rejected and
                        logged, never stored (P2). Timeline is a FACT,
                        not a filter: evidence_position records
                        before/same/after the claim

Contracts: oracle judgments land in the transcript (P3); failures only
weaken — a failed retrieval breaks that partition's coverage, a failed
verification demotes exactly the claims it touched (recorded in
``claim_coverage``), and neither can fabricate an edge (P5); no
mid-content truncation anywhere (P4). Conflicts edges are never capped:
a code-side prune must not be able to strengthen a status (§2.6).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..ir.diagnostics import Diagnostics
from ..ir.models import Claim, Constraint, Edge, Step, stable_id
from ..oracle import SessionFactory, _ask_model, _index_by_id, _safe_float

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex

_PARTITION_CHAR_BUDGET = 60_000   # max excerpt chars per oracle call (whole excerpts)
_MAX_SUPPORT_EDGES_PER_CLAIM = 6  # supports capped in trajectory order; conflicts never
_N_RETRIEVAL_SAMPLES = 2          # retrieval samples per partition, unioned (unused after evidence merge)
_VERIFY_ATTEMPTS = 2              # attempts per verification call before giving up (unused after evidence merge)

_EDGE_KINDS = ("supports", "conflicts")
_VERDICTS = ("supports", "conflicts", "neutral")


@dataclass(slots=True)
class EdgeCoverage:
    """Content coverage: did the sweep see the whole evidence space?

    ``complete`` (SCHEMA §2.4 condition 1) requires every partition to
    have at least one successful retrieval sample AND no partition to be
    coverage-degraded (an observation region larger than the budget).
    Attention and judgment coverage are per claim — ``claim_coverage``
    on the result.
    """

    n_observation_steps: int = 0
    n_partitions: int = 0
    n_failed_partitions: int = 0
    n_degraded_partitions: int = 0

    @property
    def complete(self) -> bool:
        return self.n_failed_partitions == 0 and self.n_degraded_partitions == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observation_steps": self.n_observation_steps,
            "n_partitions": self.n_partitions,
            "n_failed_partitions": self.n_failed_partitions,
            "n_degraded_partitions": self.n_degraded_partitions,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class ClaimCoverage:
    """Per-claim closed-world record (SCHEMA §2.4 conditions 2 and 3).

    ``attended`` — every partition's retrieval carried an explicit row
    for this claim (at least one successful sample). ``judged`` — every
    step retrieval nominated for it received a verification verdict.
    """

    attended: bool
    judged: bool


@dataclass(slots=True)
class EdgePassResult:
    edges: list[Edge] = field(default_factory=list)
    n_claims: int = 0
    coverage: EdgeCoverage = field(default_factory=EdgeCoverage)
    claim_coverage: dict[str, ClaimCoverage] = field(default_factory=dict)
    # full oracle record, for inspection: what retrieval nominated per
    # claim, and every verification verdict INCLUDING neutral (an edge
    # artifact that only shows survivors cannot explain a miss)
    nominations: dict[str, list[str]] = field(default_factory=dict)
    verdicts: list[dict[str, str]] = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    # constraint evidence from the merged sweep (populated only when
    # constraints are passed to build_claim_edges); maps constraint_id →
    # {status, confidence, source, evidence_step_ids, reason}
    constraint_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "edges": [
                {
                    "id": e.id, "kind": e.kind, "src": e.src, "dst": e.dst,
                    "quote": e.quote, "evidence_position": e.evidence_position,
                }
                for e in self.edges
            ],
            "n_claims": self.n_claims,
            "coverage": self.coverage.to_dict(),
            "claim_coverage": {
                cid: {"attended": cc.attended, "judged": cc.judged}
                for cid, cc in self.claim_coverage.items()
            },
            "nominations": self.nominations,
            "verdicts": self.verdicts,
            "transcript": self.diagnostics.transcript,
            "prune_log": self.diagnostics.prune_log,
        }


# ---------------------------------------------------------------------------
# gates (code)
# ---------------------------------------------------------------------------


def _quote_in_regions(step: Step, quote: str) -> bool:
    """FULL-quote verbatim containment within a SINGLE observation region.

    The quote is the load-bearing witness for the edge — the entire
    passage must appear inside one region (whitespace-normalized), not a
    prefix of it and not spliced across a region seam: the joined
    segment would let a fabricated passage straddle two disjoint
    regions.
    """
    norm_quote = " ".join(quote.split())
    if not norm_quote:
        return False
    return any(
        norm_quote in " ".join(region.split())
        for region in step.observation_region_texts
    )


def _self_quote(claim_text: str, quote: str) -> bool:
    """Is the quote just the claim's own words? (whitespace/case-insensitive
    containment either way — the decidable core of "a claim is not its own
    evidence")."""
    norm_claim = " ".join(claim_text.lower().split())
    norm_quote = " ".join(quote.lower().split())
    if not norm_quote:
        return True
    return norm_quote in norm_claim or norm_claim in norm_quote


# ---------------------------------------------------------------------------
# partition (code)
# ---------------------------------------------------------------------------


def _step_chars(s: Step) -> int:
    return len(s.observation_segment or "")


def partition_by_budget[T](
    items: list[T], budget: int, size: Callable[[T], int],
) -> list[list[T]]:
    """Greedy in-order partition into whole-item groups under ``budget`` chars.

    A single item larger than the budget forms its own partition — never
    cut. ``size`` measures one item. Shared by the constraint sweep and
    the edge-pass verify stage (which size whole steps via ``_step_chars``)
    and the retrieval stage (which sizes region-level excerpts).
    """
    partitions: list[list[T]] = []
    current: list[T] = []
    total = 0
    for it in items:
        n = size(it)
        if current and total + n > budget:
            partitions.append(current)
            current, total = [], 0
        current.append(it)
        total += n
    if current:
        partitions.append(current)
    return partitions


@dataclass(frozen=True, slots=True)
class _Excerpt:
    """One retrieval-payload unit: a whole step's observation segment, or
    a single region of an oversized step (§2.2 — split at region
    boundaries, never mid-content)."""

    step: Step
    text: str
    oversized: bool   # a single region alone exceeds the budget


def _excerpts_for(step: Step, budget: int) -> list[_Excerpt]:
    segment = step.observation_segment or ""
    if len(segment) <= budget:
        return [_Excerpt(step, segment, False)]
    return [
        _Excerpt(step, region, len(region) > budget)
        for region in step.observation_region_texts
        if region.strip()
    ]


# ---------------------------------------------------------------------------
# retrieval (oracle): nominate candidate evidence, one row per claim
# ---------------------------------------------------------------------------

async def build_claim_edges(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
    constraints: list[Constraint] | None = None,
    commit_binding: str = "",
    commit_step_id: str = "",
) -> EdgePassResult:
    """Build ``supports``/``conflicts`` edges (claim ↔ observation evidence).

    Single-pass per partition: the model finds relevant excerpts for each
    claim AND judges polarity with verbatim quotes in one call. Idempotent
    per run: existing edges of these kinds are replaced wholesale.

    When *constraints* and *commit_binding* are provided, the same partition
    sweep also judges constraint satisfaction (establish/refute/absent) —
    results are stored in ``EdgePassResult.constraint_results`` so the
    caller can emit constraint findings without a second sweep.
    """
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")

    result = EdgePassResult()
    diag = result.diagnostics

    steps = sorted(
        (s for s in index.steps.values() if not run_id or s.run_id == run_id),
        key=lambda s: s.index,
    )
    steps_by_id = {s.step_id: s for s in steps}
    claims = sorted(
        (c for c in index.claims.values() if not run_id or c.run_id == run_id),
        key=lambda c: (
            steps_by_id[c.step_id].index if c.step_id in steps_by_id else len(steps),
            c.step_id,
        ),
    )
    result.n_claims = len(claims)
    observations = [s for s in steps if s.observation_segment]
    result.coverage.n_observation_steps = len(observations)
    if not claims or not observations:
        logger.info("claim edges: nothing to link (claims={}, observations={})",
                    len(claims), len(observations))
        return result

    if commit_step_id:
        observations = [s for s in observations if s.step_id != commit_step_id]
    obs_ids = {s.step_id for s in observations}
    claim_rows = [
        {"id": i, "made_at_step": c.step_id, "text": c.text}
        for i, c in enumerate(claims)
    ]
    constraint_rows: list[dict[str, Any]] | None = None
    if constraints and commit_binding:
        constraint_rows = [
            {"id": i, "desc": c.description}
            for i, c in enumerate(constraints)
        ]

    excerpts = [
        e for s in observations for e in _excerpts_for(s, _PARTITION_CHAR_BUDGET)
    ]
    partitions = partition_by_budget(excerpts, _PARTITION_CHAR_BUDGET, lambda e: len(e.text))
    result.coverage.n_partitions = len(partitions)

    # --- single-pass per partition: find + judge in one call ---
    proposals: list[tuple[Claim, Step, str, str]] = []
    attested: list[set[int]] = []
    judged: dict[int, bool] = {}
    nominated: dict[int, set[str]] = {}

    for pi, partition in enumerate(partitions):
        if any(e.oversized for e in partition):
            result.coverage.n_degraded_partitions += 1
            diag.record("edges", f"partition:{pi}", None, 0.0,
                        "coverage-degraded: an observation region exceeds the "
                        "call budget; negatives are not entitled by this sweep")
        payload_dict: dict[str, Any] = {
            "claims": claim_rows,
            "excerpts": [{"step": e.step.step_id, "text": e.text} for e in partition],
        }
        if constraint_rows:
            payload_dict["candidate"] = commit_binding
            payload_dict["constraints"] = constraint_rows
        payload = json.dumps(payload_dict, ensure_ascii=False, indent=2)

        shown_ids = {e.step.step_id for e in partition}
        raw = await _ask_model(
            "evidence", payload, model,
            session_factory=session_factory, purpose="evidence",
            key="",
        )
        rows_seen: set[int] = set()
        if raw is None:
            result.coverage.n_failed_partitions += 1
            diag.record("edges", f"partition:{pi}", None, 0.0,
                        f"evidence call failed for partition of {len(partition)} excerpts")
            attested.append(rows_seen)
            continue

        claim_list: list[Any]
        constraint_list: list[Any]
        if isinstance(raw, dict):
            cl = raw.get("claims", raw.get("verdicts", []))
            claim_list = cl if isinstance(cl, list) else []
            co = raw.get("constraints", [])
            constraint_list = co if isinstance(co, list) else []
        elif isinstance(raw, list):
            claim_list = raw
            constraint_list = []
        else:
            claim_list, constraint_list = [], []

        by_id = _index_by_id(claim_list)
        for ci_raw in range(len(claims)):
            item = by_id.get(ci_raw)
            if not item:
                continue
            rows_seen.add(ci_raw)
            matches = item.get("matches", [])
            if not isinstance(matches, list):
                continue
            claim = claims[ci_raw]
            for m in matches:
                if not isinstance(m, dict):
                    continue
                sid = str(m.get("step", ""))
                if sid not in shown_ids:
                    diag.prune("edges", claim.id, f"verdict for unshown step {sid!r}")
                    continue
                if sid not in obs_ids:
                    diag.prune("edges", claim.id, f"step {sid!r} is not an observation step")
                    continue
                relation = str(m.get("relation", ""))
                if relation not in _VERDICTS:
                    diag.prune("edges", claim.id, f"unknown relation {relation!r}")
                    continue
                nominated.setdefault(ci_raw, set()).add(sid)
                judged[ci_raw] = True
                result.verdicts.append(
                    {"claim": claim.id, "step": sid, "relation": relation}
                )
                if relation in _EDGE_KINDS:
                    proposals.append(
                        (claim, steps_by_id[sid], relation, str(m.get("quote", "")))
                    )
        if constraints and constraint_list:
            c_by_id = _index_by_id(constraint_list)
            for i, c in enumerate(constraints):
                if c.id in result.constraint_results:
                    continue
                item = c_by_id.get(i)
                if not item:
                    continue
                outcome = str(item.get("outcome", "absent"))
                if outcome not in ("establish", "refute"):
                    continue
                ev = tuple(
                    str(s) for s in item.get("steps", [])
                    if isinstance(s, str | int) and str(s) in shown_ids
                )
                status = "verified" if outcome == "establish" else "violated"
                result.constraint_results[c.id] = {
                    "status": status,
                    "confidence": _safe_float(item, "confidence"),
                    "source": "oracle:evidence",
                    "evidence_step_ids": ev,
                    "reason": str(item.get("reason", "")),
                }
                diag.record("evidence", c.id, outcome, _safe_float(item, "confidence"),
                            f"partition:{pi} quote={str(item.get('quote', ''))[:120]}")
        attested.append(rows_seen)

    for ci, sids in nominated.items():
        result.nominations[claims[ci].id] = sorted(
            sids, key=lambda sid: steps_by_id[sid].index,
        )

    # --- gates (code): verbatim single-region non-self quote + dedup ---
    verified: dict[str, dict[tuple[str, str], Edge]] = {}
    for claim, dst, relation, quote in proposals:
        if not _quote_in_regions(dst, quote):
            diag.prune("edges", claim.id,
                       f"quote not verbatim within one region of step {dst.step_id}: {quote[:60]!r}")
            continue
        if _self_quote(claim.text, quote):
            diag.prune("edges", claim.id, f"self-quote rejected: {quote[:60]!r}")
            continue
        per_claim = verified.setdefault(claim.id, {})
        key = (dst.step_id, relation)
        if key in per_claim:
            continue
        made_at = steps_by_id.get(claim.step_id)
        if made_at is None:
            position = ""
        elif dst.index < made_at.index:
            position = "before"
        elif dst.index == made_at.index:
            position = "same"
        else:
            position = "after"
        per_claim[key] = Edge(
            id=stable_id("edg", claim.run_id, relation, claim.id, dst.step_id),
            kind=relation,
            run_id=claim.run_id,
            src=claim.id,
            dst=dst.step_id,
            quote=quote,
            evidence_position=position,
        )
        diag.record(relation, claim.id, dst.step_id, 0.0,
                    f"{position or 'unpositioned'}: quote={quote[:60]!r}")

    # cap: supports only, in trajectory order — a prune must never be able
    # to strengthen a status, so conflicts edges are always kept (§2.6)
    kept: list[Edge] = []
    for claim_id, per_claim in verified.items():
        edges = sorted(
            per_claim.values(),
            key=lambda e: (steps_by_id[e.dst].index if e.dst in steps_by_id else 0, e.kind),
        )
        supports_kept = 0
        for edge in edges:
            if edge.kind == "conflicts":
                kept.append(edge)
            elif supports_kept < _MAX_SUPPORT_EDGES_PER_CLAIM:
                kept.append(edge)
                supports_kept += 1
            else:
                diag.prune("edges", claim_id,
                           f"supports cap {_MAX_SUPPORT_EDGES_PER_CLAIM}/claim: dropped {edge.dst}")

    stale = [
        eid for eid, e in index.edges.items()
        if e.kind in _EDGE_KINDS and (not run_id or e.run_id == run_id)
    ]
    for eid in stale:
        del index.edges[eid]
    for edge in kept:
        index.edges[edge.id] = edge

    # --- per-claim closed-world record (SCHEMA §2.4) ---
    for ci, claim in enumerate(claims):
        result.claim_coverage[claim.id] = ClaimCoverage(
            attended=all(ci in rows for rows in attested),
            judged=judged.get(ci, True),   # nothing nominated → vacuously judged
        )

    result.edges = kept

    # --- unsettled constraints: omitted or unknown ---
    if constraints:
        all_ok = result.coverage.n_failed_partitions == 0 and result.coverage.n_partitions > 0
        for c in constraints:
            if c.id in result.constraint_results:
                continue
            if all_ok:
                result.constraint_results[c.id] = {
                    "status": "omitted",
                    "confidence": 0.8,
                    "source": "oracle:evidence",
                    "evidence_step_ids": (),
                    "reason": "absent across all evidence partitions",
                }
                diag.record("evidence", c.id, "omitted", 0.0,
                            "absent in all partitions -> omitted")
            else:
                result.constraint_results[c.id] = {
                    "status": "unknown",
                    "confidence": 0.0,
                    "source": "code",
                    "evidence_step_ids": (),
                    "reason": f"{result.coverage.n_failed_partitions} partition(s) failed",
                }

    by_kind: dict[str, int] = {}
    for e in kept:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    n_attended = sum(1 for cc in result.claim_coverage.values() if cc.attended)
    logger.info(
        "claim edges: {} claims -> {} nominated, {} edges {} "
        "(content {}/{} partitions ok, {} degraded; attended {}/{})",
        result.n_claims, len(nominated), len(kept), by_kind,
        result.coverage.n_partitions - result.coverage.n_failed_partitions,
        result.coverage.n_partitions, result.coverage.n_degraded_partitions,
        n_attended, result.n_claims,
    )
    return result
