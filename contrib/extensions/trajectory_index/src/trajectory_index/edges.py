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
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .adjudicate import SessionFactory, _ask_model
from .diagnostics import Diagnostics
from .index import Claim, Edge, Step, TrajectoryIndex, stable_id

_PARTITION_CHAR_BUDGET = 60_000   # max excerpt chars per oracle call (whole excerpts)
_MAX_SUPPORT_EDGES_PER_CLAIM = 6  # supports capped in trajectory order; conflicts never
_N_RETRIEVAL_SAMPLES = 2          # retrieval samples per partition, unioned
_VERIFY_ATTEMPTS = 2              # attempts per verification call before giving up

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


def partition_by_budget(steps: list[Step], budget: int) -> list[list[Step]]:
    """Greedy in-order partition into whole-step groups under ``budget`` chars.

    A single step larger than the budget forms its own partition — the
    step is sent whole, never cut. (Shared with the constraint layer's
    sweep; the edge pass itself partitions region-level excerpts.)
    """
    partitions: list[list[Step]] = []
    current: list[Step] = []
    size = 0
    for s in steps:
        n = len(s.observation_segment or "")
        if current and size + n > budget:
            partitions.append(current)
            current, size = [], 0
        current.append(s)
        size += n
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


def _partition_excerpts(excerpts: list[_Excerpt], budget: int) -> list[list[_Excerpt]]:
    partitions: list[list[_Excerpt]] = []
    current: list[_Excerpt] = []
    size = 0
    for e in excerpts:
        if current and size + len(e.text) > budget:
            partitions.append(current)
            current, size = [], 0
        current.append(e)
        size += len(e.text)
    if current:
        partitions.append(current)
    return partitions


# ---------------------------------------------------------------------------
# retrieval (oracle): nominate candidate evidence, one row per claim
# ---------------------------------------------------------------------------

_RETRIEVAL_INSTRUCTIONS = """\
An agent made claims during a task. Below are the claims and a set of
observation excerpts (tool output, fetched pages, command results the agent
had received). This is a RELEVANCE listing, not a judgment: a later focused
check decides what each excerpt actually says about each claim.

For EVERY claim, list the steps whose excerpt could bear on it — material
about the same entities, quantities, dates, or facts. Be generous: a wrongly
listed step costs one extra check; a missed step is never examined again.

  - Output exactly one row per claim, covering every claim in order.
  - A claim with no related excerpts gets an empty list. The empty row is
    required — it records that the claim was checked against this set.
  - Do not judge support or contradiction here; do not copy quotes.

Return ONLY:
{"rows": [{"claim": 0, "steps": ["12", "47"]}, {"claim": 1, "steps": []}]}
"""


# ---------------------------------------------------------------------------
# verify (oracle): focused adversarial judgment per claim
# ---------------------------------------------------------------------------

_VERIFY_INSTRUCTIONS = """\
An agent made a claim during a task; a relevance scan flagged the observation
excerpts below (tool output, fetched pages the agent had received) as possibly
bearing on it. Judge each excerpt independently and adversarially: try to
REFUTE the pairing before accepting it.

  - "supports": the excerpt itself states the same specific fact about the
    same entities as the claim.
  - "conflicts": the excerpt states something incompatible with the claim.
  - "neutral": on the same topic, but it does not settle the claim either way
    (no specific value, status, or fact in common to compare).

When the claim asserts a specific value — a date, number, status, name, or
count — find that same attribute in the excerpt and COMPARE the values. A
different value is a "conflicts", even when a matching-looking value also
appears nearby: an excerpt may hold both a requested date and the actual date,
or an asked-for figure and the returned one. Anchor on what the source
actually reports, not on the value that echoes the claim. This value mismatch
is the most commonly missed contradiction; do not overlook it.

  - For supports/conflicts, copy the decisive passage VERBATIM into "quote".
    It is checked mechanically against the excerpt; a paraphrase is discarded.
    The quote must be observation content, NEVER the claim's own sentence or a
    restatement of it. For a value mismatch, quote the passage carrying the
    source's actual value.
  - Every excerpt gets exactly one verdict row.

Return ONLY:
{"verdicts": [{"step": "12", "relation": "supports|conflicts|neutral", "quote": "..."}]}
"""


async def build_claim_edges(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
    samples: int = _N_RETRIEVAL_SAMPLES,
) -> EdgePassResult:
    """Build ``supports``/``conflicts`` edges (claim ↔ observation evidence).

    Two oracle stages (SCHEMA §2.2): partition-swept retrieval nominates
    candidates for every claim; a focused per-claim verification judges
    polarity and produces the witness quote. Idempotent per run: existing
    edges of these kinds for this run are replaced wholesale. Best-effort:
    every failure weakens coverage for exactly what it touched, never more.
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

    claim_steps = {c.id: steps_by_id.get(c.step_id) for c in claims}
    claim_rows = [
        {"claim": i, "made_at_step": c.step_id, "text": c.text}
        for i, c in enumerate(claims)
    ]

    excerpts = [
        e for s in observations for e in _excerpts_for(s, _PARTITION_CHAR_BUDGET)
    ]
    partitions = _partition_excerpts(excerpts, _PARTITION_CHAR_BUDGET)
    result.coverage.n_partitions = len(partitions)

    # --- retrieval: nominations + per-partition attention attestation ---
    nominated: dict[int, set[str]] = {}
    attested: list[set[int]] = []
    for pi, partition in enumerate(partitions):
        if any(e.oversized for e in partition):
            result.coverage.n_degraded_partitions += 1
            diag.record("edges", f"partition:{pi}", None, 0.0,
                        "coverage-degraded: an observation region exceeds the "
                        "call budget; negatives are not entitled by this sweep")
        payload = json.dumps({
            "claims": claim_rows,
            "excerpts": [{"step": e.step.step_id, "text": e.text} for e in partition],
        }, ensure_ascii=False, indent=2)
        rows_seen: set[int] = set()
        ok = 0
        for _ in range(max(1, samples)):
            raw = await _ask_model(
                _RETRIEVAL_INSTRUCTIONS, payload, model,
                session_factory=session_factory, purpose="edge_retrieval",
                key="rows",
            )
            if raw is None:
                continue
            ok += 1
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    ci = int(item.get("claim", -1))
                except (TypeError, ValueError):
                    diag.prune("edges", str(item.get("claim")), "unparseable claim id in retrieval row")
                    continue
                if not 0 <= ci < len(claims):
                    diag.prune("edges", str(item.get("claim")), "unknown claim id in retrieval row")
                    continue
                rows_seen.add(ci)
                sids = item.get("steps")
                for sid in sids if isinstance(sids, list) else []:
                    sid = str(sid)
                    if sid in steps_by_id and steps_by_id[sid].observation_segment:
                        nominated.setdefault(ci, set()).add(sid)
                    else:
                        diag.prune("edges", claims[ci].id,
                                   f"nominated step {sid!r} is not an observation step")
        if ok == 0:
            result.coverage.n_failed_partitions += 1
            diag.record("edges", f"partition:{pi}", None, 0.0,
                        f"all {samples} retrieval samples failed for partition "
                        f"of {len(partition)} excerpts; coverage broken")
        attested.append(rows_seen)

    # --- verify: focused per-claim adversarial judgment over nominations ---
    proposals: list[tuple[Claim, Step, str, str]] = []
    judged: dict[int, bool] = {}
    for ci, sids in sorted(nominated.items()):
        claim = claims[ci]
        nominees = sorted(sids, key=lambda sid: steps_by_id[sid].index)
        chunks = _partition_excerpts(
            [_Excerpt(steps_by_id[sid], steps_by_id[sid].observation_segment or "", False)
             for sid in nominees],
            _PARTITION_CHAR_BUDGET,
        )
        decided: set[str] = set()
        for chunk in chunks:
            shown = {e.step.step_id for e in chunk}
            payload = json.dumps({
                "claim": {"made_at_step": claim.step_id, "text": claim.text},
                "excerpts": [{"step": e.step.step_id, "text": e.text} for e in chunk],
            }, ensure_ascii=False, indent=2)
            raw = None
            for _ in range(_VERIFY_ATTEMPTS):
                raw = await _ask_model(
                    _VERIFY_INSTRUCTIONS, payload, model,
                    session_factory=session_factory, purpose="edge_verification",
                )
                if raw is not None:
                    break
            if raw is None:
                diag.record("edges", claim.id, None, 0.0,
                            f"verification failed for {len(chunk)} excerpts; "
                            "claim demoted to unknown, never unsourced")
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("step", ""))
                if sid not in shown:
                    diag.prune("edges", claim.id, f"verdict for unshown step {sid!r}")
                    continue
                relation = str(item.get("relation", ""))
                if relation not in _VERDICTS:
                    diag.prune("edges", claim.id, f"unknown relation {relation!r}")
                    continue
                decided.add(sid)
                result.verdicts.append(
                    {"claim": claim.id, "step": sid, "relation": relation}
                )
                if relation in _EDGE_KINDS:
                    proposals.append(
                        (claim, steps_by_id[sid], relation, str(item.get("quote", "")))
                    )
        judged[ci] = decided >= sids
        result.nominations[claim.id] = nominees

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
        made_at = claim_steps.get(claim.id)
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
    by_kind: dict[str, int] = {}
    for e in kept:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    n_attended = sum(1 for cc in result.claim_coverage.values() if cc.attended)
    logger.info(
        "claim edges: {} claims → {} nominated, {} edges {} "
        "(content {}/{} partitions ok, {} degraded; attended {}/{})",
        result.n_claims, len(nominated), len(kept), by_kind,
        result.coverage.n_partitions - result.coverage.n_failed_partitions,
        result.coverage.n_partitions, result.coverage.n_degraded_partitions,
        n_attended, result.n_claims,
    )
    return result
