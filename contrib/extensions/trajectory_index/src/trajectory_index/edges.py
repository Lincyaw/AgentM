"""Pass 2 — edge construction over Pass 1 nodes.

Pass 1 emits nodes (symbols, claims, provenance segments); this pass
connects claims to the observation evidence they relate to. Generic over
trajectory types: it presumes only the two node kinds, never a benchmark
or domain.

Edge kinds — both carry a code-verified verbatim quote from the
destination's observation segment:

* ``supports``  — the observation contains content that directly supports
  the claim;
* ``conflicts`` — the observation contains content that contradicts it.

Polarity lives on the edge because it is a LOCAL pairwise fact (one
claim against one excerpt, witnessed by a quote); the GLOBAL per-claim
status (supported / conflicted / unsourced / unknown) is a pure-code
fold over edges plus the coverage record (verification.py) — the model
never issues a global verdict.

    assemble  (code)    candidates = observation segments of steps up to
                        and including the claim's step — whole segments
    partition (code)    deterministic char-budget partitions, whole steps
                        only; together the partitions COVER the candidate
                        evidence space, which is what entitles the downstream
                        fold to an attested "unsourced" negative (P4)
    propose   (oracle)  per partition, every claim judged against that
                        partition's excerpts; positive polarity; sampled
                        ``_N_SAMPLES`` times and unioned — sampling can
                        only surface more candidate edges, never assert
                        one (verification gates them all)
    verify    (code)    endpoints exist, quote is verbatim (FULL text)
                        in the destination's observation segment and is
                        not the claim's own words; failures are rejected
                        and logged, never stored (P2). Timeline is a
                        FACT, not a filter: evidence_position records
                        before/same/after the claim — an after-conflicts
                        edge ("committed early, refuted later") is signal,
                        not noise

Contracts: oracle judgments land in the transcript (P3); a failed call
breaks coverage for that partition (recorded) and can only weaken
downstream negatives, never fabricate an edge (P5); no mid-content
truncation anywhere (P4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .adjudicate import SessionFactory, _ask_model
from .diagnostics import Diagnostics
from .index import Claim, Edge, Step, TrajectoryIndex, stable_id

_PARTITION_CHAR_BUDGET = 60_000   # max excerpt chars per oracle call (whole steps)
_MAX_EDGES_PER_CLAIM = 6          # highest-confidence edges kept per claim (pruned, logged)
_N_SAMPLES = 2                    # oracle samples per partition, unioned (variance control)

_EDGE_KINDS = ("supports", "conflicts")


@dataclass(slots=True)
class EdgeCoverage:
    """Did the oracle sweep see the whole candidate evidence space?

    ``complete`` requires every partition to have at least one successful
    sample — the entitlement for the fold's "unsourced" negative.
    """

    n_observation_steps: int = 0
    n_partitions: int = 0
    n_failed_partitions: int = 0

    @property
    def complete(self) -> bool:
        return self.n_failed_partitions == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observation_steps": self.n_observation_steps,
            "n_partitions": self.n_partitions,
            "n_failed_partitions": self.n_failed_partitions,
            "complete": self.complete,
        }


@dataclass(slots=True)
class EdgePassResult:
    edges: list[Edge] = field(default_factory=list)
    n_claims: int = 0
    coverage: EdgeCoverage = field(default_factory=EdgeCoverage)
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
            "transcript": self.diagnostics.transcript,
            "prune_log": self.diagnostics.prune_log,
        }


# ---------------------------------------------------------------------------
# assemble + partition (code)
# ---------------------------------------------------------------------------


def _quote_in_segment(segment: str, quote: str) -> bool:
    """FULL-quote verbatim containment, whitespace-normalized only.

    The quote is the load-bearing witness for the edge — the entire
    passage must appear in the observation segment, not a prefix of it
    (a prefix match would let the model fabricate everything past the
    matched head).
    """
    norm_quote = " ".join(quote.split())
    return bool(norm_quote) and norm_quote in " ".join(segment.split())


def _self_quote(claim_text: str, quote: str) -> bool:
    """Is the quote just the claim's own words? (whitespace/case-insensitive
    containment either way — the decidable core of "a claim is not its own
    evidence")."""
    norm_claim = " ".join(claim_text.lower().split())
    norm_quote = " ".join(quote.lower().split())
    if not norm_quote:
        return True
    return norm_quote in norm_claim or norm_claim in norm_quote


def partition_by_budget(steps: list[Step], budget: int) -> list[list[Step]]:
    """Greedy in-order partition into whole-step groups under ``budget`` chars.

    A single step larger than the budget forms its own partition — the
    step is sent whole, never cut.
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


# ---------------------------------------------------------------------------
# propose (oracle)
# ---------------------------------------------------------------------------

_CLAIM_EDGE_INSTRUCTIONS = """\
An agent made claims during a task. Below are the claims and a set of
observation excerpts (tool output, fetched pages, command results the agent
had received). For each claim, report the excerpts that bear on it:

  - "supports": the excerpt contains content that directly supports the
    claim — same entities AND the same specific fact.
  - "conflicts": the excerpt contains content that contradicts the claim.
  - Report nothing for excerpts that are merely on the same topic.
  - Copy the decisive passage VERBATIM into "quote" — it is checked
    mechanically against the excerpt, and a paraphrase is discarded. The
    quote must be observation content, NEVER the claim's own sentence or
    the agent's words restating it.
  - A claim may relate to several excerpts, or to none in this set —
    "none" is normal and needs no entry.

Return ONLY:
{"verdicts": [{"claim": 0, "step": "12", "relation": "supports|conflicts", "quote": "..."}]}
"""


async def build_claim_edges(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
    samples: int = _N_SAMPLES,
) -> EdgePassResult:
    """Build ``supports``/``conflicts`` edges (claim ↔ observation evidence).

    Idempotent per run: existing edges of these kinds for this run are
    replaced wholesale. Best-effort: a failed oracle call contributes no
    edges and marks its partition's coverage broken (recorded).
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
        key=lambda c: c.step_id,
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

    partitions = partition_by_budget(observations, _PARTITION_CHAR_BUDGET)
    result.coverage.n_partitions = len(partitions)

    proposals: list[tuple[Claim, Step, str, str]] = []
    for pi, partition in enumerate(partitions):
        payload = json.dumps({
            "claims": claim_rows,
            "excerpts": [
                {"step": s.step_id, "text": s.observation_segment}
                for s in partition
            ],
        }, ensure_ascii=False, indent=2)
        ok = 0
        for _ in range(max(1, samples)):
            raw = await _ask_model(
                _CLAIM_EDGE_INSTRUCTIONS, payload, model,
                session_factory=session_factory, purpose="claim_edges",
            )
            if raw is None:
                continue
            ok += 1
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    claim = claims[int(item.get("claim", -1))]
                except (ValueError, TypeError, IndexError):
                    diag.prune("edges", str(item.get("claim")), "unknown claim id in proposal")
                    continue
                dst = steps_by_id.get(str(item.get("step", "")))
                if dst is None:
                    diag.prune("edges", claim.id, f"unknown step {item.get('step')!r} in proposal")
                    continue
                relation = str(item.get("relation", ""))
                if relation not in _EDGE_KINDS:
                    diag.prune("edges", claim.id, f"unknown relation {relation!r}")
                    continue
                proposals.append((claim, dst, relation, str(item.get("quote", ""))))
        if ok == 0:
            result.coverage.n_failed_partitions += 1
            diag.record("edges", f"partition:{pi}", None, 0.0,
                        f"all {samples} samples failed for partition of {len(partition)} steps; "
                        "coverage broken")

    # verify (code): verbatim non-self quote + (src, dst, kind) dedup
    # (first verified proposal wins). Timeline is recorded as a fact, never
    # used as a filter — consistency is time-agnostic.
    verified: dict[str, dict[tuple[str, str], Edge]] = {}
    for claim, dst, relation, quote in proposals:
        segment = dst.observation_segment or ""
        if not _quote_in_segment(segment, quote):
            diag.prune("edges", claim.id,
                       f"quote not verbatim in step {dst.step_id}: {quote[:60]!r}")
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

    kept: list[Edge] = []
    for claim_id, per_claim in verified.items():
        edges = sorted(per_claim.values(), key=lambda e: e.dst)
        for extra in edges[_MAX_EDGES_PER_CLAIM:]:
            diag.prune("edges", claim_id,
                       f"edge cap {_MAX_EDGES_PER_CLAIM}/claim: dropped {extra.kind}:{extra.dst}")
        kept.extend(edges[:_MAX_EDGES_PER_CLAIM])

    stale = [
        eid for eid, e in index.edges.items()
        if e.kind in _EDGE_KINDS and (not run_id or e.run_id == run_id)
    ]
    for eid in stale:
        del index.edges[eid]
    for edge in kept:
        index.edges[edge.id] = edge

    result.edges = kept
    by_kind: dict[str, int] = {}
    for e in kept:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    logger.info("claim edges: {} claims → {} edges {} (coverage {}/{} partitions ok)",
                result.n_claims, len(kept), by_kind,
                result.coverage.n_partitions - result.coverage.n_failed_partitions,
                result.coverage.n_partitions)
    return result
