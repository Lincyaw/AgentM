"""Pass 2 — edge construction over Pass 1 nodes.

Pass 1 emits nodes (symbols, claims, provenance); this pass connects
them. Organized like the other analysis passes: code owns assembly and
verification, one oracle relation per edge kind, one diagnostics sink.

First edge kind — ``grounds``: claim → observation step. "This claim is
talking about content in that step" (its source), regardless of whether
the content agrees with the claim; agreement is a Pass 3 judgment. The
edge replaces lexical-overlap window guessing in downstream consumers.

    assemble  (code)    candidates per claim = observation steps strictly
                        before the claim's step — whole segments, complete
    partition (code)    deterministic char-budget partitions, whole steps
                        only (an oversized step becomes its own partition;
                        nothing is ever cut mid-step)
    propose   (oracle)  per partition, every claim judged against that
                        partition's excerpts — positive polarity: an edge
                        is asserted only where the source is visible; "no
                        edge in this partition" says nothing global
    verify    (code)    endpoints exist, direction holds (source step is
                        earlier than the claim), quote verbatim-present in
                        the destination's observation segment; failures
                        are rejected and logged, never stored (P2)

Contracts: oracle judgments land in the transcript (P3); missing or
unparseable output degrades to "no edges from this call", never to a
fabricated edge (P5); no mid-content truncation anywhere (P4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .adjudicate import SessionFactory, _ask_model, _safe_float
from .diagnostics import Diagnostics
from .index import Claim, Edge, Step, TrajectoryIndex, _find_boundary, stable_id

_PARTITION_CHAR_BUDGET = 60_000   # max excerpt chars per oracle call (whole steps)
_MAX_EDGES_PER_CLAIM = 4          # highest-confidence edges kept per claim (pruned, logged)


@dataclass(slots=True)
class EdgePassResult:
    edges: list[Edge] = field(default_factory=list)
    n_claims: int = 0
    n_linked: int = 0                 # claims with at least one verified edge
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "edges": [
                {
                    "id": e.id, "kind": e.kind, "src": e.src, "dst": e.dst,
                    "quote": e.quote, "confidence": e.confidence,
                }
                for e in self.edges
            ],
            "n_claims": self.n_claims,
            "n_linked": self.n_linked,
            "transcript": self.diagnostics.transcript,
            "prune_log": self.diagnostics.prune_log,
        }


# ---------------------------------------------------------------------------
# assemble + partition (code)
# ---------------------------------------------------------------------------


def _self_quote(claim_text: str, quote: str) -> bool:
    """Is the quote just the claim's own words? (whitespace/case-insensitive
    containment either way — the decidable core of "a claim is not its own
    source")."""
    norm_claim = " ".join(claim_text.lower().split())
    norm_quote = " ".join(quote.lower().split())
    if not norm_quote:
        return True
    return norm_quote in norm_claim or norm_claim in norm_quote


def _partition_by_budget(steps: list[Step], budget: int) -> list[list[Step]]:
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

_GROUNDS_INSTRUCTIONS = """\
An agent made claims during a task. Below are the claims and a set of
observation excerpts (tool/search/page content the agent had received
earlier). For each claim, identify the excerpts that contain the passage
the claim is talking about — its source — whether or not that passage
actually agrees with the claim.

  - Link only when the excerpt visibly contains the specific content the
    claim refers to: same entities AND the same specific fact or topic.
    Mere topical similarity is not a link.
  - Copy the passage VERBATIM into "quote" — it is checked mechanically
    against the excerpt, and a paraphrase is discarded.
  - The quote must be retrieved material (page text, tool output), NEVER
    the claim's own sentence or the agent's words restating it — a claim
    is not its own source.
  - A claim may link to several excerpts, or to none in this set. "None"
    is normal: the claim's source may simply not be shown here.

Return ONLY:
{"verdicts": [{"claim": 0, "step": "12", "quote": "...", "confidence": 0.9}]}
"""


async def build_grounds_edges(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
) -> EdgePassResult:
    """Build ``grounds`` edges (claim → source observation step).

    Idempotent per run: existing ``grounds`` edges for this run are
    replaced wholesale. Best-effort: a failed oracle call contributes no
    edges from that partition (recorded), never a fabricated edge.
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
    if not claims or not observations:
        logger.info("grounds edges: nothing to link (claims={}, observations={})",
                    len(claims), len(observations))
        return result

    claim_steps = {c.id: steps_by_id.get(c.step_id) for c in claims}
    claim_rows = [
        {"claim": i, "made_at_step": c.step_id, "text": c.text}
        for i, c in enumerate(claims)
    ]

    proposals: list[tuple[Claim, Step, str, float]] = []
    for partition in _partition_by_budget(observations, _PARTITION_CHAR_BUDGET):
        payload = json.dumps({
            "claims": claim_rows,
            "excerpts": [
                {"step": s.step_id, "text": s.observation_segment}
                for s in partition
            ],
        }, ensure_ascii=False, indent=2)
        raw = await _ask_model(
            _GROUNDS_INSTRUCTIONS, payload, model,
            session_factory=session_factory, purpose="grounds_edges",
        )
        if raw is None:
            diag.record("grounds", "-", None, 0.0,
                        f"oracle call failed for partition of {len(partition)} steps; "
                        "no edges from it")
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                claim = claims[int(item.get("claim", -1))]
            except (ValueError, TypeError, IndexError):
                diag.prune("grounds", str(item.get("claim")), "unknown claim id in proposal")
                continue
            dst = steps_by_id.get(str(item.get("step", "")))
            if dst is None:
                diag.prune("grounds", claim.id, f"unknown step {item.get('step')!r} in proposal")
                continue
            proposals.append((claim, dst, str(item.get("quote", "")), _safe_float(item, "confidence")))

    # verify (code): direction + verbatim quote + (src, dst) dedup, then per-claim cap.
    # Direction admits the claim's OWN step: in a mixed span the claim sits in
    # the agent segment and its source in the same span's observation segment —
    # the most common grounding shape. Only strictly-later steps are rejected.
    verified: dict[str, dict[str, Edge]] = {}
    for claim, dst, quote, conf in proposals:
        made_at = claim_steps.get(claim.id)
        if made_at is not None and dst.index > made_at.index:
            diag.prune("grounds", claim.id,
                       f"direction: step {dst.step_id} after claim step {made_at.step_id}")
            continue
        segment = dst.observation_segment or ""
        if not quote.strip() or _find_boundary(segment, quote) < 0:
            diag.prune("grounds", claim.id,
                       f"quote not verbatim in step {dst.step_id}: {quote[:60]!r}")
            continue
        if _self_quote(claim.text, quote):
            # A claim is not its own source. Trailing agent prose inside a
            # mixed span's observation segment makes the claim text itself
            # quotable — the degenerate edge this rejects.
            diag.prune("grounds", claim.id, f"self-quote rejected: {quote[:60]!r}")
            continue
        per_claim = verified.setdefault(claim.id, {})
        prior = per_claim.get(dst.step_id)
        if prior is not None and prior.confidence >= conf:
            continue
        per_claim[dst.step_id] = Edge(
            id=stable_id("edg", claim.run_id, "grounds", claim.id, dst.step_id),
            kind="grounds",
            run_id=claim.run_id,
            src=claim.id,
            dst=dst.step_id,
            quote=quote,
            confidence=round(conf, 3),
        )
        diag.record("grounds", claim.id, dst.step_id, conf, f"quote={quote[:60]!r}")

    kept: list[Edge] = []
    for claim_id, per_claim in verified.items():
        edges = sorted(per_claim.values(), key=lambda e: (-e.confidence, e.dst))
        for extra in edges[_MAX_EDGES_PER_CLAIM:]:
            diag.prune("grounds", claim_id,
                       f"edge cap {_MAX_EDGES_PER_CLAIM}/claim: dropped {extra.dst}")
        kept.extend(edges[:_MAX_EDGES_PER_CLAIM])

    stale = [
        eid for eid, e in index.edges.items()
        if e.kind == "grounds" and (not run_id or e.run_id == run_id)
    ]
    for eid in stale:
        del index.edges[eid]
    for edge in kept:
        index.edges[edge.id] = edge

    result.edges = kept
    result.n_linked = len(verified)
    logger.info("grounds edges: {} claims → {} edges ({} claims linked)",
                result.n_claims, len(kept), result.n_linked)
    return result
