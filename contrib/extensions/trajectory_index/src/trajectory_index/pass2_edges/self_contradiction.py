"""Pass 2 — agent-internal edges: claim↔claim self-contradiction (SCHEMA §2.8).

Evidence edges (``claims.py``) compare the agent to the ENVIRONMENT; this
module compares the agent to ITSELF. Two claims that assert incompatible
things about the SAME symbol are a self-contradiction an auditor can localize
WITHOUT any environment evidence — belief revision, flip-flops, an answer
quietly swapped mid-trajectory.

This is the PROPOSITIONAL tier of §2.8: irreducible model NLI, surfaced as a
MONOTONE advisory witnessed by both claim texts, never a hard verdict. (The
value-contradiction tier is code-decidable only once a value world exists,
which it does not today — SCHEMA §2.8; the retraction tier needs a Pass 1
``retract`` stance.) Division of labor as everywhere: code groups the
candidate claim pairs by shared symbol and assembles the edges; the model
does only the local pairwise contradiction call.

Best-effort and idempotent: the ``self_contradicts`` edges for the run are
cleared and rebuilt wholesale; a model failure leaves no edge (never a
fabricated one).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..ir.models import Edge, stable_id
from ..oracle import SessionFactory, _ask_model, _index_by_id

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex

EDGE_KIND = "self_contradicts"

# Bound the pairwise call: a symbol re-asserted many times would blow the
# pair count quadratically. Cap the pairs actually judged; the cap is logged
# (P2 — no silent truncation) by the caller via the result's ``capped`` count.
_MAX_PAIRS = 60

@dataclass(slots=True)
class SelfContradictionResult:
    edges: list[Edge] = field(default_factory=list)
    n_pairs: int = 0          # candidate pairs formed
    n_judged: int = 0         # pairs actually sent to the model (after the cap)
    n_capped: int = 0         # pairs dropped by the cap (logged, not silent)
    verdicts: list[dict[str, str]] = field(default_factory=list)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "edges": [
                {"id": e.id, "kind": e.kind, "src": e.src, "dst": e.dst,
                 "evidence_position": e.evidence_position}
                for e in self.edges
            ],
            "n_pairs": self.n_pairs,
            "n_judged": self.n_judged,
            "n_capped": self.n_capped,
            "verdicts": self.verdicts,
        }


def _claims_by_symbol(index: TrajectoryIndex, run_id: str) -> dict[str, list[Any]]:
    """Group a run's claims by shared symbol_ids (populated in Pass 1)."""
    claims = sorted(
        (c for c in index.claims.values() if not run_id or c.run_id == run_id),
        key=lambda c: (c.step_id, c.id),
    )
    out: dict[str, list[Any]] = {}
    for claim in claims:
        for sid in claim.symbol_ids:
            out.setdefault(sid, []).append(claim)
    return {sid: cs for sid, cs in out.items() if len(cs) >= 2}


def _step_index(index: TrajectoryIndex, run_id: str, step_id: str) -> int:
    st = index.steps.get((run_id, step_id))
    return st.index if st is not None else 0


async def build_self_contradiction_edges(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
) -> SelfContradictionResult:
    """Build claim↔claim ``self_contradicts`` edges over a run (SCHEMA §2.8).

    Idempotent: clears the run's ``self_contradicts`` edges before rebuilding.
    """
    # Idempotent clear (this pass owns only its kind; claims.py sweeps only
    # supports/conflicts, so nothing else touches these).
    index.edges = {
        eid: e for eid, e in index.edges.items()
        if not (e.kind == EDGE_KIND and (not run_id or e.run_id == run_id))
    }

    result = SelfContradictionResult()
    groups = _claims_by_symbol(index, run_id)
    if not groups:
        return result
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create offline)")

    # Form candidate pairs (dedup a pair that shares more than one symbol).
    seen_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    pair_meta: list[tuple[Any, Any]] = []   # (claim_a, claim_b) aligned to rows
    for sym_id, claims in groups.items():
        sym = index.symbols.get(sym_id)
        entity = sym.canonical_name if sym else sym_id
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                a, b = claims[i], claims[j]
                key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                result.n_pairs += 1
                if len(rows) >= _MAX_PAIRS:
                    result.n_capped += 1
                    continue
                rows.append({
                    "id": len(rows), "entity": entity,
                    "claim_a": a.text[:400], "claim_b": b.text[:400],
                })
                pair_meta.append((a, b))
    if result.n_capped:
        logger.info("self-contradiction: {} pairs, judging {} (capped {})",
                    result.n_pairs, len(rows), result.n_capped)
    if not rows:
        return result
    result.n_judged = len(rows)

    raw = await _ask_model(
        "self_contradiction", json.dumps(rows, ensure_ascii=False, indent=2), model,
        session_factory=session_factory, purpose="self_contradiction",
    )
    if raw is None:
        return result
    by_id = _index_by_id(raw)

    for i, (a, b) in enumerate(pair_meta):
        item = by_id.get(i)
        outcome = str(item.get("outcome", "unclear")) if item else "unclear"
        result.verdicts.append({"pair": f"{a.id}|{b.id}", "outcome": outcome})
        if outcome != "contradict":
            continue
        # order the edge earlier→later by step so evidence_position reads right
        ai, bi = _step_index(index, a.run_id, a.step_id), _step_index(index, b.run_id, b.step_id)
        src, dst = (a, b) if ai <= bi else (b, a)
        pos = "before" if ai != bi else "same"
        edge = Edge(
            id=stable_id("edge", run_id, EDGE_KIND, src.id, dst.id),
            kind=EDGE_KIND, run_id=run_id, src=src.id, dst=dst.id,
            quote="", evidence_position=pos,
        )
        index.edges[edge.id] = edge
        result.edges.append(edge)

    logger.info("self-contradiction: {} edges from {} judged pairs",
                len(result.edges), result.n_judged)
    return result
