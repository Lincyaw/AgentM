"""Source-claim consistency pass — checks the agent's own verification claims.

Targets the error class where the agent asserts "verified / confirmed /
according to <source>" but the observation content it just received does
not contain (or contradicts) the claimed fact. This leaves no grounding
trace (the entities are real, the values may match elsewhere) — it is a
claim ABOUT evidence, so the check compares claim text against the
adjacent observation window.

Shape (same contracts as constraints.py, P1-P7):

* claims come from Pass 1's unified extraction (``index.claims`` — one
  visit of the trajectory yields symbols AND claims; downstream passes
  consume the same first-class facts, never re-extract); code selects
  the observation window — whole steps, full text, every deselection
  logged (P2/P4);
* one batched oracle call judges each claim against ITS window only:
  supported / conflicted / not_present — all three are window-scoped
  statements ("not present in these excerpts" says nothing about other
  sources; the Entails "neither" precedent), with a verbatim quote for
  supported/conflicted (P4);
* missing/unparseable verdicts stay unknown and are never surfaced as
  alarms (P5); every oracle judgment lands in the transcript (P3);
* output is FACTS for the auditor (this kind is born ungated): the claim,
  its window, and what the check found — never an error-span designation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .adjudicate import SessionFactory, _ask_model, _index_by_id, _safe_float
from .diagnostics import Diagnostics
from .diagnostics import content_tokens as _content_tokens
from .index import Step, TrajectoryIndex

_MAX_CLAIMS_PER_STEP = 3      # claims per step accepted from the extractor (logged)
_MAX_WINDOW_STEPS = 6         # observation steps per claim window (selection, logged)


@dataclass(frozen=True, slots=True)
class SourceClaim:
    """One checked verification claim (a fact record, not a verdict)."""

    step_id: str
    claim: str                          # the claim sentence(s), verbatim
    source_step_ids: tuple[str, ...]    # observation window it was checked against
    outcome: str                        # supported | conflicted | not_present | unknown
    quote: str = ""                     # verbatim evidence for supported/conflicted
    confidence: float = 0.0


@dataclass(slots=True)
class SourceClaimAnalysis:
    claims: list[SourceClaim] = field(default_factory=list)
    n_detected: int = 0
    n_unchecked: int = 0                # detected but no observation window
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "step_id": c.step_id, "claim": c.claim,
                    "source_step_ids": list(c.source_step_ids),
                    "outcome": c.outcome, "quote": c.quote,
                    "confidence": c.confidence,
                }
                for c in self.claims
            ],
            "n_detected": self.n_detected,
            "n_unchecked": self.n_unchecked,
            "transcript": self.diagnostics.transcript,
            "prune_log": self.diagnostics.prune_log,
        }


# ---------------------------------------------------------------------------
# Claim source + window selection (code)
# ---------------------------------------------------------------------------

# Claims come from Pass 1's unified extraction (index.claims — one visit of
# the trajectory extracts symbols AND claims; every downstream consumer reads
# the same first-class facts, no per-pass re-extraction). Claim detection is
# semantic and lives in the extractor prompt; verbatim presence was already
# code-verified at populate time. A keyword detector was tried and rejected:
# it misses paraphrase ("the birth year lines up") and false-fires on intent
# ("I need to confirm this next") and negation ("unconfirmed").


def _claims_from_index(
    index: TrajectoryIndex, steps: list[Step], diag: Diagnostics,
) -> list[tuple[Step, str]]:
    """(step, claim text) pairs from the index's first-class claims."""
    steps_by_id = {s.step_id: s for s in steps}
    per_step: dict[str, int] = {}
    out: list[tuple[Step, str]] = []
    for claim in sorted(index.claims.values(), key=lambda c: c.step_id):
        step = steps_by_id.get(claim.step_id)
        if step is None:
            continue
        n = per_step.get(step.step_id, 0)
        if n >= _MAX_CLAIMS_PER_STEP:
            diag.prune("claims", step.step_id,
                       f"claim cap {_MAX_CLAIMS_PER_STEP}/step: {claim.text[:60]!r}")
            continue
        per_step[step.step_id] = n + 1
        out.append((step, claim.text))
    return out


_MIN_LINK_TOKENS = 2          # lexical link: shared content tokens to qualify


def _observation_window(
    steps: list[Step], claim_step: Step, claim_text: str, diag: Diagnostics,
) -> list[Step]:
    """The observation steps a claim should be checked against.

    Adjacent observations (between the previous assistant step and the
    claim) PLUS lexically-linked observations from anywhere earlier —
    verification claims typically appear in summary/report spans citing
    retrievals from much earlier, so adjacency alone leaves ~70% of claims
    uncheckable (measured on TELBench). Whole steps, full text; over-cap
    deselection keeps the highest-overlap steps and is logged.
    """
    prev_assistant_idx = -1
    for s in steps:
        if s.index >= claim_step.index:
            break
        if s.action_segment and s.observation_segment is None:
            prev_assistant_idx = s.index

    claim_tokens = _content_tokens(claim_text)
    adjacent: list[Step] = []
    linked: list[tuple[int, Step]] = []
    for s in steps:
        seg = s.observation_segment
        if not seg or s.index >= claim_step.index:
            continue
        if s.index > prev_assistant_idx:
            adjacent.append(s)
            continue
        overlap = sum(1 for t in claim_tokens if t in seg.lower())
        if overlap >= _MIN_LINK_TOKENS:
            linked.append((overlap, s))

    linked.sort(key=lambda x: (-x[0], x[1].index))
    merged = {s.step_id: s for s in adjacent}
    for _, s in linked:
        merged.setdefault(s.step_id, s)

    window = sorted(merged.values(), key=lambda s: s.index)
    if len(window) > _MAX_WINDOW_STEPS:
        # keep adjacent first, then highest-overlap linked
        keep = {s.step_id for s in adjacent[-_MAX_WINDOW_STEPS:]}
        for _, s in linked:
            if len(keep) >= _MAX_WINDOW_STEPS:
                break
            keep.add(s.step_id)
        for s in window:
            if s.step_id not in keep:
                diag.prune("window", s.step_id,
                           f"window cap {_MAX_WINDOW_STEPS} (adjacent + top-overlap kept)")
        window = [s for s in window if s.step_id in keep]
    return window


# ---------------------------------------------------------------------------
# Oracle half: window-scoped consistency judgment
# ---------------------------------------------------------------------------

_CHECK_INSTRUCTIONS = """\
An agent working through a research task made verification/sourcing claims.
Each numbered item gives one claim and the tool/search excerpts the agent had
just received when it made the claim. Judge each claim against ITS OWN
excerpts only — these judgments say nothing about sources elsewhere:

  - "supported": the excerpts contain the claimed fact. Copy the decisive
    text verbatim into "quote".
  - "conflicted": the excerpts contain content contradicting the claim.
    Copy the conflicting text verbatim into "quote".
  - "not_present": the claimed fact does not appear in these excerpts
    (the claim may still be true from other sources — you are only saying
    it is not in what is shown here).

Return ONLY:
{"verdicts": [{"id": 0, "outcome": "supported|conflicted|not_present", "quote": "...", "confidence": 0.9}]}
"""


async def check_source_claims(
    index: TrajectoryIndex,
    *,
    run_id: str = "",
    model: str | None = None,
    session_factory: SessionFactory | None = None,
) -> SourceClaimAnalysis:
    """Detect verification claims and check each against its observation window.

    Best-effort: oracle failure leaves outcomes unknown (P5). One batched
    call for all claims with a non-empty window.
    """
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")

    analysis = SourceClaimAnalysis()
    diag = analysis.diagnostics

    steps = sorted(
        (s for s in index.steps.values() if not run_id or s.run_id == run_id),
        key=lambda s: s.index,
    )
    detected = _claims_from_index(index, steps, diag)
    analysis.n_detected = len(detected)
    if not detected:
        logger.info("source claims: index carries no claims for this run "
                    "(old index without unified extraction, or none asserted)")
        return analysis

    items: list[tuple[Step, str, list[Step]]] = []
    for step, sent in detected:
        window = _observation_window(steps, step, sent, diag)
        if window:
            items.append((step, sent, window))
        else:
            analysis.n_unchecked += 1
            diag.record("source_claim", step.step_id, None, 0.0,
                        f"unchecked (no adjacent observations): {sent[:80]!r}")
    if not items:
        return analysis

    rows = [
        {
            "id": i,
            "claim": sent,
            "excerpts": [{"step": w.step_id, "text": w.observation_segment or ""} for w in window],
        }
        for i, (step, sent, window) in enumerate(items)
    ]
    raw = await _ask_model(
        _CHECK_INSTRUCTIONS, json.dumps(rows, ensure_ascii=False, indent=2),
        model, session_factory=session_factory, purpose="source_claim_check",
    )
    by_id = _index_by_id(raw or [])

    valid = {"supported", "conflicted", "not_present"}
    for i, (step, sent, window) in enumerate(items):
        item = by_id.get(i)
        outcome = str(item.get("outcome", "")) if item else ""
        outcome = outcome if outcome in valid else "unknown"
        quote = str(item.get("quote", "")) if item else ""
        conf = _safe_float(item, "confidence") if item else 0.0
        diag.record("source_claim", step.step_id, outcome, conf, f"claim={sent[:80]!r}")
        analysis.claims.append(SourceClaim(
            step_id=step.step_id,
            claim=sent,
            source_step_ids=tuple(w.step_id for w in window),
            outcome=outcome,
            quote=quote,
            confidence=round(conf, 3),
        ))

    n_by: dict[str, int] = {}
    for c in analysis.claims:
        n_by[c.outcome] = n_by.get(c.outcome, 0) + 1
    logger.info("source claims: detected={} checked={} outcomes={}",
                analysis.n_detected, len(analysis.claims), n_by)
    return analysis
