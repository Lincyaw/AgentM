"""Information-gain verification scheduler (design §8 default).

Pure functions: given the open hypotheses and an emergent "next prediction
to verify" question, score each open prediction by how many *other* open
hypotheses it would discriminate. The prediction with the highest score is
the next one to verify; ties break on the lowest prediction id so the
choice is deterministic across runs.

Approximation, not Bayesian-correct. "Overlap" between two predictions is
a naive substring test on their free-text ``claim``: if either claim
appears inside the other (case-insensitive, whitespace-normalised), the
predictions are considered to overlap. Phase 1 keeps this cheap; Phase 2
may swap in an embedding-based similarity once trace data justifies the
cost.

This module is **not** an atom — no ``MANIFEST``, no ``install``. The FSM
policy atom imports it as a pure helper.
"""

from __future__ import annotations

from agentm_rca.hfsm.schema import Hypothesis, Prediction


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _claims_overlap(a: str, b: str) -> bool:
    """Substring overlap on normalised claims.

    Symmetric: ``a in b`` or ``b in a``. Empty strings never overlap with
    anything (return ``False``) so degenerate predictions don't artificially
    inflate scores.
    """

    na = _normalise(a)
    nb = _normalise(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def open_predictions(hypotheses: list[Hypothesis]) -> list[Prediction]:
    """Flatten predictions of every open hypothesis that have no check yet.

    A prediction "needs verification" iff it currently carries zero
    ``CheckResult`` entries. Predictions with checks already feed into the
    gate's confirm/refute path, not the scheduler.
    """

    out: list[Prediction] = []
    for h in hypotheses:
        if h.status != "open":
            continue
        for p in h.predictions:
            if not p.checks:
                out.append(p)
    return out


def score_prediction(target: Prediction, hypotheses: list[Hypothesis]) -> int:
    """Count how many *other* open hypotheses have a prediction overlapping ``target``.

    Higher score = higher discrimination potential: a verification result
    on ``target`` would change the status of more open hypotheses.
    """

    score = 0
    for h in hypotheses:
        if h.status != "open":
            continue
        if h.id == target.hypothesis_id:
            continue
        if any(_claims_overlap(target.claim, p.claim) for p in h.predictions):
            score += 1
    return score


def pick_next(hypotheses: list[Hypothesis]) -> Prediction | None:
    """Return the highest-information-gain open prediction, or ``None``.

    Ties on score break on the lowest ``Prediction.id`` so the choice is
    deterministic. Returns ``None`` when every open hypothesis has all its
    predictions already checked (the scheduler has nothing to pick).
    """

    candidates = open_predictions(hypotheses)
    if not candidates:
        return None
    scored = [(score_prediction(p, hypotheses), p.id, p) for p in candidates]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[0][2]


__all__ = [
    "open_predictions",
    "pick_next",
    "score_prediction",
]
