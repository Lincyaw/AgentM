"""Commit 4 — scheduler picks the highest-discrimination prediction (§8 default).

Approximation: for each open prediction p, score = count(other open
hypotheses whose predictions overlap with p). Higher score = higher
expected information gain. Ties break on lowest prediction id.

One concise property test — the scheduler is intentionally an
approximation, so exhaustive coverage is not fruitful. We pin the
discrimination preference and the deterministic tie-break.
"""

from __future__ import annotations

from agentm_rca_hfsm.schema import Hypothesis, Prediction
from agentm_rca_hfsm.scheduler import pick_next, score_prediction


def _h(h_id: str, predictions: list[Prediction]) -> Hypothesis:
    return Hypothesis(id=h_id, claim=f"claim-{h_id}", predictions=predictions)


def _p(p_id: str, h_id: str, claim: str) -> Prediction:
    return Prediction(id=p_id, hypothesis_id=h_id, claim=claim, polarity="positive")


def test_scheduler_prefers_overlapping_prediction() -> None:
    """H1 shares its prediction text with H2 (overlap=1 for H1 prediction)
    while H3 has a disjoint prediction. The scheduler must pick H1's
    overlapping prediction first because verifying it would discriminate
    more open hypotheses.
    """

    h1 = _h("H1", [_p("P-h1", "H1", "disk metric crosses 95%")])
    h2 = _h("H2", [_p("P-h2", "H2", "disk metric crosses 95% in window")])
    h3 = _h("H3", [_p("P-h3", "H3", "DNS resolution failure rate spike")])

    choice = pick_next([h1, h2, h3])
    assert choice is not None
    assert choice.id in {"P-h1", "P-h2"}, choice.id
    # The chosen prediction discriminates ≥1 other hypothesis (H1 / H2
    # share). The disjoint H3 prediction scores zero.
    assert score_prediction(choice, [h1, h2, h3]) >= 1
    assert score_prediction(h3.predictions[0], [h1, h2, h3]) == 0


def test_scheduler_breaks_ties_on_lowest_prediction_id() -> None:
    """Two predictions with equal score must resolve deterministically."""

    h1 = _h("HA", [_p("P-aaa", "HA", "shared claim text alpha")])
    h2 = _h("HB", [_p("P-bbb", "HB", "shared claim text alpha")])
    # Both overlap with each other once → tie at score=1.
    choice = pick_next([h1, h2])
    assert choice is not None
    assert choice.id == "P-aaa"


def test_scheduler_returns_none_when_no_open_predictions() -> None:
    assert pick_next([]) is None
    refuted = _h("HX", [_p("P-x", "HX", "x")])
    refuted.status = "refuted"
    assert pick_next([refuted]) is None
