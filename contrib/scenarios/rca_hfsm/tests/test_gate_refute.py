"""Acceptance #3 — refute-gate downgrade (design §7.2).

Two sub-cases:

1. ``refute(H)`` with no triggered negative AND no steelman check is
   downgraded to ``refine``.
2. ``refute(H)`` with a triggered negative prediction is accepted.

The asymmetry between the two §7.2 grounds (triggered negative OR steelman)
is deliberate, mirroring Popperian falsifiability: a single triggered
negative prediction is sufficient, but "no supporting evidence found"
requires a steelman attempt to avoid lazy refutation.
"""

from __future__ import annotations

from agentm_rca_hfsm.schema import CheckResult, Hypothesis, Prediction
from agentm_rca_hfsm.updates import UpdateProposal

from tests._gate_fixtures import install_store_and_gate


def _open_hypothesis(predictions: list[Prediction]) -> Hypothesis:
    return Hypothesis(id="H1", claim="logrotate failed", predictions=predictions)


def test_refute_without_steelman_or_trigger_downgrades() -> None:
    _, gate, read = install_store_and_gate()
    # Negative prediction with a check, but verdict does not trigger the
    # claim — and no steelman check is present.
    neg = Prediction(
        id="np1",
        hypothesis_id="H1",
        claim="no rotated archive newer than 7d",
        polarity="negative",
        checks=[
            CheckResult(
                id="c-neg",
                prediction_id="np1",
                worker_session_id="w1",
                verdict_proposal="no triggering observations found",
            )
        ],
    )
    gate.apply(UpdateProposal(op="propose", hypothesis=_open_hypothesis([neg])))

    result = gate.apply(UpdateProposal(op="refute", target_id="H1"))

    assert result.kind == "downgraded"
    assert "steelman" in result.reason.lower() or "triggered" in result.reason.lower()
    assert result.downgrade is not None and result.downgrade.op == "refine"
    # §5.2 flip: downgrade no longer applies; parent stays open.
    assert result.applied_id is None
    assert read.get_hypothesis("H1").status == "open"


def test_refute_with_triggered_negative_prediction_applies() -> None:
    _, gate, read = install_store_and_gate()
    neg = Prediction(
        id="np1",
        hypothesis_id="H1",
        claim="no rotated archive newer than 7d",
        polarity="negative",
        checks=[
            CheckResult(
                id="c-neg",
                prediction_id="np1",
                worker_session_id="w1",
                verdict_proposal="rotated archive from 2h ago triggers the claim",
            )
        ],
    )
    gate.apply(UpdateProposal(op="propose", hypothesis=_open_hypothesis([neg])))

    result = gate.apply(UpdateProposal(op="refute", target_id="H1"))

    assert result.kind == "applied"
    assert result.applied_id == "H1"
    assert read.get_hypothesis("H1").status == "refuted"
