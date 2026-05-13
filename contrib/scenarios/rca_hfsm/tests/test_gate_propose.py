"""Acceptance #1 — negative-prediction-required precondition (design §6.2).

A ``propose(H)`` with zero negative predictions must be rejected, and the
rejection must carry the precise reason string the LLM-facing tool result
layer (commit 3) will surface verbatim. The wording is part of the gate's
public contract — change it here and commit 3 must change in lock-step.
"""

from __future__ import annotations

from agentm_rca_hfsm.schema import Hypothesis, Prediction
from agentm_rca_hfsm.updates import UpdateProposal

from tests._gate_fixtures import install_store_and_gate


_REQUIRED_REASON = "hypothesis must declare at least one negative prediction"


def test_propose_with_zero_negative_predictions_rejected() -> None:
    _, gate, _ = install_store_and_gate()
    h = Hypothesis(
        id="H1",
        claim="logrotate failed",
        predictions=[
            Prediction(
                id="p1",
                hypothesis_id="H1",
                claim="logrotate.service is inactive",
                polarity="positive",
            ),
        ],
    )

    result = gate.apply(UpdateProposal(op="propose", hypothesis=h))

    assert result.kind == "rejected"
    assert result.reason == _REQUIRED_REASON


def test_propose_with_no_predictions_rejected() -> None:
    _, gate, _ = install_store_and_gate()
    h = Hypothesis(id="H1", claim="logrotate failed")

    result = gate.apply(UpdateProposal(op="propose", hypothesis=h))

    assert result.kind == "rejected"
    assert result.reason == _REQUIRED_REASON


def test_propose_with_negative_prediction_applied() -> None:
    _, gate, read = install_store_and_gate()
    h = Hypothesis(
        id="H1",
        claim="logrotate failed",
        predictions=[
            Prediction(
                id="p1",
                hypothesis_id="H1",
                claim="logrotate.service is inactive",
                polarity="positive",
            ),
            Prediction(
                id="p2",
                hypothesis_id="H1",
                claim="no rotated archive newer than 7d",
                polarity="negative",
            ),
        ],
    )

    result = gate.apply(UpdateProposal(op="propose", hypothesis=h))

    assert result.kind == "applied"
    assert result.applied_id == "H1"
    assert read.get_hypothesis("H1") is not None
