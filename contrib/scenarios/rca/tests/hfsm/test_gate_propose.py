"""Acceptance #1 — propose-path light precondition + confirm-time falsification.

Phase 2 (C2) flipped the "≥1 negative prediction" rule from a propose-time
structural check to a confirm-time judge call (design §4.4 +
``rca.judge.falsified_genuinely``). Propose now only enforces the
shape-level invariants: payload present, claim non-empty, ≥1 prediction.
A hypothesis with zero negative predictions IS accepted by propose; it
is rejected at confirm-time by the falsified-genuinely judge with the
canonical "falsification gap" reason.

Three sub-cases:

1. Propose with at least one prediction (any polarity) → applied.
2. Propose with zero predictions → rejected on the shape rule.
3. Propose with no negative prediction, then confirm → downgraded via
   the falsified-genuinely judge. This is the "moved to runtime"
   half of the Phase-1 acceptance: the structural enforcement is gone
   but the equivalent decision still fires, just later in the trace.
"""

from __future__ import annotations

from agentm_rca.hfsm.schema import Hypothesis, Prediction
from agentm_rca.hfsm.updates import UpdateProposal

from tests.hfsm._gate_fixtures import install_store_and_gate


def test_propose_with_only_positive_prediction_now_applies() -> None:
    """Phase 2: propose-time no longer rejects on missing negative.

    The structural rule moved into ``rca.judge.falsified_genuinely`` at
    confirm-time. See ``test_confirm_without_negative_check_downgrades``
    for the equivalent assertion on the downstream operator.
    """

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
        ],
    )

    result = gate.apply(UpdateProposal(op="propose", hypothesis=h))

    assert result.kind == "applied"
    assert result.applied_id == "H1"
    assert read.get_hypothesis("H1") is not None


def test_propose_with_no_predictions_rejected() -> None:
    """Shape rule that survives the refactor: ≥1 prediction required."""

    _, gate, _ = install_store_and_gate()
    h = Hypothesis(id="H1", claim="logrotate failed")

    result = gate.apply(UpdateProposal(op="propose", hypothesis=h))

    assert result.kind == "rejected"
    assert "at least one prediction" in result.reason


def test_propose_with_empty_claim_rejected() -> None:
    """Shape rule: claim must be non-empty (defends against payload errors)."""

    _, gate, _ = install_store_and_gate()
    h = Hypothesis(
        id="H1",
        claim="   ",
        predictions=[
            Prediction(
                id="p1",
                hypothesis_id="H1",
                claim="something",
                polarity="positive",
            ),
        ],
    )

    result = gate.apply(UpdateProposal(op="propose", hypothesis=h))

    assert result.kind == "rejected"
    assert "claim" in result.reason


