"""Commit 4 — brief builder enforces falsification framing + hypothesis blinding.

Acceptance #9-flavoured property test (not the full Phase-2 disjointness
check): given a hypothesis with a positive and a negative prediction, the
``mode='verify'`` brief MUST contain a falsification verb (refute /
contradict) and MUST NOT contain the parent hypothesis's claim text by
default (``blind=True``).
"""

from __future__ import annotations

from agentm_rca_hfsm.schema import Hypothesis, Prediction
from agentm_rca_hfsm.updates import UpdateProposal

from tests._gate_fixtures import install_with_fsm


_FALSIFICATION_VERBS = ("refute", "refutes", "contradict", "contradicts")


def _seed_hypothesis(gate: object, read: object) -> tuple[str, str, str, str]:
    """Insert a hypothesis with one positive + one negative prediction.

    Returns ``(hypothesis_id, positive_prediction_id, negative_prediction_id,
    parent_claim_text)`` so the test can assert on visibility downstream.
    """

    claim = "logrotate.timer is the smoking gun"
    pos = Prediction(
        id="P-pos-1", hypothesis_id="H-fb-1",
        claim="logrotate.service is inactive", polarity="positive",
    )
    neg = Prediction(
        id="P-neg-1", hypothesis_id="H-fb-1",
        claim="no rotated archive newer than 7 days", polarity="negative",
    )
    h = Hypothesis(
        id="H-fb-1", claim=claim, predictions=[pos, neg], rationale="seed",
    )
    result = gate.apply(UpdateProposal(op="propose", hypothesis=h))  # type: ignore[attr-defined]
    assert result.kind == "applied", result.reason
    return h.id, pos.id, neg.id, claim


def test_verify_brief_uses_falsification_framing() -> None:
    api, gate, read, _fsm = install_with_fsm()
    h_id, pos_id, _neg_id, _claim = _seed_hypothesis(gate, read)

    brief = api.get_service("rca.brief")
    text = brief(h_id, pos_id, mode="verify").lower()

    assert any(verb in text for verb in _FALSIFICATION_VERBS), text


def test_verify_brief_blinds_parent_hypothesis_claim() -> None:
    api, gate, read, _fsm = install_with_fsm()
    h_id, pos_id, _neg_id, parent_claim = _seed_hypothesis(gate, read)

    brief = api.get_service("rca.brief")
    text = brief(h_id, pos_id, mode="verify")

    # Default blinding hides the parent hypothesis claim — the worker sees
    # only the prediction text + L1 slices. The exact parent claim string
    # must not appear anywhere in the brief.
    assert parent_claim not in text, text


def test_steelman_brief_uses_supporting_framing() -> None:
    api, gate, read, _fsm = install_with_fsm()
    h_id, pos_id, _neg_id, _claim = _seed_hypothesis(gate, read)

    brief = api.get_service("rca.brief")
    text = brief(h_id, pos_id, mode="steelman").lower()

    # Steelman mode flips the framing — the worker is asked to FIND
    # supporting evidence so the gate can later treat a steelman-failed
    # attempt as structural justification for refute (§7.2).
    assert "supporting evidence" in text or "support" in text, text


def test_unblinded_brief_does_show_parent_claim() -> None:
    """Sanity check: ``blind=False`` is the Phase 2 opt-out path."""

    api, gate, read, _fsm = install_with_fsm()
    h_id, pos_id, _neg_id, parent_claim = _seed_hypothesis(gate, read)

    brief = api.get_service("rca.brief")
    text = brief(h_id, pos_id, mode="verify", blind=False)
    assert parent_claim in text
