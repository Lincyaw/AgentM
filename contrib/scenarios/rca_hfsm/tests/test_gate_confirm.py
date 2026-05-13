"""Acceptance #2 — confirm-gate downgrade (design §7.1).

Three sub-cases, one per failing precondition:

1. No negative prediction has been checked → downgrade reason cites the
   falsification gap.
2. Only one worker session has produced positive checks → reason cites the
   independence requirement.
3. Some symptom is not yet linked through a satisfied prediction → reason
   cites unexplained symptoms by id.

Plus one positive-path sub-case so the test file documents what success
looks like.
"""

from __future__ import annotations

from agentm_rca_hfsm.schema import (
    CheckResult,
    Hypothesis,
    Observation,
    Prediction,
    Symptom,
)
from agentm_rca_hfsm.updates import UpdateProposal

from tests._gate_fixtures import install_store_and_gate


def _h_with(*, neg_check_verdict: str | None, pos_workers: list[str]) -> Hypothesis:
    """Build an open hypothesis with one neg + one pos prediction.

    ``neg_check_verdict`` controls whether the negative prediction has a
    ``CheckResult`` and what its verdict says (None = no check). Each entry
    in ``pos_workers`` is a worker_session_id that produced a supporting
    positive check.
    """

    neg = Prediction(
        id="np1",
        hypothesis_id="H1",
        claim="no rotated archive newer than 7d",
        polarity="negative",
    )
    if neg_check_verdict is not None:
        neg.checks.append(
            CheckResult(
                id="c-neg",
                prediction_id="np1",
                worker_session_id="w-neg",
                verdict_proposal=neg_check_verdict,
            )
        )
    pos = Prediction(
        id="pp1",
        hypothesis_id="H1",
        claim="logrotate.service is inactive",
        polarity="positive",
    )
    for i, wid in enumerate(pos_workers):
        pos.checks.append(
            CheckResult(
                id=f"c-pos-{i}",
                prediction_id="pp1",
                worker_session_id=wid,
                verdict_proposal="observations support the prediction",
            )
        )
    return Hypothesis(id="H1", claim="logrotate failed", predictions=[neg, pos])


def test_confirm_without_negative_check_downgrades_to_refine() -> None:
    _, gate, read = install_store_and_gate()
    h = _h_with(neg_check_verdict=None, pos_workers=["w1", "w2"])
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    result = gate.apply(UpdateProposal(op="confirm", target_id="H1"))

    assert result.kind == "downgraded"
    assert "falsification gap" in result.reason
    assert result.downgrade is not None and result.downgrade.op == "refine"
    assert result.applied_id is not None
    assert read.get_hypothesis(result.applied_id) is not None
    assert read.get_hypothesis("H1").status.startswith("refined→")


def test_confirm_with_single_worker_downgrades_to_refine() -> None:
    _, gate, read = install_store_and_gate()
    h = _h_with(neg_check_verdict="no triggering observations", pos_workers=["w1"])
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    result = gate.apply(UpdateProposal(op="confirm", target_id="H1"))

    assert result.kind == "downgraded"
    assert "independence" in result.reason
    assert result.downgrade is not None and result.downgrade.op == "refine"
    assert result.applied_id is not None
    assert read.get_hypothesis("H1").status.startswith("refined→")


def test_confirm_with_unexplained_symptoms_downgrades_to_refine() -> None:
    api, gate, read = install_store_and_gate()
    # Two symptoms, only one will be linked to a supporting observation.
    gate.apply(UpdateProposal(op="record_symptom",
                              symptom=Symptom(id="S1", text="disk full", source="user_intake")))
    gate.apply(UpdateProposal(op="record_symptom",
                              symptom=Symptom(id="S2", text="oom kills", source="user_intake")))

    h = _h_with(neg_check_verdict="no triggering observations",
                pos_workers=["w1", "w2"])
    # Wire the positive checks' observations to S1 only, leaving S2
    # unexplained.
    pos = h.predictions[1]
    for c in pos.checks:
        c.observations.append(
            Observation(
                id=f"o-{c.id}",
                text="disk usage at 99%",
                source_tool_call=f"tc-{c.id}",
                tool_signature="",
                related_symptoms=["S1"],
            )
        )
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    result = gate.apply(UpdateProposal(op="confirm", target_id="H1"))

    assert result.kind == "downgraded"
    assert "unexplained" in result.reason.lower()
    assert "S2" in result.reason
    assert "S1" not in result.reason  # only the still-unexplained symptom is named
    assert result.applied_id is not None
    assert read.get_hypothesis("H1").status.startswith("refined→")


def test_confirm_all_preconditions_met_applies() -> None:
    api, gate, read = install_store_and_gate()
    gate.apply(UpdateProposal(op="record_symptom",
                              symptom=Symptom(id="S1", text="disk full", source="user_intake")))

    h = _h_with(neg_check_verdict="no triggering observations",
                pos_workers=["w1", "w2"])
    for c in h.predictions[1].checks:
        c.observations.append(
            Observation(
                id=f"o-{c.id}",
                text="disk usage at 99%",
                source_tool_call=f"tc-{c.id}",
                tool_signature="",
                related_symptoms=["S1"],
            )
        )
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    result = gate.apply(UpdateProposal(op="confirm", target_id="H1"))

    assert result.kind == "applied"
    assert result.applied_id == "H1"
    assert read.get_hypothesis("H1").status == "confirmed"
