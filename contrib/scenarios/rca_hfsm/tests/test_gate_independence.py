"""Phase 1 independence semantics — ``worker_session_id`` only (design §7.1).

The §7.1 confirm gate requires "≥1 independent positive verification". Phase
1 defines independence as **distinct worker_session_id**; brief-slice
disjointness (acceptance #9) is Phase 2 and explicitly NOT checked here.

The test asserts the symmetric pair: two checks with identical observation
payloads but different session ids ARE treated as independent; two checks
with the same session id are NOT.
"""

from __future__ import annotations

from agentm_rca_hfsm.schema import (
    CheckResult,
    Hypothesis,
    Observation,
    Prediction,
    Symptom,
)
from agentm_rca_hfsm.updates import (
    UpdateProposal,
    independent_positive_workers,
)

from tests._gate_fixtures import install_store_and_gate


_IDENTICAL_OBS = Observation(
    id="o-shared",
    text="disk usage at 99% on /var",
    source_tool_call="tc-shared",
    tool_signature="",
    related_symptoms=["S1"],
)


def _h_with_positive_workers(worker_ids: list[str]) -> Hypothesis:
    """Build a hypothesis whose positive prediction carries one supporting
    check per ``worker_ids`` entry. All checks share identical observations
    so the only varying axis is ``worker_session_id`` — Phase 1's exact
    independence axis.
    """

    neg = Prediction(
        id="np1",
        hypothesis_id="H1",
        claim="no rotated archive newer than 7d",
        polarity="negative",
        checks=[
            CheckResult(
                id="c-neg",
                prediction_id="np1",
                worker_session_id="w-neg",
                verdict_proposal="no triggering observations found",
            )
        ],
    )
    pos = Prediction(
        id="pp1",
        hypothesis_id="H1",
        claim="logrotate.service is inactive",
        polarity="positive",
    )
    for i, wid in enumerate(worker_ids):
        pos.checks.append(
            CheckResult(
                id=f"c-pos-{i}",
                prediction_id="pp1",
                worker_session_id=wid,
                observations=[
                    Observation(
                        id=f"{_IDENTICAL_OBS.id}-{i}",
                        text=_IDENTICAL_OBS.text,
                        source_tool_call=_IDENTICAL_OBS.source_tool_call,
                        tool_signature=_IDENTICAL_OBS.tool_signature,
                        related_symptoms=list(_IDENTICAL_OBS.related_symptoms),
                    )
                ],
                verdict_proposal="observations support the prediction",
            )
        )
    return Hypothesis(id="H1", claim="logrotate failed", predictions=[neg, pos])


def test_same_session_id_not_independent() -> None:
    h = _h_with_positive_workers(["w1", "w1"])
    assert independent_positive_workers(h) == {"w1"}


def test_distinct_session_ids_are_independent() -> None:
    h = _h_with_positive_workers(["w1", "w2"])
    assert independent_positive_workers(h) == {"w1", "w2"}


def test_confirm_blocks_on_single_session_even_with_identical_obs() -> None:
    """End-to-end: identical observation payloads do NOT compensate for a
    shared ``worker_session_id``. This is the load-bearing Phase 1 claim —
    independence is structural (who ran the check), not semantic (what the
    check saw).
    """

    _, gate, read = install_store_and_gate()
    gate.apply(UpdateProposal(op="record_symptom",
                              symptom=Symptom(id="S1", text="disk full", source="user_intake")))
    h = _h_with_positive_workers(["w-same", "w-same"])
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    result = gate.apply(UpdateProposal(op="confirm", target_id="H1"))

    assert result.kind == "downgraded"
    assert "independence" in result.reason
    assert read.get_hypothesis("H1").status.startswith("refined→")


def test_confirm_passes_with_distinct_sessions_and_full_coverage() -> None:
    _, gate, read = install_store_and_gate()
    gate.apply(UpdateProposal(op="record_symptom",
                              symptom=Symptom(id="S1", text="disk full", source="user_intake")))
    h = _h_with_positive_workers(["w-a", "w-b"])
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    result = gate.apply(UpdateProposal(op="confirm", target_id="H1"))

    assert result.kind == "applied"
    assert read.get_hypothesis("H1").status == "confirmed"
