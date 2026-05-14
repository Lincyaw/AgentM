"""Commit 4 — acceptance #7: FINALIZE coverage check.

Two sub-cases:

* With unexplained symptoms: ``submit_final_report`` returns
  ``ToolResult(is_error=True)`` whose text lists the unexplained symptom
  ids and does NOT terminate the loop.
* With every symptom explained: ``submit_final_report`` returns
  :class:`ToolTerminate` so the agent loop exits via the
  ``rca_hfsm:final-report-submitted`` reason.
"""

from __future__ import annotations

import asyncio

from agentm.core.abi import ToolResult, ToolTerminate

from agentm_rca_hfsm.schema import (
    CheckResult,
    Hypothesis,
    Observation,
    Prediction,
    Symptom,
)
from agentm_rca_hfsm.updates import UpdateProposal

from tests._gate_fixtures import install_with_fsm


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _tool(api: object, name: str) -> object:
    for t in getattr(api, "tools", []):  # type: ignore[attr-defined]
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not registered")


def _text(result: object) -> str:
    if isinstance(result, ToolTerminate):
        target = result.result
    else:
        target = result
    return "\n".join(c.text for c in target.content)  # type: ignore[attr-defined]


def test_submit_final_report_rejected_when_symptoms_unexplained() -> None:
    api, gate, _read, _fsm = install_with_fsm()

    # One symptom, no covering hypothesis.
    gate.apply(
        UpdateProposal(
            op="record_symptom",
            symptom=Symptom(id="S-orphan", text="orphan symptom", source="user_intake"),
        )
    )

    tool = _tool(api, "submit_final_report")
    result = _run(tool.execute({"root_cause": "blame the network"}))  # type: ignore[attr-defined]

    assert isinstance(result, ToolResult), type(result)
    assert result.is_error is True
    text = _text(result)
    assert "unexplained symptoms remain" in text
    assert "S-orphan" in text


def test_submit_final_report_terminates_when_all_symptoms_explained() -> None:
    api, gate, read, _fsm = install_with_fsm()

    # One symptom, one confirmed hypothesis whose positive prediction is
    # satisfied with an observation linking back to the symptom.
    gate.apply(
        UpdateProposal(
            op="record_symptom",
            symptom=Symptom(id="S-covered", text="disk fills", source="user_intake"),
        )
    )
    pos = Prediction(
        id="P-cov-pos", hypothesis_id="H-cov", claim="positive", polarity="positive",
    )
    neg = Prediction(
        id="P-cov-neg", hypothesis_id="H-cov", claim="negative", polarity="negative",
    )
    h = Hypothesis(id="H-cov", claim="covering H", predictions=[pos, neg])
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    obs = Observation(
        id="O-cov",
        text="confirming fact",
        source_tool_call="sql-cov",
        tool_signature="sig-cov",
        related_symptoms=["S-covered"],
        related_predictions=["P-cov-pos"],
    )
    gate.apply(UpdateProposal(op="record_observation", observation=obs))
    check_pos = CheckResult(
        id="C-pos-1", prediction_id=pos.id, worker_session_id="w-cov",
        observations=[obs],
        verdict_proposal="observations support the prediction",
    )
    gate.apply(
        UpdateProposal(op="attach_check", prediction_id=pos.id, check=check_pos)
    )
    # Negative check — no triggering observation.
    check_neg = CheckResult(
        id="C-neg-1", prediction_id=neg.id, worker_session_id="w-cov-2",
        observations=[],
        verdict_proposal="no contradicting evidence found",
    )
    gate.apply(
        UpdateProposal(op="attach_check", prediction_id=neg.id, check=check_neg)
    )
    # Second independent positive — required by §7.1 independence rule.
    obs2 = Observation(
        id="O-cov-2",
        text="independent confirmation",
        source_tool_call="sql-cov-2",
        tool_signature="sig-cov-2",
        related_symptoms=["S-covered"],
        related_predictions=["P-cov-pos"],
    )
    gate.apply(UpdateProposal(op="record_observation", observation=obs2))
    check_pos2 = CheckResult(
        id="C-pos-2", prediction_id=pos.id, worker_session_id="w-cov-3",
        observations=[obs2],
        verdict_proposal="observations support the prediction",
    )
    gate.apply(
        UpdateProposal(op="attach_check", prediction_id=pos.id, check=check_pos2)
    )

    # Now confirm via the gate.
    confirm_result = gate.apply(UpdateProposal(op="confirm", target_id=h.id))
    assert confirm_result.kind == "applied", confirm_result.reason
    assert read.get_hypothesis(h.id).status == "confirmed"
    assert read.get_unexplained_symptoms() == []

    tool = _tool(api, "submit_final_report")
    result = _run(
        tool.execute(  # type: ignore[attr-defined]
            {
                "root_cause": "covering H is the cause",
                "supporting_observations": ["O-cov", "O-cov-2"],
                "refuted_alternatives": [],
            }
        )
    )

    assert isinstance(result, ToolTerminate), type(result)
    assert result.reason == "rca_hfsm:final-report-submitted"
    assert "status=finalized" in _text(result)
