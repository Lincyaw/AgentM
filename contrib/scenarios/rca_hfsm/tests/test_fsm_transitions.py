"""Commit 4 — FSM advances on graph mutations through INTAKE→...→FINALIZE.

Drives the FSM by directly invoking the registered evidence tools so the
test exercises the same gate-emit → FSM-policy path the live scenario
uses. The smoke test in commit 5 will compose this with a stub LLM
provider; here we want to assert the policy's structural transitions are
correct independent of LLM behaviour.

The acceptance criterion this fails-stops: the FSM must traverse every
state the design declares reachable in Phase 1; FINALIZE must only become
reachable after a confirm whose coverage check passes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import AgentStartEvent

from tests._gate_fixtures import install_with_fsm


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _tool(api: object, name: str) -> object:
    for t in getattr(api, "tools", []):  # type: ignore[attr-defined]
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not registered")


def _text(result: object) -> str:
    return "\n".join(c.text for c in result.content)  # type: ignore[attr-defined]


def test_fsm_traverses_full_lifecycle_to_finalize() -> None:
    api, _gate, read, fsm = install_with_fsm()
    assert fsm.state == "INTAKE"

    # INTAKE → OBSERVE on first symptom.
    sym_tool = _tool(api, "record_symptom")
    _run(sym_tool.execute({"text": "disk fills at 14:32 UTC"}))  # type: ignore[attr-defined]
    assert fsm.state == "OBSERVE", fsm.history

    # Capture the symptom id so the confirming observation can link to it.
    sym_id = read.get_symptoms()[0].id

    # OBSERVE → HYPOTHESIZE → VERIFY on propose.
    propose = _tool(api, "propose_hypothesis")
    _run(
        propose.execute(  # type: ignore[attr-defined]
            {
                "claim": "logrotate failed",
                "predictions": [
                    {"claim": "logrotate.service inactive", "polarity": "positive"},
                    {"claim": "no rotated archive newer than 7d", "polarity": "negative"},
                ],
            }
        )
    )
    assert fsm.state == "VERIFY", fsm.history
    assert "HYPOTHESIZE" in fsm.history

    leaf = read.get_open_leaves()[0]
    pos_pred = next(p for p in leaf.predictions if p.polarity == "positive")
    neg_pred = next(p for p in leaf.predictions if p.polarity == "negative")

    attach = _tool(api, "attach_check")
    # First positive check from worker w1.
    _run(
        attach.execute(  # type: ignore[attr-defined]
            {
                "hypothesis_id": leaf.id,
                "prediction_id": pos_pred.id,
                "worker_session_id": "w1",
                "observations": [
                    {
                        "text": "logrotate.service is inactive",
                        "source_tool_call": "sql-1",
                        "related_symptoms": [sym_id],
                        "related_predictions": [pos_pred.id],
                    }
                ],
                "interpretation": {
                    "proposed_update": "confirm H",
                    "reasoning": "matches",
                    "confidence": "high",
                },
                "verdict_proposal": "observations support the prediction",
            }
        )
    )
    assert fsm.state == "JUDGE", fsm.history

    # Independent positive check from worker w2 — required by §7.1.
    _run(
        attach.execute(  # type: ignore[attr-defined]
            {
                "hypothesis_id": leaf.id,
                "prediction_id": pos_pred.id,
                "worker_session_id": "w2",
                "observations": [
                    {
                        "text": "second worker confirms inactive",
                        "source_tool_call": "sql-2",
                        "related_symptoms": [sym_id],
                        "related_predictions": [pos_pred.id],
                    }
                ],
                "interpretation": {
                    "proposed_update": "confirm H",
                    "reasoning": "matches",
                    "confidence": "high",
                },
                "verdict_proposal": "observations support the prediction",
            }
        )
    )
    assert fsm.state == "JUDGE"

    # Negative-prediction check — no triggering observation.
    _run(
        attach.execute(  # type: ignore[attr-defined]
            {
                "hypothesis_id": leaf.id,
                "prediction_id": neg_pred.id,
                "worker_session_id": "w3",
                "observations": [
                    {
                        "text": "no rotated archive present",
                        "source_tool_call": "sql-3",
                        "related_symptoms": [sym_id],
                        "related_predictions": [neg_pred.id],
                    }
                ],
                "interpretation": {
                    "proposed_update": "negative not triggered",
                    "reasoning": "as expected",
                    "confidence": "high",
                },
                # Deliberately phrased so the gate's regex does NOT see
                # "triggered" — the negative is satisfied.
                "verdict_proposal": "no observations were found that match",
            }
        )
    )
    assert fsm.state == "JUDGE"

    # JUDGE → FINALIZE on confirm whose coverage check passes.
    explicit = _tool(api, "propose_update")
    result = _run(
        explicit.execute({"op": "confirm", "target_id": leaf.id})  # type: ignore[attr-defined]
    )
    assert _text(result).startswith("status=applied"), _text(result)
    assert fsm.state == "FINALIZE", fsm.history


def test_finalize_state_filters_tools_to_submit_final_report_only() -> None:
    """When the FSM enters FINALIZE before agent_start fires, the visible
    tool catalog narrows to ``submit_final_report``.

    Mirrors how the live scenario will behave: by the time
    ``AgentStartEvent`` fires, the prior trace state (loaded from a
    persisted session) has already advanced the FSM, and the policy atom
    locks the tool surface to the FINALIZE allow-list.
    """

    api, _gate, _read, fsm = install_with_fsm()
    fsm.transition("FINALIZE")

    api.events.fire_handlers(
        AgentStartEvent.CHANNEL, AgentStartEvent(messages=[])
    )

    tool_names = {t.name for t in api.tools}
    assert tool_names == {"submit_final_report"}, tool_names


def test_record_observation_does_not_advance_fsm() -> None:
    """Observations are not state-advancing — they're facts.

    The policy explicitly chooses not to react to ``record_observation``
    mutations so a worker streaming evidence into L1 mid-VERIFY does not
    yank the FSM out of VERIFY before the check is attached.
    """

    api, _gate, _read, fsm = install_with_fsm()
    sym = _tool(api, "record_symptom")
    _run(sym.execute({"text": "S1"}))  # type: ignore[attr-defined]
    propose = _tool(api, "propose_hypothesis")
    _run(
        propose.execute(  # type: ignore[attr-defined]
            {
                "claim": "H",
                "predictions": [
                    {"claim": "x", "polarity": "negative"},
                ],
            }
        )
    )
    assert fsm.state == "VERIFY"

    state_before = fsm.state
    rec = _tool(api, "record_observation")
    _run(
        rec.execute(  # type: ignore[attr-defined]
            {"text": "fact", "source_tool_call": "sql-x"}
        )
    )
    assert fsm.state == state_before, fsm.history


def _unused_helper(_: dict[str, Any]) -> None:  # pragma: no cover - silence linters
    return None
