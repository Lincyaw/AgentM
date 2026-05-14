"""Phase 2 C5 — finalize consults rca.judge.investigation_genuine.

After the structural coverage check passes, ``submit_final_report``
consults ``rca.judge.investigation_genuine``. Three sub-cases:

* Judge returns ``"speculation"``: report rejected, session does NOT
  terminate, the reason text appears in the tool result.
* Judge returns ``"genuine_investigation"``: report accepted, session
  terminates with ``ToolTerminate(reason="rca_hfsm:final-report-submitted")``.
* Judge returns ``"unclear"``: report rejected (treat unclear as
  not-genuine — only the canonical positive lets the report through).

These tests bypass the rest of the gate (no FSM machinery exercised)
and drive ``submit_final_report`` directly via its registered tool. The
scripted judge replaces the C5 mimic in the fixture so each test has
deterministic control of the verdict.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import ToolResult, ToolTerminate

from agentm_rca_hfsm.atoms import (
    rca_finalize,
    rca_hgraph_store,
)
from agentm_rca_hfsm.judges import JudgeContext, Verdict

from tests._gate_fixtures import install_full_stack
from tests._gate_fixtures import StubAPI  # noqa: F401 — type clarity


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _tool(api: object, name: str) -> Any:
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


class _ScriptedJudge:
    """Always returns the configured verdict; records calls for assertion."""

    kind = "investigation_genuine"

    def __init__(self, verdict: str, reason: str = "scripted") -> None:
        self._verdict = verdict
        self._reason = reason
        self.calls: list[JudgeContext] = []

    def judge(self, context: JudgeContext) -> Verdict:
        self.calls.append(context)
        return Verdict(
            verdict=self._verdict,
            reason=self._reason,
            confidence="high",
        )


def _install_with_scripted_judge(
    verdict: str, reason: str = "scripted"
) -> tuple[Any, _ScriptedJudge]:
    """Wire up the full stack with a scripted investigation_genuine judge.

    Mirrors ``install_with_fsm`` but swaps the mimic for a scripted
    judge so the test controls the verdict deterministically. The
    finalize atom is the only atom that consults this judge.
    """

    rca_hgraph_store._reset_for_tests()
    api, _gate, _read = install_full_stack()
    scripted = _ScriptedJudge(verdict=verdict, reason=reason)
    api.set_service("rca.judge.investigation_genuine", scripted)
    rca_finalize.install(api, {})
    return api, scripted


def test_finalize_rejects_when_judge_says_speculation() -> None:
    api, scripted = _install_with_scripted_judge(
        "speculation",
        reason="no symptoms recorded — call record_symptom first",
    )

    tool = _tool(api, "submit_final_report")
    result = _run(tool.execute({"root_cause": "I think it's the network"}))

    assert isinstance(result, ToolResult), type(result)
    assert result.is_error is True
    text = _text(result)
    assert "judge=investigation_genuine" in text
    assert "verdict=speculation" in text
    assert "no symptoms recorded" in text
    # Judge was consulted once.
    assert len(scripted.calls) == 1
    # Trajectory carried no symptoms — the judge prompt sees that.
    payload = scripted.calls[0].graph_slice
    assert payload["symptom_count"] == 0
    assert payload["final_report"]["root_cause"] == "I think it's the network"


def test_finalize_accepts_when_judge_says_genuine() -> None:
    api, scripted = _install_with_scripted_judge(
        "genuine_investigation", reason="symptoms recorded; hypotheses verified"
    )

    tool = _tool(api, "submit_final_report")
    result = _run(
        tool.execute(
            {
                "root_cause": "logrotate misconfig",
                "supporting_observations": ["O-1"],
                "refuted_alternatives": [],
            }
        )
    )

    assert isinstance(result, ToolTerminate), type(result)
    assert result.reason == "rca_hfsm:final-report-submitted"
    assert "status=finalized" in _text(result)
    assert "logrotate misconfig" in _text(result)
    assert len(scripted.calls) == 1


def test_finalize_rejects_when_judge_says_unclear() -> None:
    api, scripted = _install_with_scripted_judge(
        "unclear", reason="trajectory shape is contradictory"
    )

    tool = _tool(api, "submit_final_report")
    result = _run(tool.execute({"root_cause": "something"}))

    assert isinstance(result, ToolResult), type(result)
    assert result.is_error is True
    text = _text(result)
    assert "judge=investigation_genuine" in text
    assert "verdict=unclear" in text
    assert "contradictory" in text
    assert len(scripted.calls) == 1
