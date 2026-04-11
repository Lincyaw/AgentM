"""Focused regression tests for SanitizerMiddleware gating and triggers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentm.harness.scenario import TrajectorySlot
from agentm.harness.types import LoopContext
from agentm.scenarios.rca.sanitizer.code_sanitizer import CodeSanitizer
from agentm.scenarios.rca.sanitizer.critic_sanitizer import CriticSanitizer
from agentm.scenarios.rca.sanitizer.middleware import SanitizerMiddleware
from agentm.scenarios.rca.sanitizer.models import SanitizerFinding, Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker


class MockResponse:
    def __init__(self, content: str = "", tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


def _ctx(step: int = 0, tool_call_count: int = 0, max_steps: int | None = 30) -> LoopContext:
    return LoopContext(agent_id="orch", step=step, max_steps=max_steps, tool_call_count=tool_call_count, metadata={})


def _finding(code: str, severity: Severity, message: str, details: dict[str, Any] | None = None) -> SanitizerFinding:
    return SanitizerFinding(code=code, severity=severity, message=message, details=details or {})


def _mw(
    *,
    code_findings: list[SanitizerFinding] | None = None,
    critic_async_results: list[SanitizerFinding] | None = None,
    with_critic: bool = False,
    periodic_interval: int = 5,
    max_block_retries: int = 3,
    tool_call_budget: int | None = None,
    trajectory: MagicMock | None = None,
) -> SanitizerMiddleware:
    code_sanitizer = MagicMock(spec=CodeSanitizer)
    code_sanitizer.check.return_value = code_findings or []

    critic_sanitizer: CriticSanitizer | None = None
    if with_critic:
        critic_sanitizer = MagicMock(spec=CriticSanitizer)
        critic_sanitizer.check = MagicMock(return_value=[])  # type: ignore[assignment]
        critic_sanitizer.check_async = MagicMock(return_value=None)  # type: ignore[assignment]
        critic_sanitizer.collect_async_results.return_value = critic_async_results or []

    tracker = InvestigationTracker()
    traj_slot: TrajectorySlot | None = None
    if trajectory is not None:
        traj_slot = TrajectorySlot()
        traj_slot.value = trajectory

    return SanitizerMiddleware(
        code_sanitizer=code_sanitizer,
        critic_sanitizer=critic_sanitizer,
        tracker=tracker,
        hypothesis_store=MagicMock(),
        profile_store=MagicMock(),
        traj_slot=traj_slot,
        periodic_interval=periodic_interval,
        max_block_retries=max_block_retries,
        tool_call_budget=tool_call_budget,
    )


async def _noop_call_next(tool_name: str, tool_args: dict[str, Any]) -> str:
    return "OK"


@pytest.mark.asyncio
async def test_tool_interception_records_dispatch_and_hypothesis_change() -> None:
    mw = _mw()
    ctx = _ctx()
    await mw.on_tool_call(
        "dispatch_agent",
        {"agent_id": "w1", "task_type": "scout", "task": "Check svc-a"},
        _noop_call_next,
        ctx,
    )
    await mw.on_tool_call("update_hypothesis", {"id": "H1", "status": "confirmed"}, _noop_call_next, ctx)

    dispatches = mw._tracker.dispatches()
    assert len(dispatches) == 1
    assert dispatches[0].data["agent_id"] == "w1"
    assert mw._hypothesis_changed is True


@pytest.mark.asyncio
async def test_finalize_blocked_strips_finalize_tag_and_stashes_block_message() -> None:
    mw = _mw(code_findings=[_finding("C1", Severity.BLOCK, "No verify")])
    resp = MockResponse(content="Done. <decision>finalize</decision>")
    await mw.on_llm_end(resp, _ctx())
    assert "<decision>" not in resp.content
    assert mw._pending_block_message is not None
    assert "C1" in mw._pending_block_message


@pytest.mark.asyncio
async def test_findings_are_injected_on_next_llm_start_and_then_cleared() -> None:
    mw = _mw()
    mw._pending_findings = [_finding("E1", Severity.WARN, "Missing upstream")]
    messages: list[Any] = [{"role": "system", "content": "You are an agent."}]
    result = await mw.on_llm_start(messages, _ctx())
    assert len(result) == 2
    assert result[-1]["role"] == "human"
    assert "E1" in result[-1]["content"]
    assert mw._pending_findings == []


@pytest.mark.asyncio
async def test_periodic_trigger_fires_at_configured_interval() -> None:
    mw = _mw(periodic_interval=5)
    trigger_calls: list[str] = []

    def track_check(trigger: str, *args: Any, **kwargs: Any) -> list[SanitizerFinding]:
        trigger_calls.append(trigger)
        return []

    mw._code_sanitizer.check.side_effect = track_check  # type: ignore[union-attr]
    for step in range(10):
        await mw.on_llm_end(MockResponse(content="thinking"), _ctx(step=step))

    assert trigger_calls.count("periodic") == 2


@pytest.mark.asyncio
async def test_budget_degrades_e_code_block_but_keeps_c_code_blocking() -> None:
    e1_block = _finding("E1", Severity.BLOCK, "Anchoring")
    c1_block = _finding("C1", Severity.BLOCK, "No verify", details={"hypothesis_id": "H1"})
    mw = _mw(code_findings=[e1_block, c1_block], tool_call_budget=10)

    resp = MockResponse(content="<decision>finalize</decision>")
    await mw.on_llm_end(resp, _ctx(tool_call_count=10))
    assert "<decision>" not in resp.content
    assert mw._pending_block_message is not None
    assert "C1" in mw._pending_block_message


@pytest.mark.asyncio
async def test_trajectory_records_sanitizer_findings() -> None:
    traj = MagicMock()
    mw = _mw(code_findings=[_finding("E1", Severity.WARN, "test")], trajectory=traj)
    await mw.on_llm_end(MockResponse(content="thinking"), _ctx())
    traj.record_sync.assert_called()
    data = traj.record_sync.call_args.kwargs["data"]
    assert data["trigger"] == "every_round"
    assert len(data["findings"]) == 1


@pytest.mark.asyncio
async def test_async_critic_results_are_drained_and_injected_on_llm_start() -> None:
    async_finding = _finding("P2", Severity.INFO, "dispatch incomplete")
    mw = _mw(with_critic=True, critic_async_results=[async_finding])
    result = await mw.on_llm_start([{"role": "system", "content": "prompt"}], _ctx())
    assert len(result) == 2
    assert "P2" in result[-1]["content"]
