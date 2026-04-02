"""Tests for SanitizerMiddleware — Phase 4 of the Investigation Sanitizer."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """Minimal response object for on_llm_end."""

    def __init__(self, content: str = "", tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


def _make_ctx(
    step: int = 0,
    tool_call_count: int = 0,
    max_steps: int | None = 30,
) -> LoopContext:
    return LoopContext(
        agent_id="test-orch",
        step=step,
        max_steps=max_steps,
        tool_call_count=tool_call_count,
        metadata={},
    )


def _make_finding(
    code: str = "E1",
    severity: Severity = Severity.WARN,
    message: str = "test finding",
    details: dict[str, Any] | None = None,
) -> SanitizerFinding:
    return SanitizerFinding(
        code=code,
        severity=severity,
        message=message,
        details=details or {},
    )


def _build_middleware(
    code_findings: list[SanitizerFinding] | None = None,
    critic_findings: list[SanitizerFinding] | None = None,
    critic_async_results: list[SanitizerFinding] | None = None,
    with_critic: bool = False,
    periodic_interval: int = 5,
    max_block_retries: int = 3,
    tool_call_budget: int | None = None,
    trajectory: MagicMock | None = None,
) -> SanitizerMiddleware:
    """Build a SanitizerMiddleware with mock dependencies."""
    code_sanitizer = MagicMock(spec=CodeSanitizer)
    code_sanitizer.check.return_value = code_findings or []

    critic_sanitizer: CriticSanitizer | None = None
    if with_critic:
        critic_sanitizer = MagicMock(spec=CriticSanitizer)
        critic_sanitizer.check = MagicMock(return_value=critic_findings or [])  # type: ignore[assignment]
        critic_sanitizer.check_async = MagicMock(return_value=None)  # type: ignore[assignment]
        critic_sanitizer.collect_async_results.return_value = critic_async_results or []

    tracker = InvestigationTracker()
    hypothesis_store = MagicMock()
    profile_store = MagicMock()

    traj_slot: TrajectorySlot | None = None
    if trajectory is not None:
        traj_slot = TrajectorySlot()
        traj_slot.value = trajectory

    mw = SanitizerMiddleware(
        code_sanitizer=code_sanitizer,
        critic_sanitizer=critic_sanitizer,
        tracker=tracker,
        hypothesis_store=hypothesis_store,
        profile_store=profile_store,
        traj_slot=traj_slot,
        periodic_interval=periodic_interval,
        max_block_retries=max_block_retries,
        tool_call_budget=tool_call_budget,
    )
    return mw


async def _noop_call_next(tool_name: str, tool_args: dict[str, Any]) -> str:
    return "OK"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolInterception:
    """Test on_tool_call event recording."""

    @pytest.mark.asyncio
    async def test_dispatch_agent_records_dispatch_event(self) -> None:
        mw = _build_middleware()
        ctx = _make_ctx()

        await mw.on_tool_call(
            "dispatch_agent",
            {"agent_id": "w1", "task_type": "scout", "task": "Check `svc-a`"},
            _noop_call_next,
            ctx,
        )

        dispatches = mw._tracker.dispatches()
        assert len(dispatches) == 1
        assert dispatches[0].data["agent_id"] == "w1"
        assert dispatches[0].data["task_type"] == "scout"
        assert "svc-a" in dispatches[0].data["target_services"]

    @pytest.mark.asyncio
    async def test_update_hypothesis_records_event_and_sets_flag(self) -> None:
        mw = _build_middleware()
        ctx = _make_ctx()

        await mw.on_tool_call(
            "update_hypothesis",
            {"id": "H1", "status": "confirmed"},
            _noop_call_next,
            ctx,
        )

        changes = mw._tracker.hypothesis_changes()
        assert len(changes) == 1
        assert changes[0].data["hypothesis_id"] == "H1"
        assert changes[0].data["new_status"] == "confirmed"
        assert mw._hypothesis_changed is True

    @pytest.mark.asyncio
    async def test_remove_hypothesis_records_event(self) -> None:
        mw = _build_middleware()
        ctx = _make_ctx()

        await mw.on_tool_call(
            "remove_hypothesis",
            {"id": "H2"},
            _noop_call_next,
            ctx,
        )

        changes = mw._tracker.hypothesis_changes()
        assert len(changes) == 1
        assert changes[0].data["new_status"] == "removed"
        assert mw._hypothesis_changed is True

    @pytest.mark.asyncio
    async def test_update_service_profile_records_tool_call(self) -> None:
        mw = _build_middleware()
        ctx = _make_ctx()

        await mw.on_tool_call(
            "update_service_profile",
            {"service_name": "svc-x"},
            _noop_call_next,
            ctx,
        )

        events = mw._tracker.tool_calls_for("update_service_profile")
        assert len(events) == 1
        assert events[0].data["service_name"] == "svc-x"

    @pytest.mark.asyncio
    async def test_query_service_profile_records_tool_call(self) -> None:
        mw = _build_middleware()
        ctx = _make_ctx()

        await mw.on_tool_call(
            "query_service_profile",
            {"service_name": "svc-y"},
            _noop_call_next,
            ctx,
        )

        events = mw._tracker.tool_calls_for("query_service_profile")
        assert len(events) == 1
        assert events[0].data["service_name"] == "svc-y"


class TestFinalizeGate:
    """Test finalize blocking/passing in on_llm_end."""

    @pytest.mark.asyncio
    async def test_finalize_blocked_strips_tag(self) -> None:
        block_finding = _make_finding("C1", Severity.BLOCK, "No verify task")
        mw = _build_middleware(code_findings=[block_finding])
        ctx = _make_ctx()

        resp = MockResponse(content="Analysis done. <decision>finalize</decision>")
        await mw.on_llm_end(resp, ctx)

        # Finalize tag should be stripped
        assert "<decision>" not in resp.content
        # Block message should be stored separately
        assert mw._pending_block_message is not None
        assert "C1" in mw._pending_block_message

    @pytest.mark.asyncio
    async def test_finalize_pass_when_no_blocks(self) -> None:
        warn_finding = _make_finding("E1", Severity.WARN, "Missing upstream")
        mw = _build_middleware(code_findings=[warn_finding])
        ctx = _make_ctx()

        resp = MockResponse(content="Done. <decision>finalize</decision>")
        await mw.on_llm_end(resp, ctx)

        # Content should not be modified (finalize tag remains)
        assert "<decision>finalize</decision>" in resp.content

    @pytest.mark.asyncio
    async def test_no_finalize_tag_no_gate(self) -> None:
        """If there's no finalize tag, pre_finalize checks should not run."""
        mw = _build_middleware()
        ctx = _make_ctx()

        resp = MockResponse(content="Dispatching another worker.")
        await mw.on_llm_end(resp, ctx)

        # code_sanitizer.check should be called for every_round but NOT pre_finalize
        calls = mw._code_sanitizer.check.call_args_list  # type: ignore[union-attr]
        triggers = [call.args[0] for call in calls]
        assert "pre_finalize" not in triggers
        assert "every_round" in triggers


class TestFindingInjection:
    """Test on_llm_start finding injection."""

    @pytest.mark.asyncio
    async def test_findings_injected_as_human_message(self) -> None:
        mw = _build_middleware()
        mw._pending_findings = [
            _make_finding("E1", Severity.WARN, "Missing upstream obs"),
        ]
        ctx = _make_ctx()
        messages: list[Any] = [{"role": "system", "content": "You are an agent."}]

        result = await mw.on_llm_start(messages, ctx)

        assert len(result) == 2
        assert result[-1]["role"] == "human"
        assert "E1" in result[-1]["content"]
        assert "WARN" in result[-1]["content"]
        # Pending should be cleared
        assert len(mw._pending_findings) == 0

    @pytest.mark.asyncio
    async def test_no_findings_no_injection(self) -> None:
        mw = _build_middleware()
        ctx = _make_ctx()
        messages: list[Any] = [{"role": "system", "content": "You are an agent."}]

        result = await mw.on_llm_start(messages, ctx)

        assert len(result) == 1  # unchanged


class TestRetryDegradation:
    """Test degradation after max_block_retries."""

    @pytest.mark.asyncio
    async def test_degradation_after_3_blocks(self) -> None:
        finding = _make_finding(
            "C1", Severity.BLOCK, "No verify task",
            details={"hypothesis_id": "H1"},
        )
        mw = _build_middleware(
            code_findings=[finding],
            max_block_retries=3,
        )
        ctx = _make_ctx()

        # First 3 attempts — should remain BLOCK
        for i in range(3):
            mw._code_sanitizer.check.return_value = [finding]  # type: ignore[union-attr]
            resp = MockResponse(content=f"<decision>finalize</decision> try {i}")
            await mw.on_llm_end(resp, ctx)
            mw._pending_findings.clear()

        # 4th attempt — should be degraded to WARN
        mw._code_sanitizer.check.return_value = [finding]  # type: ignore[union-attr]
        resp = MockResponse(content="<decision>finalize</decision> try 4")
        await mw.on_llm_end(resp, ctx)

        # The finalize tag should NOT be stripped (no BLOCK findings remain)
        assert "<decision>finalize</decision>" in resp.content


class TestBudgetDegradation:
    """Test budget-aware severity degradation."""

    @pytest.mark.asyncio
    async def test_e_code_block_degraded_when_budget_exhausted(self) -> None:
        e1_block = _make_finding("E1", Severity.BLOCK, "Anchoring bias")
        c1_block = _make_finding(
            "C1", Severity.BLOCK, "No verify",
            details={"hypothesis_id": "H1"},
        )
        mw = _build_middleware(
            code_findings=[e1_block, c1_block],
            tool_call_budget=10,
        )
        # Budget exhausted
        ctx = _make_ctx(tool_call_count=10)

        resp = MockResponse(content="<decision>finalize</decision>")
        await mw.on_llm_end(resp, ctx)

        # E1 should be degraded (not blocking), C1 should still block
        # The finalize tag should be stripped because C1 still blocks
        assert "<decision>" not in resp.content
        assert mw._pending_block_message is not None
        assert "C1" in mw._pending_block_message

    @pytest.mark.asyncio
    async def test_no_degradation_when_budget_not_exhausted(self) -> None:
        e1_block = _make_finding("E1", Severity.BLOCK, "Anchoring bias")
        mw = _build_middleware(
            code_findings=[e1_block],
            tool_call_budget=10,
        )
        ctx = _make_ctx(tool_call_count=5)

        resp = MockResponse(content="<decision>finalize</decision>")
        await mw.on_llm_end(resp, ctx)

        # E1 should still block
        assert "<decision>" not in resp.content


class TestPeriodicTrigger:
    """Test periodic check trigger at correct intervals."""

    @pytest.mark.asyncio
    async def test_periodic_fires_at_interval(self) -> None:
        mw = _build_middleware(periodic_interval=5)

        trigger_calls: list[str] = []

        def track_check(trigger: str, *args: Any, **kwargs: Any) -> list[SanitizerFinding]:
            trigger_calls.append(trigger)
            return []

        mw._code_sanitizer.check.side_effect = track_check  # type: ignore[union-attr]

        # Run 10 rounds — each needs its own ctx with incrementing step
        for step in range(10):
            resp = MockResponse(content="thinking...")
            await mw.on_llm_end(resp, _make_ctx(step=step))

        periodic_count = trigger_calls.count("periodic")
        assert periodic_count == 2  # at round 5 (step=4) and 10 (step=9)

    @pytest.mark.asyncio
    async def test_periodic_does_not_fire_before_interval(self) -> None:
        mw = _build_middleware(periodic_interval=5)

        trigger_calls: list[str] = []

        def track_check(trigger: str, *args: Any, **kwargs: Any) -> list[SanitizerFinding]:
            trigger_calls.append(trigger)
            return []

        mw._code_sanitizer.check.side_effect = track_check  # type: ignore[union-attr]

        # Run 4 rounds (steps 0-3 → rounds 1-4, none divisible by 5)
        for step in range(4):
            resp = MockResponse(content="thinking...")
            await mw.on_llm_end(resp, _make_ctx(step=step))

        assert "periodic" not in trigger_calls


class TestHypothesisChangeTrigger:
    """Test hypothesis_change trigger."""

    @pytest.mark.asyncio
    async def test_hypothesis_change_trigger_fires(self) -> None:
        mw = _build_middleware()
        ctx = _make_ctx()

        # Simulate hypothesis change via on_tool_call
        await mw.on_tool_call(
            "update_hypothesis",
            {"id": "H1", "status": "confirmed"},
            _noop_call_next,
            ctx,
        )

        trigger_calls: list[str] = []

        def track_check(trigger: str, *args: Any, **kwargs: Any) -> list[SanitizerFinding]:
            trigger_calls.append(trigger)
            return []

        mw._code_sanitizer.check.side_effect = track_check  # type: ignore[union-attr]

        resp = MockResponse(content="Updated hypothesis.")
        await mw.on_llm_end(resp, ctx)

        assert "hypothesis_change" in trigger_calls
        assert mw._hypothesis_changed is False  # reset after check


class TestTrajectoryRecording:
    """Test trajectory recording of sanitizer events."""

    @pytest.mark.asyncio
    async def test_trajectory_recorded_on_findings(self) -> None:
        traj = MagicMock()
        finding = _make_finding("E1", Severity.WARN, "test")
        mw = _build_middleware(code_findings=[finding], trajectory=traj)
        ctx = _make_ctx()

        resp = MockResponse(content="thinking...")
        await mw.on_llm_end(resp, ctx)

        traj.record_sync.assert_called()
        call_kwargs = traj.record_sync.call_args
        assert call_kwargs.kwargs["event_type"] == "sanitizer"
        assert call_kwargs.kwargs["agent_path"] == ["orchestrator"]
        assert call_kwargs.kwargs["data"]["trigger"] == "every_round"
        assert len(call_kwargs.kwargs["data"]["findings"]) == 1

    @pytest.mark.asyncio
    async def test_no_trajectory_when_none(self) -> None:
        """No crash when trajectory is None."""
        finding = _make_finding("E1", Severity.WARN, "test")
        mw = _build_middleware(code_findings=[finding], trajectory=None)
        ctx = _make_ctx()

        resp = MockResponse(content="thinking...")
        await mw.on_llm_end(resp, ctx)
        # Should not raise


class TestAsyncCriticDrain:
    """Test async critic result collection."""

    @pytest.mark.asyncio
    async def test_async_results_collected_on_llm_start(self) -> None:
        async_finding = _make_finding("P2", Severity.INFO, "dispatch incomplete")
        mw = _build_middleware(
            with_critic=True,
            critic_async_results=[async_finding],
        )
        ctx = _make_ctx()
        messages: list[Any] = [{"role": "system", "content": "prompt"}]

        result = await mw.on_llm_start(messages, ctx)

        assert len(result) == 2
        assert "P2" in result[-1]["content"]

    @pytest.mark.asyncio
    async def test_no_async_results_no_injection(self) -> None:
        mw = _build_middleware(with_critic=True, critic_async_results=[])
        ctx = _make_ctx()
        messages: list[Any] = [{"role": "system", "content": "prompt"}]

        result = await mw.on_llm_start(messages, ctx)
        assert len(result) == 1
