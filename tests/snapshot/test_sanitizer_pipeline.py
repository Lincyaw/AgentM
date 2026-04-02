"""End-to-end integration tests for the sanitizer pipeline.

Exercises SanitizerMiddleware with real CodeSanitizer, real stores, and
real InvestigationTracker. CriticSanitizer is either mocked or None —
code-based checks are the primary focus for E2E validation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentm.harness.types import LoopContext
from agentm.scenarios.rca.hypothesis_store import HypothesisStore
from agentm.scenarios.rca.sanitizer.code_sanitizer import CodeSanitizer
from agentm.scenarios.rca.sanitizer.middleware import SanitizerMiddleware
from agentm.scenarios.rca.sanitizer.models import Severity
from agentm.scenarios.rca.sanitizer.tracker import InvestigationTracker
from agentm.scenarios.rca.service_profile import ServiceProfileStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """Mutable response object for on_llm_end."""

    def __init__(self, content: str = "", tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


def make_ctx(
    step: int = 0,
    max_steps: int = 60,
    tool_call_count: int = 0,
) -> LoopContext:
    return LoopContext(
        agent_id="orch",
        step=step,
        max_steps=max_steps,
        tool_call_count=tool_call_count,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hypothesis_store() -> HypothesisStore:
    return HypothesisStore()


@pytest.fixture
def profile_store() -> ServiceProfileStore:
    return ServiceProfileStore()


@pytest.fixture
def tracker() -> InvestigationTracker:
    return InvestigationTracker()


@pytest.fixture
def code_sanitizer() -> CodeSanitizer:
    return CodeSanitizer()


@pytest.fixture
def make_middleware(
    hypothesis_store: HypothesisStore,
    profile_store: ServiceProfileStore,
    tracker: InvestigationTracker,
    code_sanitizer: CodeSanitizer,
):
    """Factory that creates SanitizerMiddleware with optional overrides."""

    def _make(
        critic=None,
        periodic_interval: int = 5,
        max_block_retries: int = 3,
        tool_call_budget: int | None = None,
        trajectory=None,
    ) -> SanitizerMiddleware:
        return SanitizerMiddleware(
            code_sanitizer=code_sanitizer,
            critic_sanitizer=critic,
            tracker=tracker,
            hypothesis_store=hypothesis_store,
            profile_store=profile_store,
            trajectory=trajectory,
            periodic_interval=periodic_interval,
            max_block_retries=max_block_retries,
            tool_call_budget=tool_call_budget,
        )

    return _make


# ---------------------------------------------------------------------------
# Test 1: Finalize blocked by C1, then resolved
# ---------------------------------------------------------------------------


class TestFinalizeBlockedThenResolved:
    """C1 blocks finalize for unverified hypothesis; resolving clears the block."""

    @pytest.mark.asyncio
    async def test_finalize_blocked_then_resolved(
        self,
        hypothesis_store: HypothesisStore,
        tracker: InvestigationTracker,
        make_middleware,
    ) -> None:
        # Set up: confirmed hypothesis H1, no verify task
        hypothesis_store.update(id="H1", description="test hypothesis", status="confirmed")

        mw = make_middleware()
        ctx = make_ctx()

        # 1. Call on_llm_end with finalize — should be blocked
        resp = MockResponse(content="<decision>finalize</decision>")
        result = await mw.on_llm_end(resp, ctx)

        # The finalize tag should be stripped
        assert "<decision>finalize</decision>" not in result.content  # type: ignore[union-attr]

        # Block message should be stored separately
        assert mw._pending_block_message is not None
        assert "C1" in mw._pending_block_message

        # 2. on_llm_start should inject the blocked message
        messages: list[dict[str, Any]] = [{"role": "assistant", "content": "thinking"}]
        injected = await mw.on_llm_start(messages, ctx)
        assert len(injected) > len(messages)
        last_msg = injected[-1]
        assert "finalize_blocked" in last_msg["content"].lower()  # type: ignore[union-attr]

        # 3. Resolve: record a verify task completion for H1
        tracker.record(
            round=2,
            event_type="task_complete",
            data={"task_type": "verify", "hypothesis_id": "H1", "verdict": "SUPPORTED"},
        )

        # 4. Call on_llm_end again with finalize — should pass through
        resp2 = MockResponse(content="<decision>finalize</decision>")
        result2 = await mw.on_llm_end(resp2, ctx)

        # Finalize tag should remain (no BLOCK findings)
        assert "<decision>finalize</decision>" in result2.content  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 2: Drift detection (J2)
# ---------------------------------------------------------------------------


class TestDriftDetection:
    """J2 fires when last N dispatches target same service and hypothesis."""

    @pytest.mark.asyncio
    async def test_j2_drift_warning(
        self,
        tracker: InvestigationTracker,
        make_middleware,
    ) -> None:
        # Record 3 consecutive dispatches targeting same service and hypothesis
        for i in range(3):
            tracker.record(
                round=i + 1,
                event_type="dispatch",
                data={"target_services": ["ts-foo"], "hypothesis_id": "H1"},
            )

        mw = make_middleware()
        ctx = make_ctx()

        # Call on_llm_end (no finalize, just triggers every_round checks)
        resp = MockResponse(content="some analysis")
        await mw.on_llm_end(resp, ctx)

        # _pending_findings should have J2 WARN
        j2_findings = [f for f in mw._pending_findings if f.code == "J2"]
        assert len(j2_findings) == 1
        assert j2_findings[0].severity == Severity.WARN

        # on_llm_start should inject the drift warning
        messages: list[dict[str, Any]] = [{"role": "assistant", "content": "ok"}]
        injected = await mw.on_llm_start(messages, ctx)
        assert len(injected) > len(messages)
        injected_content = injected[-1]["content"]
        assert "J2" in injected_content  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Test 3: BLOCK degradation after max_block_retries
# ---------------------------------------------------------------------------


class TestBlockDegradation:
    """BLOCK findings degrade to WARN after max_block_retries attempts."""

    @pytest.mark.asyncio
    async def test_block_degrades_after_retries(
        self,
        hypothesis_store: HypothesisStore,
        make_middleware,
    ) -> None:
        # Set up C1 condition: confirmed hypothesis, no verify
        hypothesis_store.update(id="H1", description="test", status="confirmed")

        mw = make_middleware(max_block_retries=3)
        ctx = make_ctx()

        # Attempts 1-3: blocked
        for _ in range(3):
            resp = MockResponse(content="<decision>finalize</decision>")
            result = await mw.on_llm_end(resp, ctx)
            assert "<decision>finalize</decision>" not in result.content  # type: ignore[union-attr]
            # Drain pending findings so they don't accumulate
            await mw.on_llm_start([], ctx)

        # Attempt 4: degraded — finalize NOT stripped
        resp4 = MockResponse(content="<decision>finalize</decision>")
        result4 = await mw.on_llm_end(resp4, ctx)
        assert "<decision>finalize</decision>" in result4.content  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 4: Budget exhaustion degrades coverage BLOCKs but not process BLOCKs
# ---------------------------------------------------------------------------


class TestBudgetDegradation:
    """Budget exhaustion degrades E-code/J-code BLOCKs to WARN, keeps C-code BLOCKs."""

    @pytest.mark.asyncio
    async def test_budget_degrades_coverage_not_process(
        self,
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        make_middleware,
    ) -> None:
        # Set up E1 condition: anomalous service with unchecked upstream
        profile_store.update(
            service_name="ts-anomalous",
            is_anomalous=True,
            upstream_services=["ts-upstream"],
        )
        # ts-upstream has no profile → triggers E1

        # Set up C1 condition: confirmed hypothesis without verify
        hypothesis_store.update(id="H1", description="test", status="confirmed")

        # Create middleware with budget exhausted
        mw = make_middleware(tool_call_budget=20)
        ctx = make_ctx(tool_call_count=20)  # budget exhausted

        resp = MockResponse(content="<decision>finalize</decision>")
        result = await mw.on_llm_end(resp, ctx)

        # Collect all findings (pending + the ones that caused blocking/passing)
        # E1 should be WARN (budget degraded from its default WARN — stays WARN)
        # C1 should be BLOCK (not degraded)
        # The finalize should still be blocked because C1 is still BLOCK
        assert "<decision>finalize</decision>" not in result.content  # type: ignore[union-attr]

        # Block message should contain C1 (the process check that wasn't degraded)
        assert mw._pending_block_message is not None
        assert "C1" in mw._pending_block_message

        # The finalize being blocked proves C1 retained its BLOCK severity
        # even though budget is exhausted (C-codes are process checks).


# ---------------------------------------------------------------------------
# Test 5: TrajectoryCollector receives sanitizer events
# ---------------------------------------------------------------------------


class TestTrajectoryRecording:
    """Trajectory's record_sync is called with event_type='sanitizer' on findings."""

    @pytest.mark.asyncio
    async def test_trajectory_receives_sanitizer_events(
        self,
        hypothesis_store: HypothesisStore,
        make_middleware,
    ) -> None:
        # Create a mock trajectory
        mock_trajectory = MagicMock()
        mock_trajectory.record_sync = MagicMock(return_value=1)

        # Set up C1 condition
        hypothesis_store.update(id="H1", description="test", status="confirmed")

        mw = make_middleware(trajectory=mock_trajectory)
        ctx = make_ctx()

        # Trigger a finalize with C1 block
        resp = MockResponse(content="<decision>finalize</decision>")
        await mw.on_llm_end(resp, ctx)

        # trajectory.record_sync should have been called with event_type="sanitizer"
        assert mock_trajectory.record_sync.called
        sanitizer_calls = [
            c
            for c in mock_trajectory.record_sync.call_args_list
            if c.kwargs.get("event_type") == "sanitizer"
            or (c.args and c.args[0] == "sanitizer")
        ]
        assert len(sanitizer_calls) >= 1

        # Verify the call signature includes expected keys
        first_call = sanitizer_calls[0]
        # record_sync is called as record_sync(event_type=..., agent_path=..., data=...)
        if first_call.kwargs:
            assert first_call.kwargs.get("event_type") == "sanitizer"
            data = first_call.kwargs.get("data", {})
        else:
            assert first_call.args[0] == "sanitizer"
            data = first_call.args[2] if len(first_call.args) > 2 else {}
        assert "findings" in data
        assert "trigger" in data


# ---------------------------------------------------------------------------
# Test 6: Periodic coverage check fires at interval
# ---------------------------------------------------------------------------


class TestPeriodicChecks:
    """E1 fires only on rounds that are multiples of periodic_interval."""

    @pytest.mark.asyncio
    async def test_periodic_e1_fires_at_interval(
        self,
        profile_store: ServiceProfileStore,
        make_middleware,
    ) -> None:
        # Set up E1 condition: anomalous service with unchecked upstream
        profile_store.update(
            service_name="ts-anomalous",
            is_anomalous=True,
            upstream_services=["ts-upstream"],
        )

        mw = make_middleware(periodic_interval=3)

        def collect_e1_findings() -> list:
            """Extract E1 findings from _pending_findings and drain."""
            return [f for f in mw._pending_findings if f.code == "E1"]

        # Round = ctx.step + 1. Periodic fires when round % interval == 0.
        # step=0→round=1, step=1→round=2, step=2→round=3 (fires), etc.

        # Rounds 1, 2 (step 0, 1): no E1
        for step in range(2):
            ctx = make_ctx(step=step)
            resp = MockResponse(content="thinking")
            await mw.on_llm_end(resp, ctx)
            assert len(collect_e1_findings()) == 0, "E1 should not fire on non-periodic rounds"
            await mw.on_llm_start([], ctx)

        # Round 3 (step 2): E1 should fire
        ctx = make_ctx(step=2)
        resp = MockResponse(content="thinking")
        await mw.on_llm_end(resp, ctx)
        assert len(collect_e1_findings()) >= 1, "E1 should fire on periodic round 3"
        await mw.on_llm_start([], ctx)

        # Rounds 4, 5 (step 3, 4): no E1
        for step in range(3, 5):
            ctx = make_ctx(step=step)
            resp = MockResponse(content="thinking")
            await mw.on_llm_end(resp, ctx)
            assert len(collect_e1_findings()) == 0, "E1 should not fire on non-periodic rounds"
            await mw.on_llm_start([], ctx)

        # Round 6 (step 5): E1 fires again
        ctx = make_ctx(step=5)
        resp = MockResponse(content="thinking")
        await mw.on_llm_end(resp, ctx)
        assert len(collect_e1_findings()) >= 1, "E1 should fire on periodic round 6"
