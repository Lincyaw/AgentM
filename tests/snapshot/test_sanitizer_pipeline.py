"""End-to-end integration tests for the sanitizer pipeline.

Exercises SanitizerMiddleware with real CodeSanitizer, real stores, and
real InvestigationTracker. CriticSanitizer is either mocked or None —
code-based checks are the primary focus for E2E validation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentm.harness.scenario import TrajectorySlot
from agentm.harness.types import LoopContext
from agentm.scenarios.rca.hypothesis_store import HypothesisStore
from agentm.scenarios.rca.sanitizer.code_sanitizer import CodeSanitizer
from agentm.scenarios.rca.sanitizer.middleware import SanitizerMiddleware
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
        traj_slot: TrajectorySlot | None = None
        if trajectory is not None:
            traj_slot = TrajectorySlot()
            traj_slot.value = trajectory
        return SanitizerMiddleware(
            code_sanitizer=code_sanitizer,
            critic_sanitizer=critic,
            tracker=tracker,
            hypothesis_store=hypothesis_store,
            profile_store=profile_store,
            traj_slot=traj_slot,
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
# Test 2: BLOCK degradation after max_block_retries
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
# Test 3: Budget exhaustion degrades coverage BLOCKs but not process BLOCKs
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
# Test 4: TrajectoryCollector receives sanitizer events
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
