"""Tests for TaskManager exponential backoff retry.

Ref: designs/system-design-overview.md -- Layer 2: Retry (API + Task Level)
Ref: designs/orchestrator.md -- TaskManager

TaskManager wraps subgraph execution with retry + exponential backoff.
These tests verify:
- Success after transient failure triggers retry and ultimately COMPLETED status
- Exhausted retries mark the task as FAILED with the last error
- Backoff delays follow the exponential schedule from RetryConfig
- CancelledError is not retried (propagated immediately)

Bug prevented: subgraph fails once with a transient error -> task marked FAILED
immediately instead of retrying -> Orchestrator sees permanent failure for a
recoverable issue.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentm.config.schema import RetryConfig
from agentm.core.task_manager import TaskManager
from agentm.models.enums import AgentRunStatus


def _make_subgraph(side_effects: list) -> MagicMock:
    """Build a mock subgraph whose astream yields from *side_effects*.

    Each entry can be:
    - A list of tuples — yielded as events
    - An Exception — raised during iteration
    """
    call_count = 0

    async def _astream(*_args, **_kwargs):
        nonlocal call_count
        effect = side_effects[call_count]
        call_count += 1
        if isinstance(effect, Exception):
            raise effect
        for item in effect:
            yield item

    mock = MagicMock()
    mock.astream = _astream
    return mock


class TestRetrySuccessAfterFailure:
    """Task succeeds on second attempt after a transient error.

    Bug: single failure -> task FAILED -> Orchestrator gives up.
    Expected: retry -> task COMPLETED.
    """

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        events = [("ns", "updates", {"step": 1})]
        subgraph = _make_subgraph([
            RuntimeError("transient"),
            events,
        ])
        retry_cfg = RetryConfig(max_attempts=3, initial_interval=0.01, backoff_factor=2.0)

        tm = TaskManager()
        task_id = await tm.submit(
            "db", "check connections",
            subgraph=subgraph, retry_config=retry_cfg,
        )
        # Wait for the async task to complete
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None
        await managed.asyncio_task

        assert managed.status == AgentRunStatus.COMPLETED
        assert managed.result is not None


class TestRetryExhausted:
    """All retry attempts fail -> task is FAILED with last error.

    Bug: retry exhausted but status stays RUNNING forever.
    """

    @pytest.mark.asyncio
    async def test_all_attempts_fail(self):
        subgraph = _make_subgraph([
            RuntimeError("fail-1"),
            RuntimeError("fail-2"),
            RuntimeError("fail-3"),
        ])
        retry_cfg = RetryConfig(max_attempts=3, initial_interval=0.01, backoff_factor=2.0)

        tm = TaskManager()
        task_id = await tm.submit(
            "db", "check connections",
            subgraph=subgraph, retry_config=retry_cfg,
        )
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None
        await managed.asyncio_task

        assert managed.status == AgentRunStatus.FAILED
        assert managed.error_summary == "fail-3"

    @pytest.mark.asyncio
    async def test_single_attempt_no_retry(self):
        """With max_attempts=1, failure is immediate."""
        subgraph = _make_subgraph([RuntimeError("only-try")])
        retry_cfg = RetryConfig(max_attempts=1, initial_interval=0.01)

        tm = TaskManager()
        task_id = await tm.submit(
            "db", "check connections",
            subgraph=subgraph, retry_config=retry_cfg,
        )
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None
        await managed.asyncio_task

        assert managed.status == AgentRunStatus.FAILED
        assert managed.error_summary == "only-try"


class TestRetryCancelledNotRetried:
    """CancelledError propagates immediately, never retried.

    Bug: CancelledError caught by retry loop -> agent keeps running
    even after abort_task cancels it.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_not_retried(self):
        """Verify that abort (task.cancel()) is not retried."""
        # Subgraph that blocks until cancelled
        async def _blocking_astream(*_args, **_kwargs):
            await asyncio.sleep(100)
            yield ("ns", "updates", {"step": 1})  # pragma: no cover

        mock = MagicMock()
        mock.astream = _blocking_astream

        retry_cfg = RetryConfig(max_attempts=3, initial_interval=0.01)

        tm = TaskManager()
        task_id = await tm.submit(
            "db", "check connections",
            subgraph=mock, retry_config=retry_cfg,
        )
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None

        # Give the task a moment to start, then cancel
        await asyncio.sleep(0.01)
        managed.asyncio_task.cancel()

        try:
            await managed.asyncio_task
        except asyncio.CancelledError:
            pass

        # The task was cancelled, not retried to FAILED
        assert managed.asyncio_task.cancelled()
        assert managed.status != AgentRunStatus.FAILED


class TestRetryBackoffSchedule:
    """Verify that delays follow exponential backoff.

    With initial_interval=1.0 and backoff_factor=2.0:
    - After attempt 1 failure: sleep 1.0s
    - After attempt 2 failure: sleep 2.0s
    """

    @pytest.mark.asyncio
    async def test_backoff_delays(self, monkeypatch):
        recorded_delays: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            recorded_delays.append(delay)
            # Don't actually sleep in tests

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)

        subgraph = _make_subgraph([
            RuntimeError("fail-1"),
            RuntimeError("fail-2"),
            RuntimeError("fail-3"),
        ])
        retry_cfg = RetryConfig(
            max_attempts=3, initial_interval=1.0, backoff_factor=2.0
        )

        tm = TaskManager()
        task_id = await tm.submit(
            "db", "check connections",
            subgraph=subgraph, retry_config=retry_cfg,
        )
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None
        await managed.asyncio_task

        # 3 attempts, 2 sleeps (after attempt 1 and 2)
        assert recorded_delays == [1.0, 2.0]
