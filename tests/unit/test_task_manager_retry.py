"""Tests for TaskManager single-execution semantics.

Task-level retry was removed — transient failures (network, rate-limit) are
handled at the LLM request level by ChatOpenAI's built-in retry.  TaskManager
now runs the subgraph exactly once and marks the task COMPLETED or FAILED.

These tests verify:
- Successful execution marks the task COMPLETED
- Any exception marks the task FAILED with the error summary
- CancelledError is propagated (not swallowed)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agentm.core.task_manager import TaskManager
from agentm.models.enums import AgentRunStatus


def _make_subgraph(effect) -> MagicMock:
    """Build a mock subgraph whose astream yields *effect*.

    *effect* is either:
    - A list of tuples — yielded as events
    - An Exception — raised during iteration
    """

    async def _astream(*_args, **_kwargs):
        if isinstance(effect, Exception):
            raise effect
        for item in effect:
            yield item

    mock = MagicMock()
    mock.astream = _astream
    return mock


class TestExecuteSuccess:
    """Subgraph completes normally -> task COMPLETED with structured result."""

    @pytest.mark.asyncio
    async def test_success(self):
        events = [
            ("ns", "updates", {"step": 1}),
            ("ns", "updates", {"structured_response": {"findings": "ok"}}),
        ]
        subgraph = _make_subgraph(events)

        tm = TaskManager()
        task_id = await tm.submit("db", "check connections", subgraph=subgraph)
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None
        await managed.asyncio_task

        assert managed.status == AgentRunStatus.COMPLETED
        assert managed.result is not None


class TestExecuteFailure:
    """Subgraph raises an exception -> task FAILED immediately (no retry)."""

    @pytest.mark.asyncio
    async def test_exception_marks_failed(self):
        subgraph = _make_subgraph(RuntimeError("recursion limit reached"))

        tm = TaskManager()
        task_id = await tm.submit("db", "check connections", subgraph=subgraph)
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None
        await managed.asyncio_task

        assert managed.status == AgentRunStatus.FAILED
        assert "recursion limit reached" in managed.error_summary


class TestCancelledNotSwallowed:
    """CancelledError propagates immediately."""

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        async def _blocking_astream(*_args, **_kwargs):
            await asyncio.sleep(100)
            yield ("ns", "updates", {"step": 1})  # pragma: no cover

        mock = MagicMock()
        mock.astream = _blocking_astream

        tm = TaskManager()
        task_id = await tm.submit("db", "check connections", subgraph=mock)
        managed = tm.get_task(task_id)
        assert managed.asyncio_task is not None

        await asyncio.sleep(0.01)
        managed.asyncio_task.cancel()

        try:
            await managed.asyncio_task
        except asyncio.CancelledError:
            pass

        assert managed.asyncio_task.cancelled()
