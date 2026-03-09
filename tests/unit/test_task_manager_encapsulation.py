"""Tests for TaskManager public interface and encapsulation.

Bug prevented: external code bypasses validation by directly accessing
private attributes (_trajectory, _broadcast_callback), or gets unhelpful
KeyError when looking up unknown task IDs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentm.core.task_manager import TaskManager
from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus


class TestPublicInterface:
    """TaskManager should expose public methods for dependency wiring."""

    def test_trajectory_property_returns_none_by_default(self):
        tm = TaskManager()
        assert tm.trajectory is None

    def test_set_trajectory_stores_value(self):
        tm = TaskManager()
        mock_trajectory = MagicMock()
        tm.set_trajectory(mock_trajectory)
        assert tm.trajectory is mock_trajectory

    def test_set_broadcast_callback_stores_value(self):
        tm = TaskManager()
        mock_callback = MagicMock()
        tm.set_broadcast_callback(mock_callback)
        assert tm._broadcast_callback is mock_callback


class TestGetTaskError:
    """get_task should raise ValueError (not KeyError) for unknown IDs."""

    def test_raises_valueerror_for_unknown_id(self):
        tm = TaskManager()
        with pytest.raises(ValueError, match="not found"):
            tm.get_task("nonexistent-task-id")

    def test_error_message_includes_known_ids(self):
        tm = TaskManager()
        tm._tasks["task-001"] = ManagedTask(
            task_id="task-001", agent_id="test", instruction="test"
        )
        with pytest.raises(ValueError, match="task-001"):
            tm.get_task("wrong-id")


class TestAbortRecordsTrajectory:
    """abort() should record a task_abort event to trajectory."""

    @pytest.mark.asyncio
    async def test_abort_records_trajectory_event(self):
        tm = TaskManager()
        mock_trajectory = AsyncMock()
        tm.set_trajectory(mock_trajectory)

        managed = ManagedTask(
            task_id="task-001",
            agent_id="worker",
            instruction="test",
            status=AgentRunStatus.RUNNING,
        )
        tm._tasks["task-001"] = managed

        await tm.abort("task-001", "timeout")

        mock_trajectory.record.assert_called_once()
        call_kwargs = mock_trajectory.record.call_args.kwargs
        assert call_kwargs["event_type"] == "task_abort"
        assert call_kwargs["data"]["reason"] == "timeout"
