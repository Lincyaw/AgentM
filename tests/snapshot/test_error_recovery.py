"""P3, P4, P5: Error recovery tests — retry, inject rejection, abort.

Bug prevented:
- P3: API error silently swallowed → Orchestrator waits forever
- P4: inject for completed task silently queued → mental model diverges
- P5: abort returns success but task continues → resource leak
"""

from __future__ import annotations

import asyncio

import pytest

from agentm.core.task_manager import TaskManager
from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus


class TestInjectInstructionRejected:
    """P4: inject for COMPLETED task raises error."""

    @pytest.mark.asyncio
    async def test_inject_for_completed_task_raises(
        self, task_manager_with_completed_task: TaskManager
    ) -> None:
        """inject should raise ValueError for a non-RUNNING task."""
        with pytest.raises(ValueError, match="not running"):
            await task_manager_with_completed_task.inject(
                "task-completed-001", "new instruction"
            )

    @pytest.mark.asyncio
    async def test_inject_instruction_not_queued_for_completed(
        self, task_manager_with_completed_task: TaskManager
    ) -> None:
        """After rejection, instruction must NOT be in pending list."""
        try:
            await task_manager_with_completed_task.inject(
                "task-completed-001", "new instruction"
            )
        except ValueError:
            pass

        task = task_manager_with_completed_task.get_task("task-completed-001")
        assert len(task.pending_instructions) == 0


class TestAbortTask:
    """P5: abort on running task sets FAILED with reason."""

    @pytest.fixture(autouse=True)
    def _setup(self, task_manager_with_running_task: TaskManager) -> None:
        self.task_manager = task_manager_with_running_task

    @pytest.mark.asyncio
    async def test_abort_sets_failed_status(self) -> None:
        """TaskManager.abort should set task status to FAILED."""
        await self.task_manager.abort("task-running-001", "timeout")

        task = self.task_manager.get_task("task-running-001")
        assert task.status == AgentRunStatus.FAILED

    @pytest.mark.asyncio
    async def test_abort_records_reason(self) -> None:
        """TaskManager.abort should record the reason in error_summary."""
        await self.task_manager.abort("task-running-001", "timeout")

        task = self.task_manager.get_task("task-running-001")
        assert "timeout" in task.error_summary

    @pytest.mark.asyncio
    async def test_abort_cancels_asyncio_task(self) -> None:
        """TaskManager.abort should cancel the asyncio.Task if present."""
        async def _long_running():
            await asyncio.sleep(3600)

        asyncio_task = asyncio.ensure_future(_long_running())
        self.task_manager.get_task("task-running-001").asyncio_task = asyncio_task

        await self.task_manager.abort("task-running-001", "timeout")

        assert asyncio_task.cancelling() > 0

    @pytest.mark.asyncio
    async def test_abort_on_failed_task_raises(
        self, task_manager_with_failed_task: TaskManager
    ) -> None:
        """TaskManager.abort should reject non-RUNNING tasks."""
        with pytest.raises(ValueError, match="not running"):
            await task_manager_with_failed_task.abort("task-failed-001", "cleanup")
