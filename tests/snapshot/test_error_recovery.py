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
from agentm.models.enums import AgentRunStatus
from agentm.tools.orchestrator import create_orchestrator_tools


class TestInjectInstructionRejected:
    """P4: inject_instruction for COMPLETED task raises error."""

    @pytest.mark.asyncio
    async def test_inject_for_completed_task_raises(
        self, task_manager_with_completed_task: TaskManager
    ) -> None:
        """inject_instruction should raise ValueError for a non-RUNNING task."""
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
    """P5: abort_task on running task sets FAILED with reason."""

    @pytest.fixture(autouse=True)
    def _bind_tools(self, task_manager_with_running_task: TaskManager) -> None:
        self.task_manager = task_manager_with_running_task
        tools = create_orchestrator_tools(
            task_manager_with_running_task, agent_pool=None
        )
        self.abort_task = tools["abort_task"]

    @pytest.mark.asyncio
    async def test_abort_sets_failed_status(self) -> None:
        """abort_task should set task status to FAILED."""
        await self.abort_task("task-running-001", "timeout")

        task = self.task_manager.get_task("task-running-001")
        assert task.status == AgentRunStatus.FAILED

    @pytest.mark.asyncio
    async def test_abort_records_reason(self) -> None:
        """abort_task should record the reason in error_summary."""
        await self.abort_task("task-running-001", "timeout")

        task = self.task_manager.get_task("task-running-001")
        assert "timeout" in task.error_summary

    @pytest.mark.asyncio
    async def test_abort_cancels_asyncio_task(self) -> None:
        """abort_task should cancel the asyncio.Task if present."""

        async def _long_running():
            await asyncio.sleep(3600)

        asyncio_task = asyncio.ensure_future(_long_running())
        self.task_manager.get_task("task-running-001").asyncio_task = asyncio_task

        await self.abort_task("task-running-001", "timeout")

        assert asyncio_task.cancelling() > 0

    @pytest.mark.asyncio
    async def test_abort_on_failed_task_raises(
        self, task_manager_with_failed_task: TaskManager
    ) -> None:
        """abort_task via TaskManager.abort should reject non-RUNNING tasks."""
        with pytest.raises(ValueError, match="not running"):
            await task_manager_with_failed_task.abort("task-failed-001", "cleanup")
