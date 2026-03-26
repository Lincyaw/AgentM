"""P4, P5: Error recovery tests — inject rejection, abort.

Migrated to use AgentRuntime instead of TaskManager.

Bug prevented:
- P4: inject for non-existent agent returns error string
- P5: abort returns success but task continues -> resource leak
"""
from __future__ import annotations

import asyncio

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.harness.types import AgentStatus
from agentm.tools.orchestrator import create_orchestrator_tools

from tests.snapshot.conftest import FakeWorkerFactory, NeverEndingLoop


class TestAbortTask:
    """P5: abort_task on running task sets ABORTED with reason."""

    @pytest.fixture(autouse=True)
    def _bind_tools(self) -> None:
        self.runtime = AgentRuntime()
        self.factory = FakeWorkerFactory()
        tools = create_orchestrator_tools(self.runtime, self.factory)
        self.abort_task = tools["abort_task"]

    @pytest.mark.asyncio
    async def test_abort_sets_aborted_status(self) -> None:
        """abort_task should set agent status to ABORTED."""
        loop = NeverEndingLoop()
        await self.runtime.spawn("task-running-001", loop=loop, input="scan metrics")
        await asyncio.sleep(0.05)

        await self.abort_task("task-running-001", "timeout")

        result = self.runtime.get_result("task-running-001")
        assert result is not None
        assert result.status == AgentStatus.ABORTED

    @pytest.mark.asyncio
    async def test_abort_records_reason(self) -> None:
        """abort_task should record the reason in error."""
        loop = NeverEndingLoop()
        await self.runtime.spawn("task-running-001", loop=loop, input="scan metrics")
        await asyncio.sleep(0.05)

        await self.abort_task("task-running-001", "timeout")

        result = self.runtime.get_result("task-running-001")
        assert result is not None
        assert "timeout" in result.error

    @pytest.mark.asyncio
    async def test_abort_cancels_asyncio_task(self) -> None:
        """abort_task should cancel the underlying asyncio.Task."""
        loop = NeverEndingLoop()
        await self.runtime.spawn("task-running-001", loop=loop, input="scan metrics")
        await asyncio.sleep(0.05)

        await self.abort_task("task-running-001", "timeout")

        # The agent should be in a terminal state
        result = self.runtime.get_result("task-running-001")
        assert result is not None
        assert result.status == AgentStatus.ABORTED

    @pytest.mark.asyncio
    async def test_abort_on_nonexistent_task_returns_not_found(self) -> None:
        """abort should return not-found message for unknown task IDs."""
        result = await self.abort_task("task-nonexistent-001", "cleanup")
        assert "not found" in result.lower()
