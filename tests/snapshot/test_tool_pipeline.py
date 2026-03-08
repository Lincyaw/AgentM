"""P1, P2: Tool pipeline tests — spawn_worker and wait_for_workers data flow.

Bug prevented:
- P1: spawn_worker fires but TaskManager not updated → Orchestrator loses track
- P2: wait_for_workers returns data but doesn't persist in Notebook → data lost on compression
"""

from __future__ import annotations

import json

import pytest

import agentm.tools.orchestrator as orch_tools
from agentm.core.task_manager import TaskManager
from agentm.models.enums import AgentRunStatus


class TestSpawnWorkerPipeline:
    """P1: spawn_worker creates a managed task and returns task status JSON."""

    @pytest.fixture(autouse=True)
    def _bind_refs(self, task_manager: TaskManager) -> None:
        self._orig_tm = orch_tools._task_manager
        self._orig_pool = orch_tools._agent_pool
        orch_tools._task_manager = task_manager
        # Provide a minimal mock agent_pool
        orch_tools._agent_pool = _MockAgentPool()
        self.task_manager = task_manager
        yield
        orch_tools._task_manager = self._orig_tm
        orch_tools._agent_pool = self._orig_pool

    @pytest.mark.asyncio
    async def test_spawn_creates_managed_task(self) -> None:
        """spawn_worker should create a task in TaskManager."""
        result = await orch_tools.spawn_worker(
            task_type="scout",
            instructions="check connections",
        )

        # TaskManager should have exactly one task
        status = await self.task_manager.get_all_status()
        all_tasks = status["running"] + status["completed"] + status["failed"]
        assert len(all_tasks) == 1
        assert all_tasks[0]["agent_id"] == "worker-scout"

    @pytest.mark.asyncio
    async def test_spawn_returns_json_with_task_id(self) -> None:
        """Returned JSON should contain task_id, task_type, status."""
        result = await orch_tools.spawn_worker(
            task_type="scout",
            instructions="check connections",
        )

        content = json.loads(result)
        assert "task_id" in content
        assert content["task_type"] == "scout"
        assert content["status"] == "running"

    @pytest.mark.asyncio
    async def test_spawn_rejects_invalid_task_type(self) -> None:
        """spawn_worker should return error JSON for invalid task_type."""
        result = await orch_tools.spawn_worker(
            task_type="invalid",
            instructions="do something",
        )
        content = json.loads(result)
        assert "error" in content


class TestWaitForWorkersPipeline:
    """P2: wait_for_workers returns status and results."""

    @pytest.fixture(autouse=True)
    def _bind_task_manager(self, task_manager_with_completed_task: TaskManager) -> None:
        self._orig = orch_tools._task_manager
        orch_tools._task_manager = task_manager_with_completed_task
        self.task_manager = task_manager_with_completed_task
        yield
        orch_tools._task_manager = self._orig

    @pytest.mark.asyncio
    async def test_wait_returns_completed_results(self) -> None:
        """wait_for_workers should include completed task results."""
        result = await orch_tools.wait_for_workers(timeout_seconds=0)

        content = json.loads(result)
        assert len(content["completed"]) == 1
        assert content["completed"][0]["agent_id"] == "db"
        assert content["completed"][0]["result"] == {
            "connections": "200/200",
            "wait_queue": 47,
        }

    @pytest.mark.asyncio
    async def test_wait_returns_failed_results(
        self, task_manager_with_failed_task: TaskManager
    ) -> None:
        """wait_for_workers should include failed task error summary."""
        orch_tools._task_manager = task_manager_with_failed_task

        result = await orch_tools.wait_for_workers(timeout_seconds=0)

        content = json.loads(result)
        assert len(content["failed"]) == 1
        assert "timeout" in content["failed"][0]["error_summary"].lower()


class _MockAgentPool:
    """Minimal mock that returns None as the subgraph (TaskManager won't run it)."""

    def get_worker(self, task_type: str):
        return None
