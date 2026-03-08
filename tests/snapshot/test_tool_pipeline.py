"""P1, P2: Tool pipeline tests — dispatch_agent and check_tasks data flow.

Bug prevented:
- P1: dispatch_agent fires but TaskManager not updated → Orchestrator loses track
- P2: check_tasks returns data but doesn't persist in Notebook → data lost on compression
"""

from __future__ import annotations

import json

import pytest

import agentm.tools.orchestrator as orch_tools
from agentm.core.task_manager import TaskManager
from agentm.models.enums import AgentRunStatus


class TestDispatchAgentPipeline:
    """P1: dispatch_agent creates a managed task and returns task_id."""

    @pytest.fixture(autouse=True)
    def _bind_task_manager(self, task_manager: TaskManager) -> None:
        self._orig = orch_tools._task_manager
        orch_tools._task_manager = task_manager
        self.task_manager = task_manager
        yield
        orch_tools._task_manager = self._orig

    @pytest.mark.asyncio
    async def test_dispatch_creates_managed_task(self) -> None:
        """dispatch_agent should create a RUNNING task in TaskManager."""
        cmd = await orch_tools.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
            tool_call_id="call-001",
        )

        # TaskManager should have exactly one task
        status = await self.task_manager.get_all_status()
        all_tasks = status["running"] + status["completed"] + status["failed"]
        assert len(all_tasks) == 1
        assert all_tasks[0]["agent_id"] == "db"

    @pytest.mark.asyncio
    async def test_dispatch_returns_command_with_task_id(self) -> None:
        """Returned Command should contain ToolMessage with task_id."""
        cmd = await orch_tools.dispatch_agent(
            agent_id="db",
            task="check connections",
            tool_call_id="call-002",
        )

        messages = cmd.update["messages"]
        assert len(messages) == 1
        content = json.loads(messages[0].content)
        assert "task_id" in content
        assert content["agent_id"] == "db"
        assert content["status"] == "running"

    @pytest.mark.asyncio
    async def test_dispatch_with_hypothesis_id(self) -> None:
        """dispatch_agent should pass hypothesis_id to TaskManager."""
        await orch_tools.dispatch_agent(
            agent_id="db",
            task="verify H1",
            task_type="verify",
            hypothesis_id="H1",
            tool_call_id="call-003",
        )

        status = await self.task_manager.get_all_status()
        running = status["running"]
        assert len(running) == 1
        assert running[0]["hypothesis_id"] == "H1"


class TestCheckTasksPipeline:
    """P2: check_tasks returns status and results via Command."""

    @pytest.fixture(autouse=True)
    def _bind_task_manager(self, task_manager_with_completed_task: TaskManager) -> None:
        self._orig = orch_tools._task_manager
        orch_tools._task_manager = task_manager_with_completed_task
        self.task_manager = task_manager_with_completed_task
        yield
        orch_tools._task_manager = self._orig

    @pytest.mark.asyncio
    async def test_check_tasks_returns_completed_results(self) -> None:
        """check_tasks should include completed task results in ToolMessage."""
        cmd = await orch_tools.check_tasks(
            wait_seconds=0,
            tool_call_id="call-010",
        )

        messages = cmd.update["messages"]
        assert len(messages) == 1
        content = json.loads(messages[0].content)
        assert len(content["completed"]) == 1
        assert content["completed"][0]["agent_id"] == "db"
        assert content["completed"][0]["result"] == {
            "connections": "200/200",
            "wait_queue": 47,
        }

    @pytest.mark.asyncio
    async def test_check_tasks_returns_failed_results(
        self, task_manager_with_failed_task: TaskManager
    ) -> None:
        """check_tasks should include failed task error summary."""
        orch_tools._task_manager = task_manager_with_failed_task

        cmd = await orch_tools.check_tasks(
            wait_seconds=0,
            tool_call_id="call-011",
        )

        content = json.loads(cmd.update["messages"][0].content)
        assert len(content["failed"]) == 1
        assert "timeout" in content["failed"][0]["error_summary"].lower()
