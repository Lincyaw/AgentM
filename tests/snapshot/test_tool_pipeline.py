"""P1, P2: Tool pipeline tests — dispatch_agent and check_tasks data flow.

Bug prevented:
- P1: dispatch_agent fires but TaskManager not updated → Orchestrator loses track
- P2: check_tasks returns data but doesn't persist in Notebook → data lost on compression
"""

from __future__ import annotations

import json

import pytest

from agentm.core.task_manager import TaskManager
from agentm.models.enums import AgentRunStatus
from agentm.tools.orchestrator import create_orchestrator_tools


class TestDispatchAgentPipeline:
    """P1: dispatch_agent creates a managed task and returns task_id."""

    @pytest.fixture(autouse=True)
    def _bind_tools(self, task_manager: TaskManager) -> None:
        self.task_manager = task_manager
        tools = create_orchestrator_tools(task_manager, agent_pool=None)
        self.dispatch_agent = tools["dispatch_agent"]
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_dispatch_creates_managed_task(self) -> None:
        """dispatch_agent should create a RUNNING task in TaskManager."""
        cmd = await self.dispatch_agent(
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
        cmd = await self.dispatch_agent(
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
        await self.dispatch_agent(
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
    def _bind_tools(self, task_manager_with_completed_task: TaskManager) -> None:
        self.task_manager = task_manager_with_completed_task
        tools = create_orchestrator_tools(task_manager_with_completed_task, agent_pool=None)
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_check_tasks_returns_completed_results(self) -> None:
        """check_tasks should include completed task results in ToolMessage."""
        cmd = await self.check_tasks(
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
        tools = create_orchestrator_tools(task_manager_with_failed_task, agent_pool=None)
        check_tasks = tools["check_tasks"]

        cmd = await check_tasks(
            wait_seconds=0,
            tool_call_id="call-011",
        )

        content = json.loads(cmd.update["messages"][0].content)
        assert len(content["failed"]) == 1
        assert "timeout" in content["failed"][0]["error_summary"].lower()
