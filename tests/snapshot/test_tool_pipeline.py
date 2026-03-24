"""P1, P2: Tool pipeline tests — dispatch_agent and check_tasks data flow.

Bug prevented:
- P1: dispatch_agent fires but TaskManager not updated → Orchestrator loses track
- P2: check_tasks returns data but doesn't persist in Notebook → data lost on compression
"""

from __future__ import annotations

import json

import pytest

from agentm.core.task_manager import TaskManager
from agentm.tools.orchestrator import create_orchestrator_tools


class TestDispatchAgentPipeline:
    """P1: dispatch_agent creates a managed task and returns task_id."""

    @pytest.fixture(autouse=True)
    def _bind_tools(self, task_manager: TaskManager, agent_pool) -> None:
        self.task_manager = task_manager
        tools = create_orchestrator_tools(task_manager, agent_pool=agent_pool)
        self.dispatch_agent = tools["dispatch_agent"]
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_dispatch_creates_managed_task(self) -> None:
        """dispatch_agent should create a task in TaskManager.

        With auto-block (single worker), the task completes before returning.
        """
        await self.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
            tool_call_id="call-001",
        )

        status = await self.task_manager.get_all_status()
        all_tasks = status["running"] + status["completed"] + status["failed"]
        assert len(all_tasks) == 1
        assert all_tasks[0]["agent_id"] == "db"

    @pytest.mark.asyncio
    async def test_dispatch_single_worker_auto_blocks(self) -> None:
        """Single-worker dispatch should auto-block and return completed result.

        Bug prevented: Orchestrator wastes an LLM roundtrip calling check_tasks
        when there's only one worker and nothing else to do.
        """
        cmd = await self.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
            tool_call_id="call-002",
        )

        messages = cmd.update["messages"]
        assert len(messages) == 1
        content = json.loads(messages[0].content)
        assert "task_id" in content
        assert content["agent_id"] == "db"
        assert content["status"] == "completed"
        assert content["result"] is not None

    @pytest.mark.asyncio
    async def test_dispatch_with_metadata(self) -> None:
        """dispatch_agent should pass metadata to TaskManager."""
        await self.dispatch_agent(
            agent_id="db",
            task="verify H1",
            task_type="verify",
            metadata={"hypothesis_id": "H1"},
            tool_call_id="call-003",
        )

        status = await self.task_manager.get_all_status()
        # Auto-block completes the single task
        completed = status["completed"]
        assert len(completed) == 1
        assert completed[0]["metadata"]["hypothesis_id"] == "H1"


class TestCheckTasksPipeline:
    """P2: check_tasks returns status and results via Command."""

    @pytest.fixture(autouse=True)
    def _bind_tools(
        self, task_manager_with_completed_task: TaskManager, agent_pool
    ) -> None:
        self.task_manager = task_manager_with_completed_task
        tools = create_orchestrator_tools(
            task_manager_with_completed_task, agent_pool=agent_pool
        )
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_check_tasks_returns_completed_results(self) -> None:
        """check_tasks should include completed task results in ToolMessage."""
        cmd = await self.check_tasks(
            request="status",
            tool_call_id="call-010",
        )

        messages = cmd.update["messages"]
        assert len(messages) == 1
        content = json.loads(messages[0].content)
        assert len(content["completed"]) == 1
        assert content["completed"][0]["agent_id"] == "db"
        assert content["completed"][0]["result"] == {
            "findings": "connections: 200/200, wait_queue: 47",
        }

    @pytest.mark.asyncio
    async def test_check_tasks_returns_failed_results(
        self, task_manager_with_failed_task: TaskManager, agent_pool
    ) -> None:
        """check_tasks should include failed task error summary."""
        tools = create_orchestrator_tools(
            task_manager_with_failed_task, agent_pool=agent_pool
        )
        check_tasks = tools["check_tasks"]

        cmd = await check_tasks(
            request="status",
            tool_call_id="call-011",
        )

        content = json.loads(cmd.update["messages"][0].content)
        assert len(content["failed"]) == 1
        assert "timeout" in content["failed"][0]["error_summary"].lower()


class TestCheckTasksIncrementalReporting:
    """P2b: check_tasks uses incremental reporting to save context window.

    Bug prevented: completed/failed results returned on every check_tasks call
    → same data repeated in every ToolMessage → context window wasted.

    Design intent: completed/failed tasks are reported ONCE with full result,
    then marked as reported. Subsequent check_tasks calls omit already-reported
    terminal tasks. Running tasks always return a progress summary.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, task_manager_with_completed_task: TaskManager, agent_pool) -> None:
        self.task_manager = task_manager_with_completed_task
        tools = create_orchestrator_tools(
            task_manager_with_completed_task, agent_pool=agent_pool
        )
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_completed_task_reported_once(self) -> None:
        """First check_tasks returns completed result; second call omits it.

        Bug: without incremental reporting, a completed task's full result
        (potentially large JSON) appears in every check_tasks response,
        consuming context tokens repeatedly for zero new information.
        """
        # First call — result should be present
        cmd1 = await self.check_tasks(
            request="status",
            tool_call_id="call-020",
        )
        content1 = json.loads(cmd1.update["messages"][0].content)
        assert len(content1["completed"]) == 1
        assert content1["completed"][0]["result"] is not None

        # Second call — already-reported task should NOT reappear
        cmd2 = await self.check_tasks(
            request="status",
            tool_call_id="call-021",
        )
        content2 = json.loads(cmd2.update["messages"][0].content)
        assert len(content2["completed"]) == 0

    @pytest.mark.asyncio
    async def test_failed_task_reported_once(
        self, task_manager_with_failed_task: TaskManager, agent_pool
    ) -> None:
        """Failed tasks follow the same incremental rule as completed tasks."""
        tools = create_orchestrator_tools(
            task_manager_with_failed_task, agent_pool=agent_pool
        )
        check_tasks = tools["check_tasks"]

        # First call — failure details present
        cmd1 = await check_tasks(
            request="status",
            tool_call_id="call-022",
        )
        content1 = json.loads(cmd1.update["messages"][0].content)
        assert len(content1["failed"]) == 1

        # Second call — already-reported failure omitted
        cmd2 = await check_tasks(
            request="status",
            tool_call_id="call-023",
        )
        content2 = json.loads(cmd2.update["messages"][0].content)
        assert len(content2["failed"]) == 0
