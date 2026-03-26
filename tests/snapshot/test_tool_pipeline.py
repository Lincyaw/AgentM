"""P1, P2: Tool pipeline tests — dispatch_agent and check_tasks data flow.

Migrated to use AgentRuntime + WorkerLoopFactory instead of TaskManager.
Tools now return plain strings (no LangGraph Command objects).

Bug prevented:
- P1: dispatch_agent fires but runtime not updated -> Orchestrator loses track
- P2: check_tasks returns data in wrong format -> orchestrator misinterprets results
"""
from __future__ import annotations

import json

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.tools.orchestrator import create_orchestrator_tools

from tests.snapshot.conftest import FakeWorkerFactory


class TestDispatchAgentPipeline:
    """P1: dispatch_agent creates a spawned agent and returns task_id."""

    @pytest.fixture(autouse=True)
    def _bind_tools(self, runtime: AgentRuntime, worker_factory: FakeWorkerFactory) -> None:
        self.runtime = runtime
        tools = create_orchestrator_tools(runtime, worker_factory)
        self.dispatch_agent = tools["dispatch_agent"]
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_dispatch_creates_managed_task(self) -> None:
        """dispatch_agent should create an agent in AgentRuntime.

        With auto-block (single worker), the agent completes before returning.
        """
        await self.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
        )

        status = self.runtime.get_status()
        assert len(status) >= 1
        # Find the spawned agent
        agent_ids = list(status.keys())
        assert any(aid.startswith("db-") for aid in agent_ids)

    @pytest.mark.asyncio
    async def test_dispatch_single_worker_auto_blocks(self) -> None:
        """Single-worker dispatch should auto-block and return completed result."""
        result = await self.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
        )

        content = json.loads(result)
        assert "task_id" in content
        assert content["agent_id"] == "db"
        assert content["status"] == "completed"
        assert content["result"] is not None

    @pytest.mark.asyncio
    async def test_dispatch_with_metadata(self) -> None:
        """dispatch_agent should pass metadata through to the spawned agent."""
        result = await self.dispatch_agent(
            agent_id="db",
            task="verify H1",
            task_type="verify",
            metadata={"hypothesis_id": "H1"},
        )

        content = json.loads(result)
        # Auto-blocked, so we get the result directly
        assert content["status"] == "completed"


class TestCheckTasksPipeline:
    """P2: check_tasks returns status and results as JSON string."""

    @pytest.fixture(autouse=True)
    def _bind_tools(
        self, runtime: AgentRuntime, worker_factory_with_result: FakeWorkerFactory
    ) -> None:
        self.runtime = runtime
        tools = create_orchestrator_tools(runtime, worker_factory_with_result)
        self.dispatch_agent = tools["dispatch_agent"]
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_check_tasks_returns_completed_results(self) -> None:
        """check_tasks should include completed agent results."""
        # First dispatch and let it auto-block/complete
        await self.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
        )

        result = await self.check_tasks(
            request="status",
        )

        content = json.loads(result)
        assert content["completed_count"] >= 1
        completed = content["completed"]
        assert len(completed) >= 1
        assert completed[0]["agent_id"] == "db"

    @pytest.mark.asyncio
    async def test_check_tasks_returns_failed_results(self) -> None:
        """check_tasks should include failed agent error summary."""
        import asyncio
        from tests.snapshot.conftest import NeverEndingLoop

        # Spawn a loop that we'll abort to create a failed agent
        loop = NeverEndingLoop()
        await self.runtime.spawn(
            "failed-agent",
            loop=loop,
            input="search errors",
            parent_id="orchestrator",
            metadata={"task_type": "scout", "original_agent_id": "logs"},
        )
        await asyncio.sleep(0.05)
        await self.runtime.abort("failed-agent", reason="API timeout after 3 retries")

        result = await self.check_tasks(
            request="status",
        )

        content = json.loads(result)
        assert content["failed_count"] >= 1
        failed = content["failed"]
        assert len(failed) >= 1
        assert "timeout" in failed[0]["error_summary"].lower()


class TestCheckTasksIncrementalReporting:
    """P2b: AgentRuntime always reports all terminal agents.

    Note: Unlike the old TaskManager, AgentRuntime does not have
    incremental reporting (mark-as-reported). All completed/failed agents
    are always visible in get_status(). This is by design — the harness
    is stateless regarding reporting.
    """

    @pytest.fixture(autouse=True)
    def _setup(
        self, runtime: AgentRuntime, worker_factory_with_result: FakeWorkerFactory
    ) -> None:
        self.runtime = runtime
        tools = create_orchestrator_tools(runtime, worker_factory_with_result)
        self.dispatch_agent = tools["dispatch_agent"]
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_completed_task_always_visible(self) -> None:
        """Completed agents remain visible in check_tasks across calls."""
        # Dispatch and auto-block
        await self.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
        )

        # First check
        result1 = await self.check_tasks(request="status")
        content1 = json.loads(result1)
        assert content1["completed_count"] >= 1

        # Second check — still visible (no mark-as-reported in AgentRuntime)
        result2 = await self.check_tasks(request="status")
        content2 = json.loads(result2)
        assert content2["completed_count"] >= 1

    @pytest.mark.asyncio
    async def test_failed_task_always_visible(self) -> None:
        """Failed agents remain visible in check_tasks across calls."""
        import asyncio
        from tests.snapshot.conftest import NeverEndingLoop

        loop = NeverEndingLoop()
        await self.runtime.spawn(
            "failed-agent",
            loop=loop,
            input="search logs",
            parent_id="orchestrator",
            metadata={"task_type": "scout", "original_agent_id": "logs"},
        )
        await asyncio.sleep(0.05)
        await self.runtime.abort("failed-agent", reason="timeout")

        # First check
        result1 = await self.check_tasks(request="status")
        content1 = json.loads(result1)
        assert content1["failed_count"] >= 1

        # Second check — still visible
        result2 = await self.check_tasks(request="status")
        content2 = json.loads(result2)
        assert content2["failed_count"] >= 1
