"""P1, P2: Tool pipeline tests for dispatch/check data flow.

Migrated to AgentRuntime + WorkerLoopFactory.
Tools return JSON strings that orchestrator parses.
"""
from __future__ import annotations

import json

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.tools.orchestrator import create_orchestrator_tools

from tests.snapshot.conftest import FakeWorkerFactory


class TestDispatchAgentPipeline:
    """P1: dispatch_agent creates a spawned agent and returns task metadata."""

    @pytest.fixture(autouse=True)
    def _bind_tools(self, runtime: AgentRuntime, worker_factory: FakeWorkerFactory) -> None:
        self.runtime = runtime
        tools = create_orchestrator_tools(runtime, worker_factory)
        self.dispatch_agent = tools["dispatch_agent"]

    @pytest.mark.asyncio
    async def test_dispatch_returns_completed_payload_and_updates_runtime(self) -> None:
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

        status = self.runtime.get_status()
        assert any(agent_id.startswith("db-") for agent_id in status)


class TestCheckTasksPipeline:
    """P2: check_tasks reports completed and failed workers in expected shape."""

    @pytest.fixture(autouse=True)
    def _bind_tools(
        self, runtime: AgentRuntime, worker_factory_with_result: FakeWorkerFactory
    ) -> None:
        self.runtime = runtime
        tools = create_orchestrator_tools(runtime, worker_factory_with_result)
        self.dispatch_agent = tools["dispatch_agent"]
        self.check_tasks = tools["check_tasks"]

    @pytest.mark.asyncio
    async def test_check_tasks_reports_completed_agent_results(self) -> None:
        await self.dispatch_agent(
            agent_id="db",
            task="check connections",
            task_type="scout",
        )

        result = await self.check_tasks(request="status")

        content = json.loads(result)
        assert content["completed_count"] >= 1
        assert any(item["agent_id"] == "db" for item in content["completed"])

    @pytest.mark.asyncio
    async def test_check_tasks_reports_failed_agent_error_summary(self) -> None:
        import asyncio
        from tests.snapshot.conftest import NeverEndingLoop

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

        result = await self.check_tasks(request="status")

        content = json.loads(result)
        assert content["failed_count"] >= 1
        assert any("timeout" in item["error_summary"].lower() for item in content["failed"])
