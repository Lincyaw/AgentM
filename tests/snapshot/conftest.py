"""Shared fixtures for Layer 2 snapshot tests.

Provides fixtures for testing orchestrator tools and error recovery
using the new AgentRuntime + WorkerLoopFactory architecture.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.harness.types import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    RunConfig,
)
from agentm.scenarios.rca.notebook import add_hypothesis, update_hypothesis_status
from agentm.scenarios.rca.data import DiagnosticNotebook
from agentm.scenarios.rca.enums import HypothesisStatus


# ---------------------------------------------------------------------------
# Fake loops for testing
# ---------------------------------------------------------------------------


class FakeWorkerLoop:
    """A controllable AgentLoop that completes with a given result."""

    def __init__(self, result_output: Any = None, delay: float = 0) -> None:
        self._result_output = result_output
        self._delay = delay
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(
        self, input: str, *, config: RunConfig | None = None
    ) -> AgentResult:
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                return event.data["result"]
        raise RuntimeError("no complete event")

    async def stream(
        self, input: str, *, config: RunConfig | None = None
    ) -> Any:
        config = config or RunConfig()
        agent_id = config.metadata.get("agent_id", "")
        if self._delay:
            await asyncio.sleep(self._delay)
        result = AgentResult(
            agent_id=agent_id,
            status=AgentStatus.COMPLETED,
            output=self._result_output or {"findings": f"Completed: {input}"},
            steps=1,
        )
        yield AgentEvent(
            type="complete", agent_id=agent_id, data={"result": result}
        )


class NeverEndingLoop:
    """A loop that stays running until cancelled."""

    def __init__(self) -> None:
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(
        self, input: str, *, config: RunConfig | None = None
    ) -> AgentResult:
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                return event.data["result"]
        raise RuntimeError("no complete event")

    async def stream(
        self, input: str, *, config: RunConfig | None = None
    ) -> Any:
        config = config or RunConfig()
        agent_id = config.metadata.get("agent_id", "")
        yield AgentEvent(type="llm_start", agent_id=agent_id, step=0)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return


class FakeWorkerFactory:
    """Mock factory that produces FakeWorkerLoop instances."""

    def __init__(self, result_output: Any = None) -> None:
        self._result_output = result_output

    def create_worker(self, agent_id: str, task_type: str) -> FakeWorkerLoop:
        return FakeWorkerLoop(result_output=self._result_output)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def notebook() -> DiagnosticNotebook:
    """A minimal DiagnosticNotebook for testing."""
    return DiagnosticNotebook(
        task_id="test-task-001",
        task_description="Database connection timeouts reported by users",
        start_time="2026-03-08T10:00:00",
    )


@pytest.fixture
def notebook_with_hypothesis() -> DiagnosticNotebook:
    """A DiagnosticNotebook with one hypothesis in various states."""
    nb = DiagnosticNotebook(
        task_id="test-task-002",
        task_description="DB connection pool investigation",
        start_time="2026-03-08T10:00:00",
    )
    nb = add_hypothesis(
        nb,
        hypothesis_id="H1",
        description="Connection pool exhaustion",
        created_at="2026-03-08T10:05:00",
    )
    nb = update_hypothesis_status(
        nb,
        hypothesis_id="H1",
        status=HypothesisStatus.INVESTIGATING,
        last_updated="2026-03-08T10:05:00",
        evidence=["Active connections 200/200"],
    )
    return nb


@pytest.fixture
def runtime() -> AgentRuntime:
    """A fresh AgentRuntime instance."""
    return AgentRuntime()


@pytest.fixture
def worker_factory() -> FakeWorkerFactory:
    """Factory with default result (findings from input)."""
    return FakeWorkerFactory()


@pytest.fixture
def worker_factory_with_result() -> FakeWorkerFactory:
    """Factory with a specific completed result."""
    return FakeWorkerFactory(
        result_output={"findings": "connections: 200/200, wait_queue: 47"}
    )
