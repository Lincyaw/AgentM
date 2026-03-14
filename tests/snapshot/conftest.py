"""Shared fixtures for Layer 2 snapshot tests.

Provides snapshot builders, TaskManager fixtures, and helper functions
for constructing test state without a running graph.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.agents.node.worker import AgentPool
from agentm.config.schema import (
    AgentConfig,
    ExecutionConfig,
    ScenarioConfig,
    OrchestratorConfig,
    SystemTypeConfig,
)
from agentm.core.task_manager import TaskManager
from agentm.core.tool_registry import ToolRegistry
from agentm.scenarios.rca.notebook import add_hypothesis, update_hypothesis_status
from agentm.scenarios.rca.data import DiagnosticNotebook
from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus
from agentm.scenarios.rca.enums import HypothesisStatus


class _MockSubgraph:
    """A lightweight mock subgraph that completes immediately.

    Implements the astream interface expected by TaskManager._execute_agent.
    """

    async def astream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
    ):
        from langchain_core.messages import AIMessage

        instruction = ""
        for msg in input_data.get("messages", []):
            if hasattr(msg, "content"):
                instruction = msg.content
                break
        yield (
            (),
            "updates",
            {"messages": [AIMessage(content=f"Completed: {instruction}")]},
        )
        # Simulate the generate_structured_response node that
        # create_react_agent appends when response_format is set.
        yield (
            (),
            "updates",
            {"structured_response": {"findings": f"Completed: {instruction}"}},
        )


_MOCK_SUBGRAPH = _MockSubgraph()


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
def agent_pool() -> AgentPool:
    """A real AgentPool with create_worker monkeypatched to return a mock subgraph."""
    scenario_config = ScenarioConfig(
        system=SystemTypeConfig(type="hypothesis_driven"),
        orchestrator=OrchestratorConfig(model="gpt-4"),
        agents={
            "worker": AgentConfig(
                model="gpt-4",
                temperature=0.2,
                tools=[],
                execution=ExecutionConfig(max_steps=25),
            ),
        },
    )
    pool = AgentPool(scenario_config, ToolRegistry())
    # Patch create_worker to return the mock without compiling a real agent
    pool.create_worker = lambda agent_id, task_type, task_id=None: _MOCK_SUBGRAPH  # type: ignore[assignment]
    return pool


@pytest.fixture
def task_manager() -> TaskManager:
    """A fresh TaskManager instance."""
    return TaskManager()


@pytest.fixture
def task_manager_with_completed_task() -> TaskManager:
    """A TaskManager with one completed task."""
    tm = TaskManager()
    managed = ManagedTask(
        task_id="task-completed-001",
        agent_id="db",
        instruction="check connections",
        status=AgentRunStatus.COMPLETED,
        result={"findings": "connections: 200/200, wait_queue: 47"},
        completed_at="2026-03-08T10:10:00",
        duration_seconds=15.0,
    )
    tm._tasks["task-completed-001"] = managed
    return tm


@pytest.fixture
def task_manager_with_running_task() -> TaskManager:
    """A TaskManager with one running task."""
    tm = TaskManager()
    managed = ManagedTask(
        task_id="task-running-001",
        agent_id="infra",
        instruction="scan infrastructure metrics",
        status=AgentRunStatus.RUNNING,
    )
    tm._tasks["task-running-001"] = managed
    return tm


@pytest.fixture
def task_manager_with_failed_task() -> TaskManager:
    """A TaskManager with one failed task."""
    tm = TaskManager()
    managed = ManagedTask(
        task_id="task-failed-001",
        agent_id="logs",
        instruction="search error logs",
        status=AgentRunStatus.FAILED,
        error_summary="API timeout after 3 retries",
    )
    tm._tasks["task-failed-001"] = managed
    return tm
