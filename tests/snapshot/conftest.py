"""Shared fixtures for Layer 2 snapshot tests.

Provides snapshot builders, TaskManager fixtures, and helper functions
for constructing test state without a running graph.
"""

from __future__ import annotations

import pytest

from agentm.core.task_manager import TaskManager
from agentm.models.data import DiagnosticNotebook, Hypothesis, ManagedTask
from agentm.models.enums import AgentRunStatus, HypothesisStatus, Phase


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
    nb.hypotheses["H1"] = Hypothesis(
        id="H1",
        description="Connection pool exhaustion",
        status=HypothesisStatus.INVESTIGATING,
        evidence=["Active connections 200/200"],
        created_at="2026-03-08T10:05:00",
    )
    return nb


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
        result={"connections": "200/200", "wait_queue": 47},
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
        last_steps=[{"step": "query_logs", "error": "timeout"}],
    )
    tm._tasks["task-failed-001"] = managed
    return tm
