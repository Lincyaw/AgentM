"""Shared fixtures for Layer 2 snapshot tests.

Provides fixtures for testing orchestrator tools and error recovery
using the new AgentRuntime + WorkerLoopFactory architecture.
"""
from __future__ import annotations

import pytest

from agentm.harness.runtime import AgentRuntime

from tests.helpers import FakeAgentLoop, FakeWorkerFactory, NeverEndingLoop  # noqa: F401


# Backward-compatible alias for snapshot tests that reference FakeWorkerLoop
FakeWorkerLoop = FakeAgentLoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
