"""Tests for AgentMAgent defensive fallback behavior during rollout."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agentm.agents.eval_agent import AgentMAgent


_EMPTY_GRAPH: dict[str, object] = {
    "nodes": [],
    "edges": [],
    "root_causes": [],
    "component_to_service": {},
}


@pytest.mark.asyncio
async def test_should_return_fallback_graph_and_log_when_headless_raises() -> None:
    """Headless execution crash should not abort sample rollout."""
    agent = AgentMAgent()

    with (
        patch(
            "agentm.cli.run.run_investigation_headless",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch("agentm.agents.eval_agent.logger") as mock_logger,
    ):
        result = await agent.run("incident", "/tmp/data")

    assert json.loads(result.response) == _EMPTY_GRAPH
    assert result.metadata["error_type"] == "RuntimeError"
    assert result.metadata["fallback_reason"] == "exception"
    mock_logger.exception.assert_called_once()


@pytest.mark.asyncio
async def test_should_coerce_empty_response_to_fallback_graph_with_warning() -> None:
    """Empty structured response should be coerced to non-empty graph JSON."""
    agent = AgentMAgent()

    with (
        patch(
            "agentm.cli.run.run_investigation_headless",
            new=AsyncMock(return_value=("", None, "run-1", None)),
        ),
        patch("agentm.agents.eval_agent.logger") as mock_logger,
    ):
        result = await agent.run("incident", "/tmp/data")

    assert json.loads(result.response) == _EMPTY_GRAPH
    assert result.metadata["run_id"] == "run-1"
    assert result.metadata["fallback_reason"] == "empty_response"
    mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_should_log_warning_when_trajectory_json_is_invalid() -> None:
    """Trajectory parse failures should be logged but not fail the sample."""
    agent = AgentMAgent()
    valid_graph = json.dumps(_EMPTY_GRAPH)

    with (
        patch(
            "agentm.cli.run.run_investigation_headless",
            new=AsyncMock(
                return_value=(valid_graph, "{invalid", "run-2", "/tmp/t.jsonl")
            ),
        ),
        patch("agentm.agents.eval_agent.logger") as mock_logger,
    ):
        result = await agent.run("incident", "/tmp/data")

    assert result.response == valid_graph
    assert result.trajectory is None
    mock_logger.warning.assert_called_once()
