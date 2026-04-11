"""Focused interface contract tests for scenario wiring and orchestrator tools."""

from __future__ import annotations

from typing import get_type_hints

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.tools.orchestrator import create_orchestrator_tools


class _MockWorkerFactory:
    def create_worker(self, agent_id: str, task_type: str):
        return None


@pytest.fixture
def dispatch_agent_func():
    tools = create_orchestrator_tools(AgentRuntime(), _MockWorkerFactory())
    return tools["dispatch_agent"]


def test_rca_and_trajectory_judger_are_discoverable_with_expected_schema_keys() -> None:
    from agentm.harness.scenario import SetupContext, get_scenario
    from agentm.scenarios import discover

    discover()
    rca = get_scenario("hypothesis_driven").setup(
        SetupContext(vault=None, trajectory=None, tool_registry=None)
    )

    assert {"scout", "verify", "deep_analyze"}.issubset(set(rca.answer_schemas))
    assert get_scenario("trajectory_judger").name == "trajectory_judger"


def test_dispatch_agent_task_type_annotation_is_str(dispatch_agent_func) -> None:
    hints = get_type_hints(dispatch_agent_func, include_extras=True)
    assert hints.get("task_type") is str


def test_runtime_and_orchestrator_exports_match_current_contract() -> None:
    from agentm.harness.loops.simple import SimpleAgentLoop

    tools = create_orchestrator_tools(AgentRuntime(), _MockWorkerFactory())

    assert hasattr(AgentRuntime, "send")
    assert hasattr(SimpleAgentLoop, "inject")
    assert "recall_history" not in tools
    assert "_set_graph_ref" not in tools
