"""Tests for cross-module interface consistency.

These tests verify that interfaces between modules are compatible --
types, return values, and function signatures agree across module boundaries.

Bug prevented: Module A expects Module B to accept type X, but Module B's
signature says type Y -> runtime TypeError when the modules interact.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.tools.orchestrator import create_orchestrator_tools


class _MockWorkerFactory:
    """Minimal WorkerFactory for signature testing."""

    def create_worker(self, agent_id: str, task_type: str):
        return None


@pytest.fixture
def dispatch_agent_func():
    """Get dispatch_agent function from factory for testing."""
    runtime = AgentRuntime()
    factory = _MockWorkerFactory()
    tools = create_orchestrator_tools(runtime, factory)
    return tools["dispatch_agent"]


class TestTaskTypeLiteralConsistency:
    """Ref: designs/sub-agent.md, designs/orchestrator.md

    task_type is str (widened from Literal) so that all scenario task types
    flow through. The answer_schemas dict in ScenarioWiring is the source
    of truth for valid task_type values.

    Bug prevented: answer_schemas missing a task_type that workers use ->
    KeyError at runtime when creating the structured output model.
    """

    def test_answer_schema_covers_rca_task_types(self):
        """RCA scenario wiring must include the three RCA worker task types."""
        from agentm.harness.scenario import get_scenario, SetupContext
        from agentm.scenarios import discover

        discover()
        scenario = get_scenario("hypothesis_driven")
        wiring = scenario.setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        for task_type in ("scout", "verify", "deep_analyze"):
            assert task_type in wiring.answer_schemas, (
                f"RCA wiring missing task type: {task_type!r}"
            )

    def test_answer_schema_covers_trajectory_analysis_task_types(self):
        """Trajectory analysis wiring must include the 'analyze' task type."""
        from agentm.harness.scenario import get_scenario, SetupContext
        from agentm.scenarios import discover

        discover()
        scenario = get_scenario("trajectory_analysis")
        wiring = scenario.setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert "analyze" in wiring.answer_schemas, (
            "Trajectory analysis wiring missing task type: 'analyze'"
        )

    def test_task_type_is_str_annotation(self, dispatch_agent_func):
        """dispatch_agent task_type must be annotated as str (not Literal)."""
        hints = get_type_hints(dispatch_agent_func, include_extras=True)
        annotation = hints.get("task_type")
        assert annotation is str, (
            f"dispatch_agent task_type should be str, got: {annotation}"
        )

    def test_dispatch_agent_task_type_is_str(self, dispatch_agent_func):
        """dispatch_agent task_type must be str to support multiple scenario types."""
        dispatch_hints = get_type_hints(dispatch_agent_func, include_extras=True)
        dispatch_annotation = dispatch_hints.get("task_type")

        assert dispatch_annotation is str, (
            f"dispatch_agent task_type should be str, got: {dispatch_annotation}"
        )


class TestHooksModuleExports:
    """Instruction injection is now handled by AgentRuntime.send() +
    AgentLoop.inject() -- no middleware hook needed.

    Verify the runtime has the send method and SimpleAgentLoop has inject.
    """

    def test_runtime_has_send_method(self):
        from agentm.harness.runtime import AgentRuntime

        assert hasattr(AgentRuntime, "send")

    def test_simple_agent_loop_has_inject_method(self):
        from agentm.harness.loops.simple import SimpleAgentLoop

        assert hasattr(SimpleAgentLoop, "inject")


class TestOrchestratorToolsCleanup:
    """Verify dead code has been removed from orchestrator tools.

    Bug prevented: recall_history and _set_graph_ref still exported ->
    stale LangGraph dependencies remain in the SDK boundary.
    """

    def test_no_recall_history_in_tools(self):
        """recall_history was removed in Phase 3A cleanup."""
        runtime = AgentRuntime()
        factory = _MockWorkerFactory()
        tools = create_orchestrator_tools(runtime, factory)
        assert "recall_history" not in tools

    def test_no_set_graph_ref_in_tools(self):
        """_set_graph_ref was removed in Phase 3A cleanup."""
        runtime = AgentRuntime()
        factory = _MockWorkerFactory()
        tools = create_orchestrator_tools(runtime, factory)
        assert "_set_graph_ref" not in tools
