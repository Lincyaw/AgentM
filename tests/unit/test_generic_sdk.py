"""Tests for the generic SDK architecture: scenario registry,
middleware composition, storage backend, composite backend, task result, and
generic builder.

Ref: designs/generic-state-wrapper.md, designs/sdk-consistency.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentm.scenarios import discover

# Ensure scenario registrations are loaded for all tests in this module.
discover()


# ---------------------------------------------------------------------------
# Scenario Registry (replaces old Strategy Registry)
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    """ScenarioRegistry maps scenario names to Scenario instances.

    Bug prevented: typo in scenario_name silently returns None -> NoneType
    attribute errors deep in the execution pipeline.
    """

    def test_get_hypothesis_driven(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("hypothesis_driven")
        assert scenario.name == "hypothesis_driven"

    def test_get_trajectory_analysis(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("trajectory_analysis")
        assert scenario.name == "trajectory_analysis"

    def test_get_general_purpose(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("general_purpose")
        assert scenario.name == "general_purpose"

    def test_unknown_type_raises(self):
        from agentm.harness.scenario import get_scenario

        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("unknown_type_xyz")

    def test_list_scenarios_returns_registered(self):
        from agentm.harness.scenario import list_scenarios

        names = list_scenarios()
        assert "hypothesis_driven" in names
        assert "trajectory_analysis" in names
        assert "general_purpose" in names


# ---------------------------------------------------------------------------
# Hypothesis-Driven Strategy
# ---------------------------------------------------------------------------


class TestRCAScenario:
    """RCAScenario setup produces correct wiring."""

    def test_setup_returns_orchestrator_tools(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        tool_names = {t.name for t in wiring.orchestrator_tools}
        assert "update_hypothesis" in tool_names
        assert "remove_hypothesis" in tool_names

    def test_setup_returns_worker_tools(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert len(wiring.worker_tools) > 0

    def test_setup_returns_answer_schemas(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert "scout" in wiring.answer_schemas
        assert "deep_analyze" in wiring.answer_schemas
        assert "verify" in wiring.answer_schemas

    def test_setup_returns_output_schema(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert wiring.output_schema is not None

    def test_format_context_is_callable(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        result = wiring.format_context()
        assert isinstance(result, str)

    def test_hooks_configured(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert wiring.hooks.think_stall_enabled is True
        assert wiring.hooks.think_stall_limit == 3


# ---------------------------------------------------------------------------
# Trajectory-Analysis Scenario
# ---------------------------------------------------------------------------


class TestTrajectoryAnalysisScenario:
    """TrajectoryAnalysisScenario setup produces correct wiring."""

    def test_setup_returns_answer_schemas(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.trajectory_analysis.scenario import TrajectoryAnalysisScenario

        wiring = TrajectoryAnalysisScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert "analyze" in wiring.answer_schemas

    def test_setup_returns_output_schema(self):
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.trajectory_analysis.scenario import TrajectoryAnalysisScenario

        wiring = TrajectoryAnalysisScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        assert wiring.output_schema is not None

    def test_hooks_are_default(self):
        from agentm.harness.scenario import SetupContext
        from agentm.models.data import OrchestratorHooks
        from agentm.scenarios.trajectory_analysis.scenario import TrajectoryAnalysisScenario

        wiring = TrajectoryAnalysisScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
        default_hooks = OrchestratorHooks()
        assert wiring.hooks.think_stall_enabled == default_hooks.think_stall_enabled


# ---------------------------------------------------------------------------
# build_agent_system -- unknown scenario raises ValueError
# ---------------------------------------------------------------------------


class TestBuildAgentSystemValidation:
    """build_agent_system raises on unknown scenario names.

    Bug prevented: typo in scenario_name silently creates a broken system.
    """

    def test_unknown_scenario_raises(self):
        from agentm.builder import build_agent_system
        from agentm.config.schema import (
            OrchestratorConfig,
            ScenarioConfig,
            SystemTypeConfig,
        )

        config = ScenarioConfig(
            system=SystemTypeConfig(type="nonexistent"),
            orchestrator=OrchestratorConfig(model="gpt-4o", temperature=0.7, tools=[]),
            agents={},
        )
        with pytest.raises(ValueError, match="Unknown scenario"):
            build_agent_system("nonexistent_scenario_xyz", config)
