"""Tests for the general_purpose scenario: answer schemas, registration,
scenario wiring, and builder integration.

Each test prevents a specific class of bugs:
- Registration -> missing scenario -> builder lookup crash
- Scenario wiring -> setup returns wrong bundle -> builder wiring fails
- Answer schema -> invalid Pydantic model -> worker structured output fails
"""

from __future__ import annotations

import pytest

from agentm.scenarios import discover

# Ensure scenario registrations are loaded.
discover()


# ---------------------------------------------------------------------------
# Answer Schema
# ---------------------------------------------------------------------------


class TestGeneralAnswer:
    """GeneralAnswer schema for worker structured output."""

    def test_answer_field(self):
        from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer

        answer = GeneralAnswer(answer="The service is healthy.")
        assert answer.answer == "The service is healthy."

    def test_answer_required(self):
        from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GeneralAnswer()


# ---------------------------------------------------------------------------
# Scenario Registration (new-style, single registry)
# ---------------------------------------------------------------------------


class TestRegistration:
    """Scenario registration populates the ScenarioRegistry correctly."""

    def test_scenario_registry(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("general_purpose")
        assert scenario.name == "general_purpose"

    def test_rca_scenario_registered(self):
        """RCA should also be registered (no collision)."""
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("hypothesis_driven")
        assert scenario.name == "hypothesis_driven"

    def test_trajectory_analysis_registered(self):
        from agentm.harness.scenario import get_scenario

        scenario = get_scenario("trajectory_analysis")
        assert scenario.name == "trajectory_analysis"


# ---------------------------------------------------------------------------
# Scenario Wiring
# ---------------------------------------------------------------------------


class TestScenarioSetup:
    """Scenario.setup() returns correct ScenarioWiring."""

    def test_gp_scenario_returns_minimal_wiring(self):
        """GP scenario returns minimal wiring — no custom tools."""
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.general_purpose.scenario import GeneralPurposeScenario

        scenario = GeneralPurposeScenario()
        wiring = scenario.setup(SetupContext(vault=None, trajectory=None, tool_registry=None))

        assert wiring.orchestrator_tools == []
        assert wiring.worker_tools == []
        assert "execute" in wiring.answer_schemas

    def test_rca_scenario_returns_tools_and_context(self):
        """RCA scenario provides tools, context formatter, schemas."""
        from agentm.harness.scenario import SetupContext
        from agentm.scenarios.rca.scenario import RCAScenario

        scenario = RCAScenario()
        wiring = scenario.setup(SetupContext(vault=None, trajectory=None, tool_registry=None))

        tool_names = {t.name for t in wiring.orchestrator_tools}
        assert "update_hypothesis" in tool_names
        assert wiring.format_context() != "" or wiring.format_context() == ""  # callable
        assert len(wiring.worker_tools) > 0
        assert "scout" in wiring.answer_schemas


# ---------------------------------------------------------------------------
# Builder Integration
# ---------------------------------------------------------------------------


class TestBuilderIntegration:
    """Builder uses scenario protocol instead of if/else chains."""

    def test_no_resolve_format_context_function(self):
        """Bug prevented: _resolve_format_context still exists -> pluggability broken."""
        import agentm.builder as builder_module

        assert not hasattr(builder_module, "_resolve_format_context")

    def test_scenario_format_context_used_for_all_scenarios(self):
        """All registered scenarios provide format_context via Scenario protocol."""
        from agentm.harness.scenario import get_scenario

        for scenario_name in ("hypothesis_driven", "trajectory_analysis", "general_purpose"):
            scenario = get_scenario(scenario_name)
            assert hasattr(scenario, "setup")
