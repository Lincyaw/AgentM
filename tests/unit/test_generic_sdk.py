"""Focused regression tests for scenario registry and scenario wiring contracts."""

from __future__ import annotations

import pytest

from agentm.scenarios import discover

discover()


def test_scenario_registry_resolves_known_scenarios_and_rejects_unknown() -> None:
    from agentm.harness.scenario import get_scenario, list_scenarios

    assert get_scenario("hypothesis_driven").name == "hypothesis_driven"
    assert get_scenario("general_purpose").name == "general_purpose"
    assert get_scenario("trajectory_judger").name == "trajectory_judger"
    assert {"hypothesis_driven", "general_purpose", "trajectory_judger"}.issubset(set(list_scenarios()))

    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("unknown_type_xyz")


def test_rca_scenario_setup_exposes_core_wiring() -> None:
    from agentm.harness.scenario import SetupContext
    from agentm.scenarios.rca.scenario import RCAScenario

    wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
    orchestrator_tools = {t.name for t in wiring.orchestrator_tools}
    assert {"update_hypothesis", "remove_hypothesis"}.issubset(orchestrator_tools)
    assert len(wiring.worker_tools) > 0
    assert {"scout", "verify", "deep_analyze"}.issubset(set(wiring.answer_schemas))
    assert wiring.output_schema is not None


def test_trajectory_judger_setup_is_registered_and_loadable() -> None:
    from agentm.harness.scenario import SetupContext, get_scenario

    wiring = get_scenario("trajectory_judger").setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
    assert wiring is not None


def test_build_agent_system_rejects_unknown_scenario_name() -> None:
    from agentm.builder import build_agent_system
    from agentm.config.schema import OrchestratorConfig, ScenarioConfig, SystemTypeConfig

    config = ScenarioConfig(
        system=SystemTypeConfig(type="nonexistent"),
        orchestrator=OrchestratorConfig(model="gpt-4o", temperature=0.7, tools=[]),
        agents={},
    )
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_agent_system("nonexistent_scenario_xyz", config)
