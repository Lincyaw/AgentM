"""Focused regression tests for scenario registration and wiring."""

from __future__ import annotations

import pytest

from agentm.scenarios import discover

discover()


def test_general_answer_requires_answer_field() -> None:
    from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer
    import pydantic

    assert GeneralAnswer(answer="ok").answer == "ok"
    with pytest.raises(pydantic.ValidationError):
        GeneralAnswer()


def test_core_scenarios_are_registered() -> None:
    from agentm.harness.scenario import get_scenario

    assert get_scenario("general_purpose").name == "general_purpose"
    assert get_scenario("hypothesis_driven").name == "hypothesis_driven"
    assert get_scenario("trajectory_judger").name == "trajectory_judger"


def test_general_purpose_wiring_is_minimal_but_has_execute_schema() -> None:
    from agentm.harness.scenario import SetupContext
    from agentm.scenarios.general_purpose.scenario import GeneralPurposeScenario

    wiring = GeneralPurposeScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
    assert wiring.orchestrator_tools == []
    assert wiring.worker_tools == []
    assert "execute" in wiring.answer_schemas


def test_rca_wiring_exposes_tools_and_answer_schemas() -> None:
    from agentm.harness.scenario import SetupContext
    from agentm.scenarios.rca.scenario import RCAScenario

    wiring = RCAScenario().setup(SetupContext(vault=None, trajectory=None, tool_registry=None))
    assert "update_hypothesis" in {t.name for t in wiring.orchestrator_tools}
    assert len(wiring.worker_tools) > 0
    assert "scout" in wiring.answer_schemas


def test_builder_removed_legacy_resolve_format_context_helper() -> None:
    import agentm.builder as builder_module

    assert not hasattr(builder_module, "_resolve_format_context")
