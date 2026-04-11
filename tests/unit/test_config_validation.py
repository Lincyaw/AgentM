"""Focused schema validation boundary tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentm.config.schema import (
    AgentConfig,
    OrchestratorConfig,
    RecoveryConfig,
    ScenarioConfig,
    StorageBackendConfig,
    StorageConfig,
    SystemConfig,
    SystemTypeConfig,
)


def test_system_config_rejects_missing_models() -> None:
    backend = StorageBackendConfig(backend="postgres", url="postgresql://localhost/db")
    with pytest.raises(ValidationError):
        SystemConfig(
            storage=StorageConfig(checkpointer=backend, store=backend),
            recovery=RecoveryConfig(),
        )  # type: ignore[call-arg]


def test_agent_config_requires_tools_but_prompt_is_optional() -> None:
    cfg = AgentConfig(model="gpt-4o", temperature=0.0, tools=[])
    assert cfg.prompt is None
    with pytest.raises(ValidationError):
        AgentConfig(model="gpt-4o", temperature=0.0, prompt="p.j2")  # type: ignore[call-arg]


def test_orchestrator_and_scenario_required_sections_are_enforced() -> None:
    with pytest.raises(ValidationError):
        OrchestratorConfig()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ScenarioConfig(system=SystemTypeConfig(type="hypothesis_driven"), agents={})  # type: ignore[call-arg]


def test_complete_scenario_config_is_accepted() -> None:
    cfg = ScenarioConfig(
        system=SystemTypeConfig(type="hypothesis_driven"),
        orchestrator=OrchestratorConfig(model="gpt-4o"),
        agents={
            "db-agent": AgentConfig(
                model="gpt-4o-mini",
                temperature=0.0,
                prompt="prompts/db.j2",
                tools=["vault_search"],
            )
        },
    )
    assert cfg.system.type == "hypothesis_driven"
    assert "db-agent" in cfg.agents
