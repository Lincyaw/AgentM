"""Tests for Pydantic config schema validation boundaries.

Ref: designs/system-design-overview.md § Configuration System — Pydantic schema validation
Ref: designs/sub-agent.md § Configuration — AgentConfig fields

Only tests validation boundaries — cases where Pydantic SHOULD reject invalid config.
Does NOT test that valid defaults work (Pydantic guarantees that).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentm.config.schema import (
    AgentConfig,
    OrchestratorConfig,
    ScenarioConfig,
    StorageBackendConfig,
    StorageConfig,
    SystemConfig,
    SystemTypeConfig,
    RecoveryConfig,
)


class TestMandatoryFieldRejection:
    """Ref: designs/system-design-overview.md § Configuration — startup validation

    Bug: config file missing required field → system starts with None/default
    instead of failing fast at startup.
    """

    def test_system_config_rejects_missing_models(self):
        backend = StorageBackendConfig(
            backend="postgres", url="postgresql://localhost/db"
        )
        with pytest.raises(ValidationError):
            SystemConfig(
                storage=StorageConfig(checkpointer=backend, store=backend),
                recovery=RecoveryConfig(),
            )  # type: ignore[call-arg]

    def test_agent_config_accepts_missing_prompt(self):
        cfg = AgentConfig(model="gpt-4o", temperature=0.0, tools=[])
        assert cfg.prompt is None

    def test_agent_config_rejects_missing_tools(self):
        with pytest.raises(ValidationError):
            AgentConfig(model="gpt-4o", temperature=0.0, prompt="p.j2")  # type: ignore[call-arg]

    def test_orchestrator_config_rejects_missing_model(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig()  # type: ignore[call-arg]

    def test_scenario_config_rejects_missing_orchestrator(self):
        with pytest.raises(ValidationError):
            ScenarioConfig(system=SystemTypeConfig(type="hypothesis_driven"), agents={})  # type: ignore[call-arg]


class TestScenarioConfigComposition:
    """Ref: designs/system-design-overview.md § Configuration — scenario.yaml structure

    Bug: scenario config accepts structurally invalid nested objects → runtime crash
    when AgentSystemBuilder tries to access expected fields.
    """

    def test_accepts_complete_scenario(self):
        cfg = ScenarioConfig(
            system=SystemTypeConfig(type="hypothesis_driven"),
            orchestrator=OrchestratorConfig(model="gpt-4o"),
            agents={
                "db-agent": AgentConfig(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    prompt="prompts/db.j2",
                    tools=["knowledge_search"],
                )
            },
        )
        assert cfg.system.type == "hypothesis_driven"
        assert "db-agent" in cfg.agents
