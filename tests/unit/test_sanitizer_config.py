"""Focused tests for sanitizer config schema integration."""

from __future__ import annotations

from pathlib import Path

import yaml

from agentm.config.schema import OrchestratorConfig, SanitizerConfig, ScenarioConfig


def test_sanitizer_config_defaults_and_custom_values() -> None:
    defaults = SanitizerConfig()
    assert defaults.enabled is False
    assert defaults.periodic_interval == 5
    assert defaults.block_on == ["C1", "C2", "J3"]

    custom = SanitizerConfig(enabled=True, critic_model="gpt-4o-mini", periodic_interval=10, disable=["P3"])
    assert custom.enabled is True
    assert custom.critic_model == "gpt-4o-mini"
    assert custom.periodic_interval == 10
    assert custom.disable == ["P3"]


def test_orchestrator_config_accepts_optional_sanitizer_field() -> None:
    assert OrchestratorConfig(model="gpt-4o").sanitizer is None
    assert OrchestratorConfig(model="gpt-4o", sanitizer=SanitizerConfig(enabled=True)).sanitizer.enabled is True  # type: ignore[union-attr]


def test_rca_hypothesis_yaml_parses_with_sanitizer_block_present() -> None:
    yaml_path = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "rca_hypothesis" / "scenario.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = ScenarioConfig(**raw)
    assert cfg.system.type == "hypothesis_driven"
    assert cfg.orchestrator.sanitizer is not None
    assert cfg.orchestrator.sanitizer.periodic_interval == 5
    assert cfg.orchestrator.sanitizer.drift_window == 3
