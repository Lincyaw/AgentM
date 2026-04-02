"""Tests for SanitizerConfig integration in the config schema.

Ref: designs/investigation-sanitizer.md § Configuration
Bug prevented: invalid or missing sanitizer config accepted at startup,
causing runtime errors when the middleware tries to read fields.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agentm.config.schema import (
    OrchestratorConfig,
    SanitizerConfig,
    ScenarioConfig,
)


class TestSanitizerConfigDefaults:
    """Verify SanitizerConfig default values match the design spec."""

    def test_defaults(self):
        cfg = SanitizerConfig()
        assert cfg.enabled is False
        assert cfg.critic_model == ""
        assert cfg.periodic_interval == 5
        assert cfg.drift_window == 3
        assert cfg.drift_threshold == 3
        assert cfg.max_block_retries == 3
        assert cfg.block_on == ["C1", "C2", "J3"]
        assert cfg.warn_on == ["E1", "E2", "E3", "E4", "C4", "J2", "P1"]
        assert cfg.disable == []

    def test_custom_values(self):
        cfg = SanitizerConfig(
            enabled=True,
            critic_model="gpt-4o-mini",
            periodic_interval=10,
            drift_window=5,
            drift_threshold=5,
            max_block_retries=2,
            block_on=["C1"],
            warn_on=["E1"],
            disable=["P3"],
        )
        assert cfg.enabled is True
        assert cfg.critic_model == "gpt-4o-mini"
        assert cfg.periodic_interval == 10
        assert cfg.drift_window == 5
        assert cfg.max_block_retries == 2
        assert cfg.block_on == ["C1"]
        assert cfg.warn_on == ["E1"]
        assert cfg.disable == ["P3"]


class TestOrchestratorConfigSanitizer:
    """Verify OrchestratorConfig accepts the optional sanitizer field."""

    def test_without_sanitizer(self):
        """Backward compat: sanitizer defaults to None when omitted."""
        cfg = OrchestratorConfig(model="gpt-4o")
        assert cfg.sanitizer is None

    def test_with_sanitizer(self):
        cfg = OrchestratorConfig(
            model="gpt-4o",
            sanitizer=SanitizerConfig(enabled=True),
        )
        assert cfg.sanitizer is not None
        assert cfg.sanitizer.enabled is True

    def test_with_sanitizer_none_explicit(self):
        cfg = OrchestratorConfig(model="gpt-4o", sanitizer=None)
        assert cfg.sanitizer is None


class TestScenarioYamlParsing:
    """Parse the actual rca_hypothesis scenario.yaml to verify schema compatibility."""

    def test_rca_hypothesis_yaml_parses(self):
        yaml_path = (
            Path(__file__).resolve().parents[2]
            / "config"
            / "scenarios"
            / "rca_hypothesis"
            / "scenario.yaml"
        )
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        cfg = ScenarioConfig(**raw)
        assert cfg.system.type == "hypothesis_driven"
        assert cfg.orchestrator.sanitizer is not None
        assert cfg.orchestrator.sanitizer.enabled is True
        assert cfg.orchestrator.sanitizer.periodic_interval == 5
        assert cfg.orchestrator.sanitizer.block_on == ["C1", "C2", "J3"]
        assert cfg.orchestrator.sanitizer.warn_on == [
            "E1", "E2", "E3", "E4", "C4", "J2", "P1",
        ]

    def test_rca_hypothesis_yaml_sanitizer_drift_window(self):
        yaml_path = (
            Path(__file__).resolve().parents[2]
            / "config"
            / "scenarios"
            / "rca_hypothesis"
            / "scenario.yaml"
        )
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        cfg = ScenarioConfig(**raw)
        assert cfg.orchestrator.sanitizer is not None
        assert cfg.orchestrator.sanitizer.drift_window == 3
