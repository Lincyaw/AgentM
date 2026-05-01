from __future__ import annotations

from pathlib import Path

import yaml


def _load_scenario(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_shipped_scenarios_reference_registry_provider_ids() -> None:
    root = Path(__file__).resolve().parents[3] / "config" / "scenarios"

    for scenario_path in sorted(root.glob("*/scenario.yaml")):
        payload = _load_scenario(scenario_path)

        orchestrator = payload.get("orchestrator")
        assert isinstance(orchestrator, dict), f"{scenario_path} missing orchestrator block"
        provider = orchestrator.get("provider")
        assert isinstance(provider, str) and provider, (
            f"{scenario_path} orchestrator must declare a provider registry key"
        )

        agents = payload.get("agents")
        assert isinstance(agents, dict), f"{scenario_path} missing agents block"
        for agent_name, agent_config in agents.items():
            assert isinstance(agent_config, dict), f"{scenario_path} agent {agent_name} must be a mapping"
            agent_provider = agent_config.get("provider")
            assert isinstance(agent_provider, str) and agent_provider, (
                f"{scenario_path} agent {agent_name} must declare a provider registry key"
            )


def test_system_model_catalog_uses_registry_provider_mapping() -> None:
    path = Path(__file__).resolve().parents[3] / "config" / "system.yaml"
    payload = _load_scenario(path)

    models = payload.get("models")
    assert isinstance(models, dict), "config/system.yaml must declare models"

    for model_name, config in models.items():
        assert isinstance(config, dict), f"model {model_name} must be a mapping"
        provider = config.get("provider")
        declared_model = config.get("model")
        assert isinstance(provider, str) and provider, (
            f"model {model_name} must declare a provider registry key"
        )
        assert isinstance(declared_model, str) and declared_model, (
            f"model {model_name} must declare its registry model id"
        )
        assert "api_key" not in config, (
            f"model {model_name} must not embed direct api_key configuration"
        )
