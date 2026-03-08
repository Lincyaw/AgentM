"""Configuration loading utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from agentm.config.schema import ScenarioConfig, SystemConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def substitute_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively substitute ${VAR} placeholders with environment variable values."""
    return _substitute(data)  # type: ignore[return-value]


def _substitute(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(item) for item in value]
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise KeyError(f"Environment variable '{var_name}' is not set")
            return os.environ[var_name]
        return _ENV_VAR_PATTERN.sub(_replace, value)
    return value


def load_system_config(path: Path | str) -> SystemConfig:
    """Load and validate system.yaml into a SystemConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    resolved = substitute_env_vars(raw)
    return SystemConfig(**resolved)


def load_scenario_config(path: Path | str) -> ScenarioConfig:
    """Load and validate scenario.yaml into a ScenarioConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    resolved = substitute_env_vars(raw)
    return ScenarioConfig(**resolved)


def load_tool_definitions(tools_dir: Path | str) -> dict[str, Any]:
    """Load all tool YAML definitions from a directory into a dict."""
    tools_dir = Path(tools_dir)
    result: dict[str, Any] = {}
    for yaml_file in sorted(tools_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text())
        if data and "tools" in data:
            result.update(data["tools"])
    return result
