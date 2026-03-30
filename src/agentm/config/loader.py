"""Configuration loading utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from agentm.config.schema import ScenarioConfig, SystemConfig
from agentm.exceptions import ConfigError

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

T = TypeVar("T", bound=BaseModel)


def _resolve_env_ref(ref: str) -> str:
    """Resolve a single ${VAR} or ${VAR:default} reference."""
    if ":" in ref:
        var_name, default = ref.split(":", 1)
        return os.environ.get(var_name, default)
    if ref not in os.environ:
        raise ConfigError(f"Environment variable '{ref}' is not set")
    return os.environ[ref]


def substitute_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively substitute ${VAR} placeholders with environment variable values."""
    return _substitute(data)  # type: ignore[return-value]


def _substitute(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(item) for item in value]
    if isinstance(value, str):
        has_env_ref = _ENV_VAR_PATTERN.search(value) is not None

        def _replace(match: re.Match[str]) -> str:
            return _resolve_env_ref(match.group(1))

        result = _ENV_VAR_PATTERN.sub(_replace, value)
        if has_env_ref and not result:
            return None
        return result
    return value


def _load_validated_yaml(path: str | Path, model_cls: type[T]) -> T:
    """Load YAML, substitute env vars, and validate against a Pydantic model."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    resolved = substitute_env_vars(raw)
    try:
        return model_cls(**resolved)
    except Exception as e:
        raise ConfigError(f"Invalid config in {path}: {e}") from e


def load_system_config(path: Path | str) -> SystemConfig:
    """Load and validate system.yaml into a SystemConfig."""
    return _load_validated_yaml(path, SystemConfig)


def load_scenario_config(path: Path | str) -> ScenarioConfig:
    """Load and validate scenario.yaml into a ScenarioConfig."""
    return _load_validated_yaml(path, ScenarioConfig)


def load_tool_definitions(tools_dir: Path | str) -> dict[str, Any]:
    """Load all tool YAML definitions from a directory into a dict.

    Supports list format: tools: [{name: ..., module: ..., function: ...}]
    Returns {tool_name: tool_definition_dict}.
    """
    tools_dir = Path(tools_dir)
    result: dict[str, Any] = {}
    for yaml_file in sorted(tools_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in tool file {yaml_file}: {e}") from e
        if data and "tools" in data:
            tools = data["tools"]
            if isinstance(tools, list):
                for tool in tools:
                    result[tool["name"]] = tool
            elif isinstance(tools, dict):
                result.update(tools)
    return result
