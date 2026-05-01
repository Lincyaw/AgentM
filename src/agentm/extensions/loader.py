from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml
from dataclasses import dataclass

from agentm.harness.extension import ExtensionLoadError


class ScenarioLoadError(ExtensionLoadError):
    """Raised when a scenario YAML cannot be resolved or validated."""


_PACKAGE = "agentm.extensions"


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    name: str
    description: str
    extensions: list[tuple[str, dict[str, Any]]]
    provider: str
    model: str
    provider_config: dict[str, Any]


def load_scenario(name_or_path: str) -> list[tuple[str, dict[str, Any]]]:
    return load_scenario_definition(name_or_path).extensions


def load_scenario_definition(name_or_path: str) -> ScenarioDefinition:
    candidate = Path(name_or_path)
    if candidate.is_absolute():
        return _load_from_path(candidate)

    resource = files(_PACKAGE).joinpath("scenarios", f"{name_or_path}.yaml")
    with as_file(resource) as resolved:
        return _load_from_path(resolved)


def _load_from_path(path: Path) -> ScenarioDefinition:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ScenarioLoadError(str(path), exc) from exc
    return _parse_definition(payload, source=str(path))


def _parse_definition(
    payload: Any,
    *,
    source: str,
) -> ScenarioDefinition:
    if not isinstance(payload, dict):
        raise ScenarioLoadError(source, ValueError("scenario must be a mapping"))

    raw_name = payload.get("name")
    raw_description = payload.get("description")
    if raw_name is not None and (not isinstance(raw_name, str) or not raw_name):
        raise ScenarioLoadError(source, ValueError("'name' must be a string"))
    if raw_description is not None and not isinstance(raw_description, str):
        raise ScenarioLoadError(
            source, ValueError("'description' must be a string when present")
        )
    name = raw_name or Path(source).stem
    description = raw_description or ""

    raw_provider = payload.get("provider", {})
    if not isinstance(raw_provider, dict):
        raise ScenarioLoadError(
            source, ValueError("'provider' must be a mapping when present")
        )
    provider = raw_provider.get("id", "anthropic")
    model = raw_provider.get("model", "claude-sonnet-4-6")
    provider_config = raw_provider.get("config", {})
    if not isinstance(provider, str) or not provider:
        raise ScenarioLoadError(source, ValueError("'provider.id' must be a string"))
    if not isinstance(model, str) or not model:
        raise ScenarioLoadError(
            source, ValueError("'provider.model' must be a string")
        )
    if not isinstance(provider_config, dict):
        raise ScenarioLoadError(
            source, ValueError("'provider.config' must be a mapping")
        )

    raw_extensions = payload.get("extensions")
    if not isinstance(raw_extensions, list):
        raise ScenarioLoadError(source, ValueError("'extensions' must be a list"))

    extensions: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(raw_extensions):
        if not isinstance(item, dict):
            raise ScenarioLoadError(
                source,
                ValueError(_entry_error(index, "must be a mapping")),
            )

        module = item.get("module")
        config = item.get("config", {})
        if not isinstance(module, str) or not module:
            raise ScenarioLoadError(
                source,
                ValueError(_entry_error(index, "missing string 'module'")),
            )
        if not isinstance(config, dict):
            raise ScenarioLoadError(
                source,
                ValueError(_entry_error(index, "'config' must be a mapping")),
            )
        extensions.append((module, dict(config)))

    return ScenarioDefinition(
        name=name,
        description=description,
        extensions=extensions,
        provider=provider,
        model=model,
        provider_config=dict(provider_config),
    )


def _entry_error(index: int, detail: str) -> str:
    return f"extensions[{index}] {detail}"


__all__ = ["ScenarioDefinition", "ScenarioLoadError", "load_scenario", "load_scenario_definition"]
