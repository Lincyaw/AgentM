from __future__ import annotations

import importlib
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

from agentm.harness.extension import ExtensionLoadError


class ScenarioLoadError(ExtensionLoadError):
    """Raised when a scenario YAML cannot be resolved or validated."""


_PACKAGE = "agentm.extensions"


def load_scenario(name_or_path: str) -> list[tuple[str, dict[str, Any]]]:
    candidate = Path(name_or_path)
    if candidate.is_absolute():
        return _load_from_path(candidate)

    resource = files(_PACKAGE).joinpath("scenarios", f"{name_or_path}.yaml")
    with as_file(resource) as resolved:
        return _load_from_path(resolved)


def _load_from_path(path: Path) -> list[tuple[str, dict[str, Any]]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ScenarioLoadError(str(path), exc) from exc
    return _parse_extensions(payload, source=str(path))


def _parse_extensions(
    payload: Any,
    *,
    source: str,
) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ScenarioLoadError(source, ValueError("scenario must be a mapping"))

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
        _validate_module(source, index, module)
        extensions.append((module, dict(config)))

    return extensions


def _validate_module(source: str, index: int, module: str) -> None:
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"module {module!r} is not importable: {exc}")
            ),
        ) from exc
    if not callable(getattr(mod, "install", None)):
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"module {module!r} does not export install()")
            ),
        )


def _entry_error(index: int, detail: str) -> str:
    return f"extensions[{index}] {detail}"


__all__ = ["ScenarioLoadError", "load_scenario"]
