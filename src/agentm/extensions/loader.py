"""Scenario loader.

A scenario is a directory under ``<cwd>/scenarios/<name>/`` containing a
``manifest.yaml`` and (optionally) one or more scenario-local atom modules.
The yaml lists extensions in declaration order; each entry is one of:

- ``module: <python.import.path>`` — references a builtin atom by its
  importable dotted path.
- ``local: <stem>`` — references ``<scenario_dir>/<stem>.py`` as a
  scenario-local atom; the loader registers it under the synthetic module
  name ``agentm._scenarios.<scenario>.<stem>`` so subsequent
  ``importlib.import_module`` calls resolve without re-execution.

Path resolution:

- An absolute path argument loads that file directly (or its
  ``manifest.yaml`` if a directory).
- A bare name resolves to ``<cwd>/contrib/scenarios/<name>/manifest.yaml``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionLoadError


class ScenarioLoadError(ExtensionLoadError):
    """Raised when a scenario YAML cannot be resolved or validated."""


def load_scenario(name_or_path: str) -> list[tuple[str, dict[str, Any]]]:
    """Resolve and parse a scenario manifest.

    Returns a list of ``(module_path, config)`` pairs in declaration order.
    For ``local:`` entries the module is registered into ``sys.modules``
    under its synthetic name before this function returns.
    """

    candidate = Path(name_or_path)
    if candidate.is_absolute():
        manifest_path = (
            candidate / "manifest.yaml" if candidate.is_dir() else candidate
        )
    else:
        manifest_path = (
            Path(os.getcwd()) / "contrib" / "scenarios" / name_or_path / "manifest.yaml"
        )

    if not manifest_path.is_file():
        raise ScenarioLoadError(
            str(manifest_path),
            FileNotFoundError(
                f"scenario manifest not found at {manifest_path}"
            ),
        )

    return _load_from_path(manifest_path)


def _load_from_path(path: Path) -> list[tuple[str, dict[str, Any]]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ScenarioLoadError(str(path), exc) from exc

    scenario_dir = path.parent
    _ensure_scenario_import_roots(scenario_dir)
    scenario_name = _resolve_scenario_name(payload, scenario_dir, source=str(path))
    return _parse_extensions(
        payload,
        source=str(path),
        scenario_dir=scenario_dir,
        scenario_name=scenario_name,
    )


def _resolve_scenario_name(
    payload: Any,
    scenario_dir: Path,
    *,
    source: str,
) -> str:
    if not isinstance(payload, dict):
        raise ScenarioLoadError(source, ValueError("scenario must be a mapping"))

    declared = payload.get("name")
    dir_name = scenario_dir.name
    if declared is None:
        return dir_name
    if not isinstance(declared, str) or not declared:
        raise ScenarioLoadError(
            source, ValueError("scenario 'name' must be a non-empty string")
        )
    if declared != dir_name:
        raise ScenarioLoadError(
            source,
            ValueError(
                f"scenario name {declared!r} does not match directory name "
                f"{dir_name!r}"
            ),
        )
    return declared


def _parse_extensions(
    payload: Any,
    *,
    source: str,
    scenario_dir: Path,
    scenario_name: str,
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

        config = item.get("config", {})
        if not isinstance(config, dict):
            raise ScenarioLoadError(
                source,
                ValueError(_entry_error(index, "'config' must be a mapping")),
            )

        has_module = "module" in item
        has_local = "local" in item
        if has_module and has_local:
            raise ScenarioLoadError(
                source,
                ValueError(
                    _entry_error(
                        index,
                        "entry must declare exactly one of 'module' or 'local'",
                    )
                ),
            )

        if has_module:
            module = item["module"]
            if not isinstance(module, str) or not module:
                raise ScenarioLoadError(
                    source,
                    ValueError(_entry_error(index, "missing string 'module'")),
                )
            _validate_module(source, index, module)
            extensions.append((module, dict(config)))
        elif has_local:
            stem = item["local"]
            if not isinstance(stem, str) or not stem:
                raise ScenarioLoadError(
                    source,
                    ValueError(_entry_error(index, "missing string 'local'")),
                )
            synthetic = _register_local(
                source=source,
                index=index,
                scenario_dir=scenario_dir,
                scenario_name=scenario_name,
                stem=stem,
            )
            extensions.append((synthetic, dict(config)))
        else:
            raise ScenarioLoadError(
                source,
                ValueError(
                    _entry_error(
                        index,
                        "entry must declare exactly one of 'module' or 'local'",
                    )
                ),
            )

    return sort_extensions_by_requires(extensions, source=source)


def sort_extensions_by_requires(
    extensions: list[tuple[str, dict[str, Any]]],
    *,
    source: str = "<extensions>",
) -> list[tuple[str, dict[str, Any]]]:
    manifests: dict[str, ExtensionManifest] = {}
    name_by_module: dict[str, str] = {}
    entries_by_name: dict[str, tuple[str, dict[str, Any]]] = {}

    for module_path, config in extensions:
        try:
            module = importlib.import_module(module_path)
        except Exception:  # noqa: BLE001
            continue
        manifest = getattr(module, "MANIFEST", None)
        if not isinstance(manifest, ExtensionManifest):
            continue
        existing = entries_by_name.get(manifest.name)
        if existing is not None:
            raise ScenarioLoadError(
                source,
                ValueError(
                    f"extension {manifest.name!r} is loaded more than once; "
                    "duplicate extension entries are not supported"
                ),
            )
        manifests[manifest.name] = manifest
        name_by_module[module_path] = manifest.name
        entries_by_name[manifest.name] = (module_path, config)

    for name, manifest in manifests.items():
        for dep in manifest.requires:
            if dep not in entries_by_name:
                raise ScenarioLoadError(
                    source,
                    ValueError(
                        f"extension {name!r} requires {dep!r}, but {dep!r} "
                        "is not loaded"
                    ),
                )

    sorted_entries: list[tuple[str, dict[str, Any]]] = []
    temporary: set[str] = set()
    permanent: set[str] = set()
    emitted_modules: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise ScenarioLoadError(
                source, ValueError(f"extension dependency cycle involving {name!r}")
            )
        temporary.add(name)
        for dep in manifests[name].requires:
            visit(dep)
        temporary.remove(name)
        permanent.add(name)
        entry = entries_by_name[name]
        if entry[0] not in emitted_modules:
            sorted_entries.append(entry)
            emitted_modules.add(entry[0])

    for entry in extensions:
        module_path = entry[0]
        module_name = name_by_module.get(module_path)
        if module_name is None:
            if module_path not in emitted_modules:
                sorted_entries.append(entry)
                emitted_modules.add(module_path)
            continue
        visit(module_name)

    return sorted_entries


def _validate_module(source: str, index: int, module: str) -> None:
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"module {module!r} is not importable")
            ),
        )

    # Prefer a cheap existence check here: scenario manifests should validate
    # even when optional third-party runtime dependencies (for example DuckDB
    # in the RCA package) are not installed in the current test environment.
    # If the module has already been imported elsewhere, keep the stronger
    # ``install()`` assertion.
    loaded = sys.modules.get(module)
    if loaded is not None and not callable(getattr(loaded, "install", None)):
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"module {module!r} does not export install()")
            ),
        )


def _find_project_root(start: Path) -> Path | None:
    """Locate the topmost project root above ``start``.

    A scenario directory may itself be a workspace member with its own
    ``pyproject.toml`` (e.g. ``contrib/scenarios/rca/pyproject.toml``),
    so the *first* marker hit when ascending is not the project root —
    it's the nested member. We want the outermost project so
    cross-package references like ``contrib.extensions.<name>`` resolve.

    Strategy: prefer the directory containing ``.git`` if any ancestor
    has one (the canonical monorepo boundary); otherwise return the
    highest ancestor that still carries a Python project marker.
    """

    project_markers = ("pyproject.toml", "setup.py", "setup.cfg")
    highest_with_marker: Path | None = None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
        if any((candidate / m).is_file() for m in project_markers):
            highest_with_marker = candidate
    return highest_with_marker


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _ensure_scenario_import_roots(scenario_dir: Path) -> None:
    """Make in-tree scenario packages importable during manifest load.

    Scenario manifests reference modules in two ways:

    1. ``<scenario_dir>/src/<pkg>`` — editable-style scenario packages
       (e.g. ``agentm_rca.tools.duckdb_sql`` under
       ``contrib/scenarios/rca/src/agentm_rca/``).
    2. ``<project_root>/<pkg>`` — peer packages from the same checkout
       (e.g. ``contrib.extensions.rcabench_contract`` under
       ``contrib/extensions/``).

    Entry-point scripts launch with ``sys.path[0]`` pointing at the
    venv bin dir rather than the project root, so neither path is on
    ``sys.path`` by default and ``import`` fails — silently, because
    the caller in ``session_factory`` swallows the manifest load error
    and the session ends up with zero tools. We surface both roots
    here so the imports resolve regardless of how the process was
    launched. Project root is located via the standard Python project
    markers rather than hard-coded directory names.
    """

    src_root = scenario_dir / "src"
    if src_root.is_dir():
        _prepend_sys_path(src_root)

    project_root = _find_project_root(scenario_dir)
    if project_root is not None:
        _prepend_sys_path(project_root)


def _register_local(
    *,
    source: str,
    index: int,
    scenario_dir: Path,
    scenario_name: str,
    stem: str,
) -> str:
    """Resolve ``<scenario_dir>/<stem>.py`` and register it under a
    synthetic module name. Returns the synthetic name."""

    synthetic = f"agentm._scenarios.{scenario_name}.{stem}"
    if synthetic in sys.modules:
        # Idempotent: a previous load already registered this atom.
        return synthetic

    file_path = scenario_dir / f"{stem}.py"
    if not file_path.is_file():
        raise ScenarioLoadError(
            source,
            FileNotFoundError(
                _entry_error(
                    index,
                    f"local atom file not found at {file_path}",
                )
            ),
        )

    spec = importlib.util.spec_from_file_location(synthetic, file_path)
    if spec is None or spec.loader is None:
        raise ScenarioLoadError(
            source,
            RuntimeError(
                _entry_error(index, f"could not build import spec for {file_path}")
            ),
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[synthetic] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"failed to execute {file_path}: {exc}")
            ),
        ) from exc

    manifest_obj = getattr(module, "MANIFEST", None)
    if not isinstance(manifest_obj, ExtensionManifest):
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(
                    index,
                    f"local atom {stem!r} is missing a module-level "
                    "MANIFEST: ExtensionManifest constant",
                )
            ),
        )
    if manifest_obj.name != stem:
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(
                    index,
                    f"local atom {stem!r} has MANIFEST.name="
                    f"{manifest_obj.name!r}; must equal the file stem",
                )
            ),
        )
    if not callable(getattr(module, "install", None)):
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(
                    index,
                    f"local atom {stem!r} does not export install()",
                )
            ),
        )
    return synthetic


def _entry_error(index: int, detail: str) -> str:
    return f"extensions[{index}] {detail}"


__all__ = ["ScenarioLoadError", "load_scenario", "sort_extensions_by_requires"]
