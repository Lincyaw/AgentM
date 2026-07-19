"""Scenario loader — resolves named scenarios from manifest.yaml files.

Search order for a bare name:
1. ``contrib/scenarios/<name>/manifest.yaml`` relative to the agentm package root
2. ``<cwd>/contrib/scenarios/<name>/manifest.yaml``
3. Built-in hardcoded scenarios (empty, minimal)

A ``manifest.yaml`` declares extensions as a list of entries:
- ``module: <python.import.path>`` — builtin atom
- ``local: <stem>`` — ``<scenario_dir>/<stem>.py``

Each entry may carry an optional ``config:`` dict.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from agentm.core.abi.session_api import ExtensionSpec, ScenarioSpec


_BUILTIN_SCENARIOS: dict[str, ScenarioSpec] = {
    "empty": ScenarioSpec(extensions=()),
    "minimal": ScenarioSpec(
        extensions=(
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.operations", {}),
            ("agentm.extensions.builtin.local_resources", {}),
            ("agentm.extensions.builtin.retry_policy", {}),
            ("agentm.extensions.builtin.tool_result_cap", {}),
            ("agentm.extensions.builtin.tool_error_messages", {}),
            ("agentm.extensions.builtin.file_tools", {}),
            ("agentm.extensions.builtin.tool_bash", {}),
            ("agentm.extensions.builtin.system_prompt", {}),
        ),
    ),
}


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(root: Path) -> None:
        resolved = str(root.resolve())
        if resolved not in seen:
            roots.append(root)
            seen.add(resolved)

    try:
        import agentm as _pkg

        pkg_dir = Path(_pkg.__file__).parent
        for _ in range(4):
            if (pkg_dir / "contrib").is_dir():
                add(pkg_dir)
                break
            pkg_dir = pkg_dir.parent
    except Exception:
        logger.warning("cannot locate agentm package root for scenario search")
    add(Path.cwd())
    return roots


def _find_manifest(name: str) -> Path | None:
    if ":" in name:
        base, variant = name.split(":", 1)
        filename = f"manifest.{variant}.yaml"
    else:
        base = name
        filename = "manifest.yaml"

    for root in _candidate_roots():
        candidate = root / "contrib" / "scenarios" / base / filename
        if candidate.is_file():
            return candidate
    return None


def _register_local_module(scenario_dir: Path, scenario_name: str, stem: str) -> str:
    """Register a scenario-local .py file as an importable module."""
    module_name = f"agentm._scenarios.{scenario_name}.{stem}"
    if module_name in sys.modules:
        return module_name
    source = scenario_dir / f"{stem}.py"
    if not source.is_file():
        raise FileNotFoundError(f"local atom not found: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    parent_pkg = f"agentm._scenarios.{scenario_name}"
    if parent_pkg not in sys.modules:
        grand = "agentm._scenarios"
        if grand not in sys.modules:
            import types

            pkg = types.ModuleType(grand)
            pkg.__path__ = []
            pkg.__package__ = grand
            sys.modules[grand] = pkg
        import types

        parent = types.ModuleType(parent_pkg)
        parent.__path__ = [str(scenario_dir)]
        parent.__package__ = parent_pkg
        sys.modules[parent_pkg] = parent
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module_name


def _load_manifest(path: Path) -> ScenarioSpec:
    """Parse a manifest.yaml into a ScenarioSpec."""
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"manifest.yaml must be a YAML mapping, got {type(data).__name__}"
        )

    scenario_dir = path.parent
    scenario_name = data.get("name") or scenario_dir.name

    raw_extensions: list[dict] = []
    includes = data.get("includes")
    if includes is not None:
        if not isinstance(includes, list):
            raise ValueError(f"'includes' must be a list of relative paths in {path}")
        for inc_path in includes:
            overlay_path = scenario_dir / inc_path
            if not overlay_path.is_file():
                raise FileNotFoundError(f"overlay not found: {overlay_path}")
            overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
            if not isinstance(overlay, dict) or not isinstance(
                overlay.get("extensions"), list
            ):
                raise ValueError(
                    f"overlay {overlay_path} must contain an 'extensions' list"
                )
            raw_extensions.extend(overlay["extensions"])

    declared = data.get("extensions")
    if declared is not None:
        if not isinstance(declared, list):
            raise ValueError(f"'extensions' must be a list in {path}")
        raw_extensions.extend(declared)

    if not raw_extensions:
        raise ValueError(
            f"scenario must declare extensions (via 'extensions' or 'includes') in {path}"
        )

    extensions: list[ExtensionSpec] = []

    for entry in raw_extensions:
        if not isinstance(entry, dict):
            raise ValueError(
                f"each extension entry must be a mapping, got {type(entry).__name__}"
            )
        config = dict(entry.get("config") or {})
        if "module" in entry:
            extensions.append((entry["module"], config))
        elif "local" in entry:
            module_name = _register_local_module(
                scenario_dir, scenario_name, entry["local"]
            )
            extensions.append((module_name, config))
        else:
            raise ValueError(
                f"extension entry must have 'module' or 'local' key: {entry}"
            )

    return ScenarioSpec(extensions=tuple(extensions), base_dir=str(scenario_dir))


def packaged_scenario_names() -> tuple[str, ...]:
    names = set(_BUILTIN_SCENARIOS.keys())
    for root in _candidate_roots():
        scenarios_dir = root / "contrib" / "scenarios"
        if scenarios_dir.is_dir():
            for child in scenarios_dir.iterdir():
                if not child.is_dir():
                    continue
                if (child / "manifest.yaml").is_file():
                    names.add(child.name)
                for mf in child.glob("manifest.*.yaml"):
                    variant = mf.stem.removeprefix("manifest.")
                    names.add(f"{child.name}:{variant}")
    return tuple(sorted(names))


def builtin_scenario_loader(scenario: str) -> ScenarioSpec:
    """Resolve a scenario name to a ScenarioSpec.

    Tries manifest.yaml on disk first, falls back to built-in hardcoded
    scenarios.
    """
    manifest_path = _find_manifest(scenario)
    if manifest_path is not None:
        logger.debug("loading scenario {!r} from {}", scenario, manifest_path)
        return _load_manifest(manifest_path)

    if scenario in _BUILTIN_SCENARIOS:
        spec = _BUILTIN_SCENARIOS[scenario]
        return ScenarioSpec(
            extensions=tuple(
                (module, dict(config)) for module, config in spec.extensions
            ),
            base_dir=spec.base_dir,
        )

    known = ", ".join(packaged_scenario_names())
    raise ValueError(f"unknown scenario {scenario!r}; known: {known}")


__all__ = [
    "builtin_scenario_loader",
    "packaged_scenario_names",
]
