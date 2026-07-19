"""Host-side scenario discovery and manifest parsing.

The loader returns data only. Scenario-local Python files are represented by
content-addressed ``ExtensionSpec`` values and are validated and executed by
the runtime extension loader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from agentm.core.abi.session_api import (
    ExtensionSpec,
    ScenarioSpec,
    normalize_extension_spec,
)


_BUILTIN_SCENARIOS: dict[str, ScenarioSpec] = {
    "empty": ScenarioSpec(extensions=()),
    "minimal": ScenarioSpec(
        extensions=tuple(
            ExtensionSpec.from_module(module)
            for module in (
                "agentm.extensions.builtin.observability",
                "agentm.extensions.builtin.operations",
                "agentm.extensions.builtin.local_resources",
                "agentm.extensions.builtin.retry_policy",
                "agentm.extensions.builtin.tool_result_cap",
                "agentm.extensions.builtin.tool_error_messages",
                "agentm.extensions.builtin.file_tools",
                "agentm.extensions.builtin.tool_bash",
                "agentm.extensions.builtin.system_prompt",
            )
        ),
    ),
}


class ScenarioManifestError(ValueError):
    """Raised when a scenario manifest violates the host-side schema."""


def _scenario_roots() -> tuple[Path, ...]:
    """Return explicit host policy roots in deterministic precedence order."""

    roots = [
        Path.cwd() / "contrib" / "scenarios",
        _agentm_home() / "contrib" / "scenarios",
        Path(__file__).parents[2] / "contrib" / "scenarios",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        identity = os.path.abspath(root)
        if identity not in seen:
            unique.append(root)
            seen.add(identity)
    return tuple(unique)


def _agentm_home() -> Path:
    configured = os.environ.get("AGENTM_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".agentm"


def _scenario_parts(name: str) -> tuple[str, str]:
    if not isinstance(name, str) or not name.strip():
        raise ScenarioManifestError("scenario name must be non-empty")
    if name != name.strip():
        raise ScenarioManifestError("scenario name cannot have surrounding whitespace")
    if ":" in name:
        base, variant = name.split(":", 1)
        if not base or not variant or ":" in variant:
            raise ScenarioManifestError(f"invalid scenario variant name: {name!r}")
        return base, f"manifest.{variant}.yaml"
    return name, "manifest.yaml"


def _find_manifest(name: str) -> Path | None:
    base, filename = _scenario_parts(name)
    if Path(base).name != base or base in {".", ".."}:
        raise ScenarioManifestError(f"invalid scenario name: {name!r}")
    for root in _scenario_roots():
        candidate = root / base / filename
        if candidate.is_file():
            return candidate
    return None


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScenarioManifestError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ScenarioManifestError(
            f"{label} {path} must be a YAML mapping, got {type(raw).__name__}"
        )
    if not all(isinstance(key, str) for key in raw):
        raise ScenarioManifestError(f"{label} {path} keys must be strings")
    return raw


def _contained_path(base_dir: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ScenarioManifestError(f"{label} must be a non-empty relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ScenarioManifestError(f"{label} must be relative: {relative!r}")
    real_base = base_dir.resolve()
    real_target = (base_dir / relative_path).resolve()
    if not real_target.is_relative_to(real_base):
        raise ScenarioManifestError(f"{label} escapes scenario directory: {relative!r}")
    return real_target


def _extension_entries(
    manifest: Mapping[str, Any],
    *,
    scenario_dir: Path,
    manifest_path: Path,
) -> list[object]:
    entries: list[object] = []
    includes = manifest.get("includes", ())
    if not isinstance(includes, Sequence) or isinstance(includes, (str, bytes)):
        raise ScenarioManifestError(
            f"'includes' must be a list of relative paths in {manifest_path}"
        )
    for include in includes:
        if not isinstance(include, str):
            raise ScenarioManifestError(
                f"scenario include must be a string in {manifest_path}"
            )
        overlay_path = _contained_path(
            scenario_dir,
            include,
            label="scenario include",
        )
        if not overlay_path.is_file():
            raise ScenarioManifestError(f"scenario overlay not found: {overlay_path}")
        overlay = _load_yaml_mapping(overlay_path, "scenario overlay")
        overlay_entries = overlay.get("extensions")
        if not isinstance(overlay_entries, Sequence) or isinstance(
            overlay_entries,
            (str, bytes),
        ):
            raise ScenarioManifestError(
                f"scenario overlay {overlay_path} must contain an extensions list"
            )
        entries.extend(overlay_entries)

    declared = manifest.get("extensions", ())
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise ScenarioManifestError(f"'extensions' must be a list in {manifest_path}")
    entries.extend(declared)
    return entries


def _extension_from_entry(entry: object, *, scenario_dir: Path) -> ExtensionSpec:
    if not isinstance(entry, Mapping):
        raise ScenarioManifestError(
            f"scenario extension must be a mapping, got {type(entry).__name__}"
        )
    if not all(isinstance(key, str) for key in entry):
        raise ScenarioManifestError("scenario extension keys must be strings")
    unknown = set(entry) - {"module", "local", "config"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ScenarioManifestError(f"unknown scenario extension fields: {names}")
    sources = [key for key in ("module", "local") if key in entry]
    if len(sources) != 1:
        raise ScenarioManifestError(
            "scenario extension must declare exactly one of 'module' or 'local'"
        )

    raw_config = entry.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ScenarioManifestError("scenario extension config must be a mapping")
    if not all(isinstance(key, str) for key in raw_config):
        raise ScenarioManifestError("scenario extension config keys must be strings")

    source_kind = sources[0]
    source_value = entry[source_kind]
    if not isinstance(source_value, str) or not source_value:
        raise ScenarioManifestError(
            f"scenario extension {source_kind} must be a non-empty string"
        )
    if source_kind == "module":
        return ExtensionSpec.from_module(source_value, raw_config)

    local_name = source_value if source_value.endswith(".py") else f"{source_value}.py"
    source_path = _contained_path(
        scenario_dir,
        local_name,
        label="scenario-local extension",
    )
    if not source_path.is_file():
        raise ScenarioManifestError(
            f"scenario-local extension not found: {source_path}"
        )
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise ScenarioManifestError(
            f"cannot read scenario-local extension {source_path}: {exc}"
        ) from exc
    digest = f"sha256:{hashlib.sha256(source_bytes).hexdigest()}"
    return ExtensionSpec.from_file(
        str(source_path),
        digest=digest,
        config=raw_config,
    )


def _load_manifest(path: Path, requested_name: str) -> ScenarioSpec:
    manifest = _load_yaml_mapping(path, "scenario manifest")
    declared_name = manifest.get("name")
    if declared_name is not None and declared_name != requested_name:
        raise ScenarioManifestError(
            f"scenario manifest {path} declares name {declared_name!r}, "
            f"expected {requested_name!r}"
        )
    scenario_dir = path.parent
    entries = _extension_entries(
        manifest,
        scenario_dir=scenario_dir,
        manifest_path=path,
    )
    extensions = tuple(
        _extension_from_entry(entry, scenario_dir=scenario_dir) for entry in entries
    )
    real_scenario_dir = scenario_dir.resolve()
    return ScenarioSpec(
        extensions=extensions,
        base_dir=str(real_scenario_dir),
    )


def packaged_scenario_names() -> tuple[str, ...]:
    """Return discoverable manifest and in-memory scenario names."""

    names = set(_BUILTIN_SCENARIOS)
    for root in _scenario_roots():
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "manifest.yaml").is_file():
                names.add(child.name)
            for manifest in child.glob("manifest.*.yaml"):
                variant = manifest.stem.removeprefix("manifest.")
                names.add(f"{child.name}:{variant}")
    return tuple(sorted(names))


def builtin_scenario_loader(scenario: str) -> ScenarioSpec:
    """Resolve a host-discoverable or packaged in-memory scenario."""

    manifest_path = _find_manifest(scenario)
    if manifest_path is not None:
        return _load_manifest(manifest_path, scenario)
    try:
        spec = _BUILTIN_SCENARIOS[scenario]
    except KeyError as exc:
        known = ", ".join(packaged_scenario_names())
        raise ValueError(
            f"unknown packaged scenario {scenario!r}; known: {known}"
        ) from exc
    extensions = tuple(normalize_extension_spec(item) for item in spec.extensions)
    return ScenarioSpec(extensions=extensions, base_dir=spec.base_dir)


def load_scenario_manifest(
    path: str | Path,
    *,
    requested_name: str,
) -> ScenarioSpec:
    """Load one host-selected scenario manifest from an explicit path."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ScenarioManifestError(f"scenario manifest not found: {manifest_path}")
    _scenario_parts(requested_name)
    return _load_manifest(manifest_path, requested_name)


__all__ = [
    "ScenarioManifestError",
    "builtin_scenario_loader",
    "load_scenario_manifest",
    "packaged_scenario_names",
]
