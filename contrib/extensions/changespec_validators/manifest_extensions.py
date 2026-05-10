"""Validator for ``ChangeSpec.kind == 'manifest_extensions'``.

A ``manifest_extensions`` change rewrites the ``extensions:`` list in a
scenario's ``manifest.yaml``. The validator round-trips the proposed new
manifest content through the scenario loader and rejects if the loader
rejects (catching unknown atoms, bad ``local:`` references, etc.).

ChangeSpec semantics:

- ``path``: scenario manifest path (relative or absolute).
- ``new_content``: full replacement YAML for the manifest. The validator
  parses it, ensures the ``extensions`` list is well-formed, then writes
  it to a temp scratch dir and asks the loader to resolve it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml


def validate(
    change_spec: dict[str, Any], cwd: Path, target_scenario: str
) -> dict[str, Any]:
    path_value = change_spec.get("path")
    new_content = change_spec.get("new_content")
    if not isinstance(path_value, str) or not path_value:
        return {
            "ok": False,
            "error": (
                "evidence missing: target.path is required for "
                "kind='manifest_extensions'"
            ),
            "resolved_path": None,
        }
    if not isinstance(new_content, str) or not new_content:
        return {
            "ok": False,
            "error": (
                "evidence missing: target.new_content must be a non-empty "
                "manifest YAML string"
            ),
            "resolved_path": None,
        }

    resolved = _resolve_under_scenario(cwd, target_scenario, path_value)
    if resolved is None:
        return {
            "ok": False,
            "error": (
                f"target.path must resolve under "
                f"contrib/scenarios/{target_scenario}/; got {path_value!r}"
            ),
            "resolved_path": None,
        }
    if resolved.name != "manifest.yaml":
        return {
            "ok": False,
            "error": (
                "kind='manifest_extensions' must target a manifest.yaml file"
            ),
            "resolved_path": None,
        }

    try:
        parsed = yaml.safe_load(new_content)
    except yaml.YAMLError as exc:
        return {
            "ok": False,
            "error": f"new_content is not valid YAML: {exc}",
            "resolved_path": None,
        }
    if not isinstance(parsed, dict) or "extensions" not in parsed:
        return {
            "ok": False,
            "error": "new manifest must declare an 'extensions' list",
            "resolved_path": None,
        }
    extensions = parsed.get("extensions")
    if not isinstance(extensions, list):
        return {
            "ok": False,
            "error": "'extensions' must be a list",
            "resolved_path": None,
        }

    # Round-trip through the loader: copy the existing scenario tree to a
    # scratch dir, overwrite the manifest, ask the loader to resolve it.
    # Loader errors come back as ScenarioLoadError; we surface their text.
    try:
        from agentm.extensions.loader import (
            ScenarioLoadError,
            load_scenario,
        )
    except ImportError as exc:  # pragma: no cover - import path stable
        return {
            "ok": False,
            "error": f"loader unavailable: {exc}",
            "resolved_path": None,
        }

    scenario_root = (cwd / "contrib" / "scenarios" / target_scenario).resolve()
    if not scenario_root.is_dir():
        return {
            "ok": False,
            "error": f"scenario root missing: {scenario_root}",
            "resolved_path": None,
        }

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / target_scenario
        shutil.copytree(scenario_root, scratch)
        (scratch / "manifest.yaml").write_text(new_content, encoding="utf-8")
        try:
            load_scenario(str(scratch))
        except ScenarioLoadError as exc:
            return {
                "ok": False,
                "error": f"scenario loader rejected new manifest: {exc}",
                "resolved_path": None,
            }
        except Exception as exc:  # noqa: BLE001 - loader may raise broadly
            return {
                "ok": False,
                "error": f"scenario loader raised: {exc}",
                "resolved_path": None,
            }

    return {"ok": True, "error": None, "resolved_path": str(resolved)}


def _resolve_under_scenario(
    cwd: Path, target_scenario: str, path_value: str
) -> Path | None:
    raw = Path(path_value)
    scenario_root = (cwd / "contrib" / "scenarios" / target_scenario).resolve()
    candidate = raw.resolve() if raw.is_absolute() else (scenario_root / raw).resolve()
    try:
        candidate.relative_to(scenario_root)
    except ValueError:
        return None
    return candidate
