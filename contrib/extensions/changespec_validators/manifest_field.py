"""Validator for ``ChangeSpec.kind == 'manifest_field'``.

A ``manifest_field`` change rewrites a single scalar field inside a
scenario's ``manifest.yaml`` (or another scalar-bearing YAML under the
scenario tree). The change-spec semantics for this kind:

- ``path``  : YAML file under ``contrib/scenarios/<name>/`` (relative or absolute).
- ``new_content`` : the YAML-dotted-path of the field to set, plus its new
  scalar value, joined by ``=``. Example: ``promotion.threshold_relative=0.07``.
  We pick this string-encoded form (rather than two parameters) so the
  ChangeSpec dataclass stays uniform across kinds; B-3's design note in
  the task file flags that more structured payloads are a Phase-3 lift.

The validator confirms:

- The YAML file exists under the scenario root.
- ``new_content`` parses as ``key.path=value``.
- ``key.path`` resolves to an existing scalar in the YAML.
- The new scalar parses as the same type as the existing one.
"""

from __future__ import annotations

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
                "kind='manifest_field'"
            ),
            "resolved_path": None,
        }
    if not isinstance(new_content, str) or not new_content:
        return {
            "ok": False,
            "error": (
                "evidence missing: target.new_content must be a non-empty "
                "string of the form 'dotted.path=value'"
            ),
            "resolved_path": None,
        }
    if "=" not in new_content:
        return {
            "ok": False,
            "error": (
                "target.new_content for kind='manifest_field' must be "
                "'dotted.path=value'; missing '='"
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
    if not resolved.is_file():
        return {
            "ok": False,
            "error": f"manifest file not found: {resolved}",
            "resolved_path": None,
        }

    dotted, _, raw_value = new_content.partition("=")
    dotted = dotted.strip()
    raw_value = raw_value.strip()
    if not dotted:
        return {
            "ok": False,
            "error": "dotted.path is empty in target.new_content",
            "resolved_path": None,
        }

    try:
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            "ok": False,
            "error": f"manifest file is not valid YAML: {exc}",
            "resolved_path": None,
        }
    if not isinstance(loaded, dict):
        return {
            "ok": False,
            "error": "manifest file root must be a mapping",
            "resolved_path": None,
        }

    existing = _walk_dotted(loaded, dotted.split("."))
    if existing is _MISSING:
        return {
            "ok": False,
            "error": (
                f"manifest_field target {dotted!r} not found in "
                f"{resolved.name}"
            ),
            "resolved_path": None,
        }
    if not _is_scalar(existing):
        return {
            "ok": False,
            "error": (
                f"manifest_field target {dotted!r} is not a scalar "
                f"(got {type(existing).__name__})"
            ),
            "resolved_path": None,
        }

    new_scalar = _coerce_like(existing, raw_value)
    if new_scalar is _COERCE_FAIL:
        return {
            "ok": False,
            "error": (
                f"new value {raw_value!r} could not be coerced to the "
                f"existing scalar type ({type(existing).__name__})"
            ),
            "resolved_path": None,
        }
    return {"ok": True, "error": None, "resolved_path": str(resolved)}


_MISSING = object()
_COERCE_FAIL = object()


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


def _walk_dotted(root: Any, parts: list[str]) -> Any:
    cur: Any = root
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _coerce_like(existing: Any, raw: str) -> Any:
    """Coerce ``raw`` to the same scalar type as ``existing``. Returns
    ``_COERCE_FAIL`` on type mismatch."""
    if existing is None:
        # Allow setting anything when the existing value is null.
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError:
            return _COERCE_FAIL
    if isinstance(existing, bool):
        low = raw.lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        return _COERCE_FAIL
    if isinstance(existing, int) and not isinstance(existing, bool):
        try:
            return int(raw)
        except ValueError:
            return _COERCE_FAIL
    if isinstance(existing, float):
        try:
            return float(raw)
        except ValueError:
            return _COERCE_FAIL
    if isinstance(existing, str):
        return raw
    return _COERCE_FAIL
