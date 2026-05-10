"""Validator for ``ChangeSpec.kind == 'system_prompt'``.

A ``system_prompt`` change rewrites a prompt file referenced from a
scenario's manifest (typically a ``.md`` under ``contrib/scenarios/<name>/``).
The validator checks the target file lives under the scenario root, the
new content is non-empty UTF-8 within a soft size cap, and the path
extension is one of the prompt-friendly suffixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Soft cap to keep a single prompt mutation from dominating the manifest;
# the GEPA paper's prompt mutations stay well under this.
_MAX_BYTES = 64_000

_ALLOWED_SUFFIXES = (".md", ".txt", ".yaml", ".yml")


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
                "kind='system_prompt'"
            ),
            "resolved_path": None,
        }
    if not isinstance(new_content, str) or not new_content:
        return {
            "ok": False,
            "error": (
                "evidence missing: target.new_content must be a "
                "non-empty string for kind='system_prompt'"
            ),
            "resolved_path": None,
        }
    encoded = new_content.encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        return {
            "ok": False,
            "error": (
                f"target.new_content exceeds {_MAX_BYTES} bytes "
                f"(got {len(encoded)})"
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
    if resolved.suffix.lower() not in _ALLOWED_SUFFIXES:
        return {
            "ok": False,
            "error": (
                f"target.path suffix {resolved.suffix!r} not in "
                f"{list(_ALLOWED_SUFFIXES)} for kind='system_prompt'"
            ),
            "resolved_path": None,
        }
    return {"ok": True, "error": None, "resolved_path": str(resolved)}


def _resolve_under_scenario(
    cwd: Path, target_scenario: str, path_value: str
) -> Path | None:
    """Resolve ``path_value`` (scenario-relative or absolute) and confirm
    it stays under the scenario root. Returns the absolute path or None.
    """
    raw = Path(path_value)
    scenario_root = (cwd / "contrib" / "scenarios" / target_scenario).resolve()
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (scenario_root / raw).resolve()
    try:
        candidate.relative_to(scenario_root)
    except ValueError:
        return None
    return candidate
