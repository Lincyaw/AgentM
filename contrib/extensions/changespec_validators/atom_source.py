"""Validator for ``ChangeSpec.kind == 'atom_source'``.

Atom-source changes were the MVP path; this module re-expresses the
existing per-call checks as a validator so the dispatch in
``tool_propose_change`` is uniform across kinds. The atom-resolution
(filesystem scan that finds a MANIFEST whose ``name`` matches
``target_atom``) stays in ``tool_propose_change`` itself because it
needs ``ExtensionAPI`` access; this validator only enforces the static
shape contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate(
    change_spec: dict[str, Any], cwd: Path, target_scenario: str
) -> dict[str, Any]:
    path_value = change_spec.get("path")
    new_content = change_spec.get("new_content")
    target_atom = change_spec.get("target_atom")
    if not isinstance(path_value, str) or not path_value:
        return {
            "ok": False,
            "error": (
                "evidence missing: target.path is required for "
                "kind='atom_source'"
            ),
            "resolved_path": None,
        }
    if not isinstance(new_content, str) or not new_content:
        return {
            "ok": False,
            "error": (
                "evidence missing: target.new_content is required and "
                "must be a non-empty string"
            ),
            "resolved_path": None,
        }
    if target_atom is None or (
        isinstance(target_atom, str) and not target_atom
    ):
        return {
            "ok": False,
            "error": (
                "evidence missing: target.target_atom is required for "
                "kind='atom_source'"
            ),
            "resolved_path": None,
        }
    if not isinstance(target_atom, str):
        return {
            "ok": False,
            "error": (
                "target.target_atom must be a string (atom name) for "
                "kind='atom_source'"
            ),
            "resolved_path": None,
        }
    # Resolution to a concrete path is done by the caller via the
    # existing _find_atom_* helpers — they need ExtensionAPI access.
    return {"ok": True, "error": None, "resolved_path": None}
