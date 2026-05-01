"""Public constitution-layer catalog surface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from agentm.core.catalog import _layout
from agentm.core.catalog.browse import (
    CatalogAtom,
    get_manifest_at,
    list_versions,
    runs_for,
)
from agentm.core.catalog.freeze import freeze_current
from agentm.core.catalog.hashing import (
    compute_active_set_fingerprint,
    compute_atom_hash,
)
from agentm.core.catalog.manifest import is_constitution_path



def list_atoms(*, root: Path | None = None) -> list[dict[str, Any]]:
    atoms_root = _layout.atoms_dir(root=root)
    if not atoms_root.exists():
        return []

    atoms: list[dict[str, Any]] = []
    for atom_dir in sorted(path for path in atoms_root.iterdir() if path.is_dir()):
        current_hash = _read_current_hash(atom_dir.name, root=root)
        if current_hash is None:
            continue

        manifest_path = _layout.atom_manifest_path(
            atom_dir.name, current_hash, root=root
        )
        manifest = _load_manifest(manifest_path)
        atoms.append(
            {
                "name": atom_dir.name,
                "current_hash": current_hash,
                "tier": manifest.get("tier"),
                "api_version": manifest.get("api_version"),
            }
        )
    return atoms



def _read_current_hash(name: str, *, root: Path | None = None) -> str | None:
    current_path = _layout.atom_current_symlink(name, root=root)
    if current_path.is_symlink():
        return Path(os.readlink(current_path)).name
    if current_path.exists():
        content_hash = current_path.read_text(encoding="utf-8").strip()
        return content_hash or None
    return None



def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"catalog manifest at {path} must deserialize to a dict")
    return data


__all__ = [
    "CatalogAtom",
    "compute_active_set_fingerprint",
    "compute_atom_hash",
    "freeze_current",
    "get_manifest_at",
    "is_constitution_path",
    "list_atoms",
    "list_versions",
    "runs_for",
]
