"""Public constitution-layer catalog surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core._internal.catalog import _layout
from agentm.core._internal.catalog.browse import (
    CatalogAtom,
    UnparseableManifestError,
    current_version,
    get_manifest_at,
    get_source_at,
    list_versions,
    runs_for,
)
from agentm.core._internal.catalog.hashing import (
    compute_active_set_fingerprint,
    compute_atom_hash,
)
from agentm.core._internal.catalog.manifest import is_constitution_path
from agentm.extensions.discover import discover_builtin


def list_atoms(*, root: Path | None = None) -> list[dict[str, Any]]:
    atoms_root = _layout.atoms_dir(root=root)
    if not atoms_root.exists():
        return []

    builtin = discover_builtin()
    atoms: list[dict[str, Any]] = []
    for atom_dir in sorted(path for path in atoms_root.iterdir() if path.is_dir()):
        versions = [
            child.name
            for child in atom_dir.iterdir()
            if child.is_dir() and not child.name.startswith(_layout.LEGACY_PREFIX)
        ]
        if not versions:
            continue
        manifest = builtin.get(atom_dir.name)
        atoms.append(
            {
                "name": atom_dir.name,
                "current_hash": sorted(versions)[-1],
                "tier": None if manifest is None else manifest.manifest.tier,
                "api_version": (
                    None if manifest is None else manifest.manifest.api_version
                ),
            }
        )
    return atoms


def freeze_current(
    name: str,
    source: str,
    manifest: Any,
    *,
    root: Path | None = None,
) -> str:
    from agentm.core._internal.catalog.freeze import (
        freeze_current as _freeze_current,
    )

    return _freeze_current(name, source, manifest, root=root)


def source_path_for_hash(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    from agentm.core._internal.catalog.freeze import (
        source_path_for_hash as _source_path_for_hash,
    )

    return _source_path_for_hash(name, version_key, root=root)


__all__ = [
    "CatalogAtom",
    "UnparseableManifestError",
    "compute_active_set_fingerprint",
    "compute_atom_hash",
    "current_version",
    "freeze_current",
    "get_manifest_at",
    "get_source_at",
    "is_constitution_path",
    "list_atoms",
    "list_versions",
    "runs_for",
    "source_path_for_hash",
]
