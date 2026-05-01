"""Freeze the currently loaded atom into the on-disk catalog."""

from __future__ import annotations

import os
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agentm.core.catalog import _layout
from agentm.core.catalog.hashing import compute_atom_hash
from agentm.extensions import ExtensionManifest



def freeze_current(
    name: str,
    source: str,
    manifest: ExtensionManifest,
    *,
    root: Path | None = None,
) -> str:
    if manifest.name != name:
        raise ValueError(
            f"manifest.name {manifest.name!r} does not match atom name {name!r}"
        )

    content_hash = compute_atom_hash(source)
    atom_dir = _layout.atoms_dir(name, root=root)
    version_dir = _layout.atom_version_dir(name, content_hash, root=root)
    source_path = _layout.atom_source_path(name, content_hash, root=root)
    manifest_path = _layout.atom_manifest_path(name, content_hash, root=root)
    runs_dir = _layout.atom_runs_dir(name, content_hash, root=root)

    atom_dir.mkdir(parents=True, exist_ok=True)

    if source_path.exists() and manifest_path.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)
        _replace_current_pointer(name, content_hash, root=root)
        return content_hash

    version_dir.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    manifest_path.write_text(
        yaml.safe_dump(
            _manifest_payload(content_hash, manifest),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    _replace_current_pointer(name, content_hash, root=root)
    return content_hash



def _manifest_payload(
    content_hash: str, manifest: ExtensionManifest
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content_hash": content_hash,
        "parent_hash": None,
        "author": "human",
        "authored_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    for field in fields(manifest):
        value = getattr(manifest, field.name)
        payload[field.name] = list(value) if isinstance(value, tuple) else value
    return payload



def _replace_current_pointer(
    name: str, content_hash: str, *, root: Path | None = None
) -> None:
    current_path = _layout.atom_current_symlink(name, root=root)
    temp_path = current_path.with_name(f"{current_path.name}.tmp")
    _remove_path(temp_path)
    try:
        os.symlink(content_hash, temp_path)
    except (NotImplementedError, OSError):
        temp_path.write_text(content_hash + "\n", encoding="utf-8")
    os.replace(temp_path, current_path)



def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
