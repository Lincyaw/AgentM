"""Read-only helpers for browsing the on-disk catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentm.core._internal.catalog import _layout


@dataclass(frozen=True, slots=True)
class CatalogAtom:
    name: str
    versions: tuple[str, ...]


def list_versions(name: str, root: Path | None = None) -> list[str]:
    atom_root = _layout.atoms_dir(name, root=root)
    if not atom_root.exists():
        return []
    versions = {
        entry.name
        for entry in atom_root.iterdir()
        if entry.name != "current" and entry.is_dir()
    }
    current = _read_current_version(name, root=root)
    if current:
        versions.add(current)
    return sorted(versions)


def get_manifest_at(
    name: str, version: str, root: Path | None = None
) -> dict[str, Any]:
    manifest_path = _layout.atom_manifest_path(name, version, root=root)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"Manifest at {manifest_path} must decode to a mapping, got {type(payload).__name__}"
        )
    return payload


def runs_for(fingerprint: dict[str, Any] | str, root: Path | None = None) -> list[str]:
    refs = _normalize_fingerprint(fingerprint)
    if not refs:
        return []

    trace_sets: list[set[str]] = []
    for name, version in refs.items():
        runs_dir = _layout.atom_runs_dir(name, version, root=root)
        if not runs_dir.exists():
            return []
        trace_sets.append({entry.name for entry in runs_dir.iterdir()})
    if not trace_sets:
        return []
    trace_ids = set.intersection(*trace_sets)
    return sorted(trace_ids)


def _normalize_fingerprint(
    fingerprint: dict[str, Any] | str,
) -> dict[str, str]:
    if isinstance(fingerprint, str):
        atom, version = _split_atom_ref(fingerprint, None)
        return {atom: version}
    if not isinstance(fingerprint, dict):
        raise TypeError("fingerprint must be a dict or 'atom@version' string")

    raw_atoms = fingerprint.get("atoms")
    atom_map = raw_atoms if isinstance(raw_atoms, dict) else fingerprint
    refs: dict[str, str] = {}
    for key, value in atom_map.items():
        atom, version = _split_atom_ref(str(value), str(key))
        refs[atom] = version
    return refs


def _split_atom_ref(value: str, fallback_atom: str | None) -> tuple[str, str]:
    atom, separator, version = value.partition("@")
    if separator:
        if not atom or not version:
            raise ValueError(f"Invalid atom reference: {value!r}")
        return atom, version
    if fallback_atom is None:
        raise ValueError(f"Expected 'atom@version', got {value!r}")
    return fallback_atom, value


def _read_current_version(name: str, root: Path | None = None) -> str | None:
    current_path = _layout.atom_current_symlink(name, root=root)
    if current_path.is_symlink():
        return Path(os.readlink(current_path)).name
    if current_path.exists():
        text = current_path.read_text(encoding="utf-8").strip()
        return text or None
    return None
