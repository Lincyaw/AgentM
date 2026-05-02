"""Read-only helpers for browsing the on-disk catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core._internal.catalog import _layout


@dataclass(frozen=True, slots=True)
class CatalogAtom:
    name: str
    versions: tuple[str, ...]


def list_versions(name: str, root: Path | None = None) -> list[str]:
    atom_root = _layout.atoms_dir(name, root=root)
    if not atom_root.exists():
        return []
    return sorted(
        entry.name
        for entry in atom_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(_layout.LEGACY_PREFIX)
    )


def get_manifest_at(
    name: str, version: str, root: Path | None = None
) -> dict[str, Any]:
    raise RuntimeError(
        f"manifest browsing for {name}@{version} moved to git-backed plumbing and is handled in issue #3"
    )


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
