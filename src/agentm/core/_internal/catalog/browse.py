"""Read-only access to immutable, content-addressed catalog snapshots."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from agentm.core._internal.catalog.hashing import compute_atom_hash
from agentm.core.abi.catalog import ActiveSetFingerprint, ManifestSnapshot

_VERSION_RE = re.compile(r"[0-9a-f]{12}")
_SOURCE_FILENAME = "source.py"
_MANIFEST_FILENAME = "manifest.json"
_CURRENT_FILENAME = "current"


class CatalogCorruptionError(RuntimeError):
    """The catalog contradicts the content identity encoded by its path."""


def list_versions(name: str, root: Path | None = None) -> list[str]:
    atom_dir = _atom_dir(name, root)
    if not atom_dir.exists():
        return []
    versions: list[str] = []
    for child in sorted(atom_dir.iterdir(), key=lambda path: path.name):
        if child.name == _CURRENT_FILENAME:
            continue
        if child.is_symlink():
            raise CatalogCorruptionError(
                f"catalog atom entry must not be a symlink: {child}"
            )
        if not child.is_dir():
            raise CatalogCorruptionError(
                f"unexpected non-directory catalog atom entry: {child}"
            )
        _validate_version(child.name)
        _read_snapshot(name, child.name, root)
        versions.append(child.name)
    return versions


def current_version(name: str, root: Path | None = None) -> str:
    pointer = _atom_dir(name, root) / _CURRENT_FILENAME
    if pointer.is_symlink():
        raise CatalogCorruptionError(
            f"catalog current pointer must not be a symlink: {pointer}"
        )
    try:
        version = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise KeyError(f"No current catalog version for atom {name!r}") from exc
    _validate_version(version)
    _read_snapshot(name, version, root)
    return version


def get_source_at(name: str, version: str, root: Path | None = None) -> bytes:
    source, _manifest = _read_snapshot(name, version, root)
    return source


def get_manifest_at(
    name: str,
    version: str,
    root: Path | None = None,
) -> ManifestSnapshot:
    _source, manifest = _read_snapshot(name, version, root)
    return manifest


def runs_for(
    fingerprint: ActiveSetFingerprint | str,
    root: Path | None = None,
) -> list[str]:
    refs = _normalize_fingerprint(fingerprint)
    if not refs:
        return []

    cwd_root = _root_path(root)
    trace_sets: list[set[str]] = []
    for name, version in refs.items():
        _read_snapshot(name, version, root)
        runs_dir = cwd_root / ".agentm" / "catalog" / "atoms" / name / version / "runs"
        if runs_dir.is_symlink():
            raise CatalogCorruptionError(
                f"catalog runs directory must not be a symlink: {runs_dir}"
            )
        if not runs_dir.exists():
            return []
        trace_sets.append({entry.name for entry in runs_dir.iterdir()})
    if not trace_sets:
        return []
    return sorted(set.intersection(*trace_sets))


def _read_snapshot(
    name: str,
    version: str,
    root: Path | None,
) -> tuple[bytes, ManifestSnapshot]:
    _validate_atom_name(name)
    _validate_version(version)
    version_dir = _atom_dir(name, root) / version
    source_path = version_dir / _SOURCE_FILENAME
    manifest_path = version_dir / _MANIFEST_FILENAME
    if version_dir.is_symlink():
        raise CatalogCorruptionError(
            f"catalog version directory must not be a symlink: {version_dir}"
        )
    for snapshot_path in (source_path, manifest_path):
        if snapshot_path.is_symlink():
            raise CatalogCorruptionError(
                f"catalog snapshot file must not be a symlink: {snapshot_path}"
            )
    try:
        source = source_path.read_bytes()
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeyError(f"Missing catalog snapshot {name}@{version}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogCorruptionError(
            f"invalid manifest JSON for {name}@{version}: {exc}"
        ) from exc

    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogCorruptionError(
            f"atom source is not UTF-8 for {name}@{version}"
        ) from exc
    actual_hash = compute_atom_hash(source_text)
    if actual_hash != version:
        raise CatalogCorruptionError(
            f"source hash mismatch for {name}@{version}: computed {actual_hash}"
        )
    if not isinstance(raw_manifest, dict):
        raise CatalogCorruptionError(
            f"manifest for {name}@{version} must be a JSON object"
        )
    manifest = cast(ManifestSnapshot, raw_manifest)
    if manifest.get("name") != name:
        raise CatalogCorruptionError(f"manifest name mismatch for {name}@{version}")
    if manifest.get("content_hash") != version:
        raise CatalogCorruptionError(
            f"manifest content_hash mismatch for {name}@{version}"
        )
    return source, manifest


def _normalize_fingerprint(
    fingerprint: ActiveSetFingerprint | str,
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
        if value is None:
            continue
        atom, version = _split_atom_ref(str(value), str(key))
        refs[atom] = version
    return refs


def _split_atom_ref(value: str, fallback_atom: str | None) -> tuple[str, str]:
    atom, separator, version = value.partition("@")
    if separator:
        if not atom or not version:
            raise ValueError(f"Invalid atom reference: {value!r}")
    elif fallback_atom is not None:
        atom, version = fallback_atom, value
    else:
        raise ValueError(f"Expected 'atom@version', got {value!r}")
    _validate_atom_name(atom)
    _validate_version(version)
    return atom, version


def _root_path(root: Path | None) -> Path:
    return (root or Path.cwd()).resolve()


def _atom_dir(name: str, root: Path | None) -> Path:
    _validate_atom_name(name)
    cwd_root = _root_path(root)
    atom_dir = cwd_root / ".agentm" / "catalog" / "atoms" / name
    _reject_symlink_components(atom_dir, root=cwd_root)
    return atom_dir


def _validate_atom_name(name: str) -> None:
    if not name.isidentifier():
        raise ValueError(f"invalid atom name {name!r}")


def _validate_version(version: str) -> None:
    if _VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"invalid content hash {version!r}")


def _reject_symlink_components(path: Path, *, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"catalog path {path} is outside root {root}") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise CatalogCorruptionError(
                f"catalog path component must not be a symlink: {current}"
            )


__all__ = [
    "CatalogCorruptionError",
    "current_version",
    "get_manifest_at",
    "get_source_at",
    "list_versions",
    "runs_for",
]
