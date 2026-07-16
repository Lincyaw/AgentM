"""Materialize immutable, content-addressed atom snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from agentm.core._internal.catalog.browse import CatalogCorruptionError
from agentm.core._internal.catalog.hashing import compute_atom_hash
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.lib.paths import expand_path
from agentm.core.runtime.catalog import _layout


def freeze_current(
    name: str,
    source: str,
    manifest: ExtensionManifest,
    *,
    root: str | Path | None = None,
) -> str:
    """Freeze one atom and move its ``current`` pointer to that snapshot.

    Version directories are immutable. Re-freezing identical source is
    idempotent; any byte mismatch at an existing content hash is treated as
    catalog corruption and fails loudly.
    """

    if manifest.name != name:
        raise ValueError(
            f"manifest.name {manifest.name!r} does not match atom name {name!r}"
        )
    _validate_atom_name(name)
    cwd_root = expand_path(root).resolve() if root is not None else Path.cwd().resolve()
    version = compute_atom_hash(source)
    snapshot = _manifest_snapshot(manifest, content_hash=version)

    atom_dir = _layout.atoms_dir(name, root=cwd_root)
    version_dir = _layout.atom_version_dir(name, version, root=cwd_root)
    _reject_symlink_components(atom_dir, root=cwd_root)
    _reject_symlink_components(version_dir, root=cwd_root)
    version_dir.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        _layout.atom_source_path(name, version, root=cwd_root),
        source.encode("utf-8"),
    )
    _write_immutable(
        _layout.atom_manifest_path(name, version, root=cwd_root),
        _canonical_json(snapshot),
    )
    runs_dir = _layout.atom_runs_dir(name, version, root=cwd_root)
    _reject_symlink_components(runs_dir, root=cwd_root)
    runs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    _write_atomic(
        _layout.atom_current_path(name, root=cwd_root),
        f"{version}\n".encode(),
    )
    return version


def _manifest_snapshot(
    manifest: ExtensionManifest,
    *,
    content_hash: str,
) -> dict[str, object]:
    schema = manifest.config_schema
    schema_name = None if schema is None else schema.__qualname__
    effects = {
        channel: effect.model_dump(mode="json")
        for channel, effect in sorted(manifest.effects.items())
    }
    return {
        "name": manifest.name,
        "description": manifest.description,
        "registers": list(manifest.registers),
        "config_schema": schema_name,
        "requires": list(manifest.requires),
        "conflicts": list(manifest.conflicts),
        "api_version": manifest.api_version,
        "affects": list(manifest.affects),
        "tier": manifest.tier,
        "mountable_via_command": manifest.mountable_via_command,
        "provides_role": list(manifest.provides_role),
        "effects": effects,
        "content_hash": content_hash,
    }


def _canonical_json(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CatalogCorruptionError(
            f"catalog snapshot file must not be a symlink: {path}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink():
                raise CatalogCorruptionError(
                    f"catalog snapshot file must not be a symlink: {path}"
                )
            existing = path.read_bytes()
            if existing != content:
                raise CatalogCorruptionError(
                    f"immutable catalog snapshot mismatch at {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_atom_name(name: str) -> None:
    if not name.isidentifier():
        raise ValueError(f"invalid atom name {name!r}")


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


__all__ = ["CatalogCorruptionError", "freeze_current"]
