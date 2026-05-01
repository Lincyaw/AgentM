"""Catalog path contract for the constitution-layer storage tree."""

from __future__ import annotations

from pathlib import Path

CATALOG_ROOT = Path(".agentm/catalog")


def catalog_root(*, root: Path | None = None) -> Path:
    return (root or Path.cwd()) / CATALOG_ROOT


def atoms_dir(
    name: str | None = None, *, root: Path | None = None
) -> Path:
    base = catalog_root(root=root) / "atoms"
    if name is None:
        return base
    return base / name


def atom_version_dir(
    name: str, content_hash: str, *, root: Path | None = None
) -> Path:
    return atoms_dir(name, root=root) / content_hash


def atom_runs_dir(
    name: str, content_hash: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, content_hash, root=root) / "runs"


def atom_metrics_path(
    name: str, content_hash: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, content_hash, root=root) / "metrics.jsonl"


def atom_manifest_path(
    name: str, content_hash: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, content_hash, root=root) / "manifest.yaml"


def atom_source_path(
    name: str, content_hash: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, content_hash, root=root) / "source.py"


def atom_current_symlink(name: str, *, root: Path | None = None) -> Path:
    return atoms_dir(name, root=root) / "current"



def core_dir(content_hash: str, *, root: Path | None = None) -> Path:
    return catalog_root(root=root) / "core" / content_hash
