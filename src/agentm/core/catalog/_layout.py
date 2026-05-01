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


# --- Indexer helpers (PR #44) -------------------------------------------------
#
# The indexer treats its ``root`` argument as the catalog root itself
# (``<cwd>/.agentm/catalog``) rather than the cwd. The helpers below provide
# that view onto the same on-disk tree so freeze.py / indexer.py can share the
# same ``_layout`` module while keeping their respective conventions.

METRICS_FILENAME = "metrics.jsonl"
RUNS_DIRNAME = "runs"


def resolve_root(root: Path) -> Path:
    """Return the catalog root as an absolute Path, creating it if needed."""

    resolved = Path(root).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atoms_root(root: Path) -> Path:
    """Return the atoms directory under a catalog root (not a cwd)."""

    return Path(root) / "atoms"


def _from_catalog_root(catalog_root_path: Path) -> Path:
    """Translate a catalog-root path back into the cwd-style ``root`` accepted
    by the freeze.py-flavoured helpers (``catalog_root(root=cwd)``)."""

    catalog_root_path = Path(catalog_root_path)
    # CATALOG_ROOT is ``.agentm/catalog`` (2 components); strip them off.
    return catalog_root_path.parent.parent
