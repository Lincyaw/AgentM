"""Catalog path contract for the on-disk storage tree.

Lives in the runtime layer: filesystem-shaped helpers used by freeze /
migrate / indexer. The kernel boundary uses this only via the
:class:`agentm.core.abi.project_layout.ProjectLayout` Protocol.
"""

from __future__ import annotations

from pathlib import Path

CATALOG_ROOT = Path(".agentm/catalog")
CURRENT_FILENAME = "current"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.jsonl"
RUNS_DIRNAME = "runs"
SOURCE_FILENAME = "source.py"


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
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    return atoms_dir(name, root=root) / version_key


def atom_runs_dir(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, version_key, root=root) / RUNS_DIRNAME


def atom_source_path(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, version_key, root=root) / SOURCE_FILENAME


def atom_manifest_path(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, version_key, root=root) / MANIFEST_FILENAME


def atom_current_path(name: str, *, root: Path | None = None) -> Path:
    return atoms_dir(name, root=root) / CURRENT_FILENAME


def atom_metrics_path(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, version_key, root=root) / METRICS_FILENAME


def atom_decisions_path(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    # Single source of truth lives in core.abi.catalog so atoms and runtime
    # agree on the shape without atoms having to reach into ``_layout``.
    from agentm.core.abi.catalog import atom_decisions_path as _abi_impl

    return _abi_impl(name, version_key, root=root)


# --- Indexer helpers -----------------------------------------------------
# The indexer treats its ``root`` argument as the catalog root itself
# (``<cwd>/.agentm/catalog``) rather than the cwd. The helpers below provide
# that view onto the same on-disk tree so other catalog modules can share the
# same ``_layout`` module while keeping their respective conventions.


def atoms_root(root: Path) -> Path:
    """Return the atoms directory under a catalog root (not a cwd)."""

    return Path(root) / "atoms"

