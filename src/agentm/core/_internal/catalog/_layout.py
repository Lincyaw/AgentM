"""Catalog path contract for the constitution-layer storage tree."""

from __future__ import annotations

from pathlib import Path

CATALOG_ROOT = Path(".agentm/catalog")
LEGACY_PREFIX = ".legacy-"
METRICS_FILENAME = "metrics.jsonl"
RUNS_DIRNAME = "runs"


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


def atom_metrics_path(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    return atom_version_dir(name, version_key, root=root) / METRICS_FILENAME


# --- Indexer helpers -----------------------------------------------------
# The indexer treats its ``root`` argument as the catalog root itself
# (``<cwd>/.agentm/catalog``) rather than the cwd. The helpers below provide
# that view onto the same on-disk tree so other catalog modules can share the
# same ``_layout`` module while keeping their respective conventions.


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
    by the freeze/indexer-flavoured helpers (``catalog_root(root=cwd)``)."""

    catalog_root_path = Path(catalog_root_path)
    return catalog_root_path.parent.parent
