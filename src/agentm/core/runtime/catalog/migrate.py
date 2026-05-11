"""One-shot migration helpers for git-backed catalog versioning."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from agentm.core.runtime.catalog import _layout

logger = logging.getLogger(__name__)

_MIGRATION_MARKER = ".migration-v2"
_LEGACY_FILES = {"source.py", "manifest.yaml"}


def migrate_catalog_v2(*, root: Path | None = None) -> bool:
    cwd_root = (root or Path.cwd()).resolve()
    catalog_root = _layout.catalog_root(root=cwd_root)
    marker = catalog_root / _MIGRATION_MARKER
    if marker.exists():
        return False

    atoms_root = _layout.atoms_dir(root=cwd_root)
    if atoms_root.exists():
        for atom_dir in atoms_root.iterdir():
            if not atom_dir.is_dir():
                continue
            for version_dir in list(atom_dir.iterdir()):
                if not version_dir.is_dir() or version_dir.name.startswith(_layout.LEGACY_PREFIX):
                    continue
                legacy = _migrate_legacy_version_dir(version_dir)
                if legacy:
                    logger.info("catalog migration marked legacy version dir %s", legacy)

    catalog_root.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    return True


def _migrate_legacy_version_dir(version_dir: Path) -> Path | None:
    legacy_files = [version_dir / name for name in _LEGACY_FILES if (version_dir / name).exists()]
    if not legacy_files and _is_git_sha(version_dir.name):
        return None

    for path in legacy_files:
        path.unlink()

    legacy_name = f"{_layout.LEGACY_PREFIX}{version_dir.name}"
    legacy_dir = version_dir.with_name(legacy_name)
    if legacy_dir.exists():
        if version_dir == legacy_dir:
            return legacy_dir
        _merge_dir(version_dir, legacy_dir)
        shutil.rmtree(version_dir)
        return legacy_dir
    version_dir.rename(legacy_dir)
    return legacy_dir


def _merge_dir(source: Path, dest: Path) -> None:
    for child in source.iterdir():
        target = dest / child.name
        if child.is_dir() and target.exists() and target.is_dir():
            _merge_dir(child, target)
            child.rmdir()
        elif not target.exists():
            child.rename(target)


def _is_git_sha(value: str) -> bool:
    return len(value) == 40 and all(ch in "0123456789abcdef" for ch in value.lower())
