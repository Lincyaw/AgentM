"""Harness-side catalog operations: filesystem + discovery orchestration.

Pure kernel functions (hashing, manifest parsing, browse) live in
``agentm.core._internal.catalog`` and remain available there. This package
collects everything that touches the filesystem, walks discovery, or
otherwise enacts project-layout policy on top of the constitution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from agentm.core.abi.project_layout import ProjectLayout
from agentm.core.lib import expand_path
from agentm.core.runtime.catalog import _layout
from agentm.core.runtime.catalog.freeze import freeze_current
from agentm.core.runtime.catalog.indexer import (
    IndexerResult,
    index_trace,
    rebuild_catalog,
)
from agentm.core.runtime.catalog.migrate import migrate_catalog_v2


class CatalogAtomRow(TypedDict):
    name: str
    current_hash: str
    tier: int | None
    api_version: int | None


def list_atoms(*, root: Path | None = None) -> list[CatalogAtomRow]:
    """List atoms currently materialized under the catalog root.

    Walks ``<root>/.agentm/catalog/atoms/`` and consults
    ``discover_builtin`` for tier/api_version metadata.
    """

    # Imported lazily so importing this package never walks the extensions
    # tree at import time.
    from agentm.extensions.discover import discover_builtin

    atoms_root = _layout.atoms_dir(root=root)
    if not atoms_root.exists():
        return []

    builtin = discover_builtin()
    atoms: list[CatalogAtomRow] = []
    for atom_dir in sorted(path for path in atoms_root.iterdir() if path.is_dir()):
        versions = [
            child.name
            for child in atom_dir.iterdir()
            if child.is_dir() and not child.name.startswith(_layout.LEGACY_PREFIX)
        ]
        if not versions:
            continue
        manifest = builtin.get(atom_dir.name)
        atoms.append(
            {
                "name": atom_dir.name,
                "current_hash": sorted(versions)[-1],
                "tier": None if manifest is None else manifest.manifest.tier,
                "api_version": (
                    None if manifest is None else manifest.manifest.api_version
                ),
            }
        )
    return atoms


@dataclass(frozen=True, slots=True)
class DefaultProjectLayout:
    """Default :class:`ProjectLayout` for workspace-scoped state.

    Catalog, project skills, prompt templates, and shared artifacts are rooted
    under ``<cwd>/.agentm/...``. Observability is deliberately user-scoped via
    ``resolve_observability_dir`` (normally ``$AGENTM_HOME/observability``) so
    ordinary session traces do not get written into the source checkout.
    Constructing the layout does not touch the filesystem; callers ``mkdir``
    lazily as they write.
    """

    cwd: Path

    def catalog_root(self) -> Path:
        return self.cwd / ".agentm" / "catalog"

    def skills_dirs(self) -> list[Path]:
        return [self.cwd / ".agentm" / "skills"]

    def artifacts_root(self, session_id: str) -> Path:
        return self.cwd / ".agentm" / "artifacts" / session_id

    def prompts_dirs(self) -> list[Path]:
        return [self.cwd / ".agentm" / "prompts"]

    def observability_root(self) -> Path:
        from agentm.core.lib.observability_dir import resolve_observability_dir

        return resolve_observability_dir(self.cwd)


def default_project_layout(cwd: str | Path) -> ProjectLayout:
    """Build the default :class:`ProjectLayout` for a given workspace."""

    return DefaultProjectLayout(cwd=expand_path(cwd).resolve())


__all__ = [
    "DefaultProjectLayout",
    "IndexerResult",
    "ProjectLayout",
    "default_project_layout",
    "freeze_current",
    "index_trace",
    "list_atoms",
    "migrate_catalog_v2",
    "rebuild_catalog",
]
