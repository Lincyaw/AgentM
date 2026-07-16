"""Harness-side catalog operations: filesystem + discovery orchestration.

Pure kernel functions (hashing and validated snapshot browsing) live in
``agentm.core._internal.catalog`` and remain available there. This package
collects snapshot materialization, indexing, and project-layout policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from agentm.core.abi.project_layout import ProjectLayout
from agentm.core.lib import expand_path
from agentm.core.runtime.catalog.freeze import freeze_current
from agentm.core.runtime.catalog.indexer import (
    IndexerResult,
    index_trace,
    rebuild_catalog,
)
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
    "rebuild_catalog",
]
