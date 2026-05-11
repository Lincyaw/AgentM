"""Default ``CatalogService`` impl + helpers.

The atom-facing :class:`CatalogService` Protocol lives in
:mod:`agentm.core.abi.catalog`. This module holds only the default
implementation (a thin pass-through over ``agentm.core._internal.catalog``)
and the convenience builders used by ``ExtensionAPI`` wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.abi.catalog import CatalogService
from agentm.core.abi.project_layout import ProjectLayout


class _DefaultCatalogService:
    def list_versions(
        self, name: str, root: Path | None = None
    ) -> list[str]:
        from agentm.core._internal.catalog import list_versions as _impl

        return _impl(name, root)

    def current_version(
        self, path: str, root: Path | None = None
    ) -> str:
        from agentm.core._internal.catalog import current_version as _impl

        return _impl(path, root)

    def get_source_at(
        self, path: str, version: str, root: Path | None = None
    ) -> bytes:
        from agentm.core._internal.catalog import get_source_at as _impl

        return _impl(path, version, root)

    def get_manifest_at(
        self, name: str, version: str, root: Path | None = None
    ) -> dict[str, Any]:
        from agentm.core._internal.catalog import get_manifest_at as _impl

        return _impl(name, version, root)

    def runs_for(
        self,
        fingerprint: dict[str, Any] | str,
        root: Path | None = None,
    ) -> list[str]:
        from agentm.core._internal.catalog import runs_for as _impl

        return _impl(fingerprint, root)

    def compute_atom_hash(self, source: str) -> str:
        from agentm.core._internal.catalog import compute_atom_hash as _impl

        return _impl(source)

    def compute_active_set_fingerprint(
        self,
        loaded: dict[str, str],
        scenario: str | None,
        core_hash: str | None,
    ) -> dict[str, Any]:
        from agentm.core._internal.catalog import (
            compute_active_set_fingerprint as _impl,
        )

        return _impl(loaded, scenario, core_hash)


# --- Default builders ------------------------------------------------------


def default_catalog_service() -> CatalogService:
    return _DefaultCatalogService()


def default_project_layout(cwd: str) -> ProjectLayout:
    """Return the harness's default :class:`ProjectLayout` for ``cwd``."""

    from agentm.core.runtime.catalog import default_project_layout as _impl

    return _impl(cwd)


__all__ = [
    "CatalogService",
    "ProjectLayout",
    "default_catalog_service",
    "default_project_layout",
]
