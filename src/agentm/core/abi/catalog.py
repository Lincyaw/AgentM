"""Catalog service Protocol — atom-facing surface.

Atoms call ``api.catalog`` to query versions, fingerprints, and provenance
without reaching into ``agentm.core._internal`` directly. The default
implementation lives in :mod:`agentm.core.runtime.services`; this module
holds only the stable Protocol plus a small set of path-shaped helpers that
atoms need from the catalog layout (currently ``atom_decisions_path``).

Pluggability hard rule: this module imports only stdlib + ABI siblings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CatalogService(Protocol):
    def list_versions(
        self, name: str, root: Path | None = None
    ) -> list[str]: ...

    def current_version(
        self, path: str, root: Path | None = None
    ) -> str: ...

    def get_source_at(
        self, path: str, version: str, root: Path | None = None
    ) -> bytes: ...

    def get_manifest_at(
        self, name: str, version: str, root: Path | None = None
    ) -> dict[str, Any]: ...

    def runs_for(
        self,
        fingerprint: dict[str, Any] | str,
        root: Path | None = None,
    ) -> list[str]: ...

    def compute_atom_hash(self, source: str) -> str: ...

    def compute_active_set_fingerprint(
        self,
        loaded: dict[str, str],
        scenario: str | None,
        core_hash: str | None,
    ) -> dict[str, Any]: ...


# --- Path helpers ----------------------------------------------------------
#
# These mirror :mod:`agentm.core.runtime.catalog._layout` and are exposed
# here because the ``tool_catalog`` atom needs the path shape to read
# decisions logs without reaching the private ``_layout`` module. They are
# pure path-arithmetic functions — no filesystem side effects.


_CATALOG_ROOT = Path(".agentm/catalog")


def atom_decisions_path(
    name: str, version_key: str, *, root: Path | None = None
) -> Path:
    """Return the path of the per-atom-version decisions log.

    The catalog tree shape is ``<root>/.agentm/catalog/atoms/<name>/<version>/decisions.jsonl``.
    Atoms call this rather than importing the private ``_layout`` module
    in :mod:`agentm.core.runtime.catalog`.
    """

    base = (root or Path.cwd()) / _CATALOG_ROOT / "atoms" / name / version_key
    return base / "decisions.jsonl"


__all__ = [
    "CatalogService",
    "atom_decisions_path",
]
