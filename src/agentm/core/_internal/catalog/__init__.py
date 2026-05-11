"""Constitution-layer catalog: pure kernel functions only.

Filesystem-shaped helpers (freeze, migrate, indexer, project layout) live
in :mod:`agentm.core.runtime.catalog`. This package is import-safe in any cwd:
no filesystem reads happen at import time, no path-walking heuristics,
no implicit reach into :mod:`agentm.extensions`.
"""

from __future__ import annotations

from agentm.core._internal.catalog.browse import (
    CatalogAtom,
    UnparseableManifestError,
    current_version,
    get_manifest_at,
    get_source_at,
    list_versions,
    runs_for,
)
from agentm.core._internal.catalog.hashing import (
    compute_active_set_fingerprint,
    compute_atom_hash,
)
from agentm.core._internal.catalog.manifest import is_constitution_path

__all__ = [
    "CatalogAtom",
    "UnparseableManifestError",
    "compute_active_set_fingerprint",
    "compute_atom_hash",
    "current_version",
    "get_manifest_at",
    "get_source_at",
    "is_constitution_path",
    "list_versions",
    "runs_for",
]
