"""Constitution-layer catalog package.

Currently re-exports the manifest parser and the boundary predicate. More
modules (storage, tool_catalog API) land in later tasks of the self-mod
plan.
"""

from agentm.core.catalog.manifest import (
    CoreManifest,
    is_constitution_path,
    load_core_manifest,
    reload_manifest,
)

__all__ = [
    "CoreManifest",
    "is_constitution_path",
    "load_core_manifest",
    "reload_manifest",
]
