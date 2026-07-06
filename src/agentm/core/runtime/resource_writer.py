"""Compatibility re-exports -- canonical location is now builtin/writer/local.py.

All public names that were historically importable from this module are still
available. Internal callers (session_factory, extension, atom_reloader, etc.)
continue to work without changes.
"""

from agentm.extensions.builtin.writer.local import (
    DEFAULT_PROTECTED_BRANCHES,
    GitBackedResourceWriter,
    GitOperationError,
    ProtectedBranchError,
)
from agentm.core.abi.resource import (
    BatchHandle,
    PathClass,
    ResourceWriter,
    WriteResult,
    WriterAuthor,
)

__all__ = [
    "BatchHandle",
    "DEFAULT_PROTECTED_BRANCHES",
    "GitBackedResourceWriter",
    "GitOperationError",
    "PathClass",
    "ProtectedBranchError",
    "ResourceWriter",
    "WriteResult",
    "WriterAuthor",
]
