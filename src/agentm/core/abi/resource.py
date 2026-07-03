"""Resource-writer Protocol shared between the kernel and runtime.

Atoms (and the catalog freeze CLI) depend on this surface so they don't
need to import from ``agentm.core.runtime.resource_writer`` directly. The
concrete ``GitBackedResourceWriter`` lives in the runtime layer and
implements ``ResourceWriter`` implicitly.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

WriterAuthor = Literal["agent", "human", "indexer"]
PathClass = Literal["managed", "unmanaged", "constitution"]


@dataclass(frozen=True, slots=True)
class WriteResult:
    path: str
    path_class: PathClass
    committed: bool
    commit_sha_before: str | None
    commit_sha_after: str | None
    error: str | None = None

    @classmethod
    def _error(
        cls,
        path: str,
        path_class: PathClass,
        error: str,
        *,
        commit_sha_before: str | None = None,
    ) -> WriteResult:
        return cls(
            path=path,
            path_class=path_class,
            committed=False,
            commit_sha_before=commit_sha_before,
            commit_sha_after=None,
            error=error,
        )

    @classmethod
    def _uncommitted(
        cls,
        path: str,
        path_class: PathClass,
        *,
        commit_sha_before: str | None = None,
        commit_sha_after: str | None = None,
    ) -> WriteResult:
        return cls(
            path=path,
            path_class=path_class,
            committed=False,
            commit_sha_before=commit_sha_before,
            commit_sha_after=commit_sha_after,
        )


class BatchHandle(Protocol):
    async def write(self, path: str, content: bytes) -> None: ...

    async def replace(self, path: str, old: bytes, new: bytes) -> None: ...

    async def delete(self, path: str) -> None: ...


@runtime_checkable
class ResourceWriter(Protocol):
    async def read(self, path: str) -> bytes:
        """Read file content. Not restricted by path classification — any
        readable path is allowed (reads are safe; only writes are gated)."""
        ...

    async def exists(self, path: str) -> bool:
        """Return True if *path* exists and is readable."""
        ...

    async def list_dir(self, path: str) -> list[str]:
        """List entries in *path*. Raises ``FileNotFoundError`` if absent."""
        ...

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult: ...

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult: ...

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult: ...

    def classify(self, path: str) -> PathClass: ...

    def restore(self, path: Path, version: str) -> None:
        """Restore a managed resource to a previously recorded version."""
        ...

    def current_version_for_path(self, path: str) -> str | None:
        """Return the current version token for ``path`` if the writer tracks one."""
        ...

    def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> AbstractAsyncContextManager[BatchHandle]: ...


__all__ = [
    "BatchHandle",
    "PathClass",
    "ResourceWriter",
    "WriteResult",
    "WriterAuthor",
]
