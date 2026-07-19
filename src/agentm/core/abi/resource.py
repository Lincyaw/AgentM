"""Resource mutation port for SDK atoms.

ResourceWriter is the boundary for writes that outlive a tool call: files,
artifacts, generated reports, or host-managed workspace entries. Tool atoms
can ask for a resource mutation without deciding where that resource lives,
how it is audited, or which paths are protected.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


WriterAuthor = Literal["agent", "human", "indexer"]
PathClass = Literal["managed", "unmanaged", "constitution"]
ResourceMutationOp = Literal["write", "replace", "delete"]
ResourceMeta = dict[str, str | int | float | bool | None]


@dataclass(frozen=True, slots=True)
class ResourceRef:
    """Stable identity for a mutable host resource."""

    namespace: str
    path: str


@dataclass(frozen=True, slots=True)
class ResourceMutation:
    """One durable mutation produced inside a resource transaction."""

    ref: ResourceRef
    op: ResourceMutationOp
    before_version: str | None = None
    after_version: str | None = None
    metadata: ResourceMeta = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceTxnContext:
    """Trajectory identity attached to one resource transaction."""

    session_id: str
    turn_id: str
    turn_index: int
    rationale: str = ""
    author: WriterAuthor = "agent"


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome returned by a ResourceWriter mutation."""

    path: str
    path_class: PathClass
    error: str | None = None


class BatchHandle(Protocol):
    """Atomic mutation group returned by ``ResourceWriter.batch``."""

    async def write(self, path: str, content: bytes) -> None: ...

    async def replace(self, path: str, old: bytes, new: bytes) -> None: ...

    async def delete(self, path: str) -> None: ...


@runtime_checkable
class ResourceTxn(Protocol):
    """Turn-scoped resource mutation transaction."""

    async def write(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        ...

    async def replace(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        ...

    async def delete(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        ...

    async def commit(self) -> list[ResourceMutation]:
        ...

    async def abandon(self) -> None:
        ...


@runtime_checkable
class ResourceWriter(Protocol):
    """Host-provided resource access and mutation boundary."""

    async def read(self, path: str) -> bytes: ...

    async def exists(self, path: str) -> bool: ...

    async def list_dir(self, path: str) -> list[str]: ...

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

    def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> AbstractAsyncContextManager[BatchHandle]: ...


@runtime_checkable
class TransactionalResourceWriter(ResourceWriter, Protocol):
    """ResourceWriter that can create turn-scoped transactions."""

    async def begin_txn(self, context: ResourceTxnContext) -> ResourceTxn:
        ...


__all__ = [
    "BatchHandle",
    "PathClass",
    "ResourceMeta",
    "ResourceMutation",
    "ResourceMutationOp",
    "ResourceRef",
    "ResourceTxn",
    "ResourceTxnContext",
    "ResourceWriter",
    "TransactionalResourceWriter",
    "WriteResult",
    "WriterAuthor",
]
