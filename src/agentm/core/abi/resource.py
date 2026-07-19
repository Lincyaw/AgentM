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
ResourceNamespace = Literal[
    "workspace",
    "artifact",
    "sandbox",
    "summary",
    "content",
    "catalog",
    "observability",
    "environment",
]
ResourceMeta = dict[str, str | int | float | bool | None]
RESOURCE_NAMESPACE_WORKSPACE = "workspace"
RESOURCE_NAMESPACE_ARTIFACT = "artifact"
RESOURCE_NAMESPACE_SANDBOX = "sandbox"
RESOURCE_NAMESPACE_SUMMARY = "summary"
RESOURCE_NAMESPACE_CONTENT = "content"
RESOURCE_NAMESPACE_CATALOG = "catalog"
RESOURCE_NAMESPACE_OBSERVABILITY = "observability"
RESOURCE_NAMESPACE_ENVIRONMENT = "environment"


@dataclass(frozen=True, slots=True)
class ResourceRef:
    """Stable identity for a mutable host resource.

    ``namespace`` is logical, not physical. Backends decide whether
    ``content:foo`` is Postgres, S3, local disk, or an environment-local file.
    """

    namespace: str
    path: str

    def uri(self) -> str:
        """Return a compact, stable ``namespace:path`` representation."""

        return f"{self.namespace}:{self.path}"

    @classmethod
    def parse(cls, value: str) -> "ResourceRef":
        """Parse ``namespace:path`` into a ``ResourceRef``."""

        namespace, separator, path = value.partition(":")
        if not separator or not namespace or not path:
            raise ValueError(f"invalid ResourceRef URI: {value!r}")
        return cls(namespace=namespace, path=path)


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
class ResourceReader(Protocol):
    """Backend-neutral read authority for logical ``ResourceRef`` values.

    Existing file-oriented code may still read through ``Operations`` or the
    compatibility methods on ``ResourceWriter``. New code that needs to
    dereference trajectory ``content_ref`` values should use this Protocol.
    """

    async def read_ref(self, ref: ResourceRef) -> bytes: ...

    async def exists_ref(self, ref: ResourceRef) -> bool: ...

    async def list_ref(self, ref: ResourceRef) -> list[ResourceRef]: ...


@runtime_checkable
class ResourceStore(ResourceReader, Protocol):
    """Durable read/write authority for logical ``ResourceRef`` values.

    Unlike ``ResourceWriter``, this port has no workspace-path semantics.
    Policies use it for host-managed content such as summaries and artifacts
    that may live in a filesystem, object store, or database.
    """

    async def write_ref(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation: ...

    async def replace_ref(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation: ...

    async def delete_ref(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation: ...


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
    """Host-provided resource mutation boundary.

    The path-based read helpers remain for compatibility with existing file
    tools. Backend-neutral content dereference should use ``ResourceReader`` or
    environment ``Operations`` according to caller authority.
    """

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
    "RESOURCE_NAMESPACE_ARTIFACT",
    "RESOURCE_NAMESPACE_CATALOG",
    "RESOURCE_NAMESPACE_CONTENT",
    "RESOURCE_NAMESPACE_ENVIRONMENT",
    "RESOURCE_NAMESPACE_OBSERVABILITY",
    "RESOURCE_NAMESPACE_SANDBOX",
    "RESOURCE_NAMESPACE_SUMMARY",
    "RESOURCE_NAMESPACE_WORKSPACE",
    "ResourceMeta",
    "ResourceMutation",
    "ResourceMutationOp",
    "ResourceNamespace",
    "ResourceReader",
    "ResourceRef",
    "ResourceStore",
    "ResourceTxn",
    "ResourceTxnContext",
    "ResourceWriter",
    "TransactionalResourceWriter",
    "WriteResult",
    "WriterAuthor",
]
