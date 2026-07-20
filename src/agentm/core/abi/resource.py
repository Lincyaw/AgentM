"""Resource mutation port for SDK atoms.

ResourceWriter is the boundary for writes that outlive a tool call: files,
artifacts, generated reports, or host-managed workspace entries. Tool atoms
can ask for a resource mutation without deciding where that resource lives,
how it is audited, or which paths are protected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable


WriterAuthor = Literal["agent", "human", "indexer"]
PathClass = Literal["managed", "unmanaged", "constitution"]
ResourceMutationOp = Literal["create", "write", "replace", "delete"]
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
ResourceMeta = Mapping[str, str | int | float | bool | None]
RESOURCE_NAMESPACE_WORKSPACE = "workspace"
RESOURCE_NAMESPACE_ARTIFACT = "artifact"
RESOURCE_NAMESPACE_SANDBOX = "sandbox"
RESOURCE_NAMESPACE_SUMMARY = "summary"
RESOURCE_NAMESPACE_CONTENT = "content"
RESOURCE_NAMESPACE_CATALOG = "catalog"
RESOURCE_NAMESPACE_OBSERVABILITY = "observability"
RESOURCE_NAMESPACE_ENVIRONMENT = "environment"


def _require_nonempty_string(
    value: object,
    label: str,
    *,
    optional: bool = False,
) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value:
        expected = "a non-empty string or None" if optional else "a non-empty string"
        raise TypeError(f"{label} must be {expected}")


def _require_index(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")


def _freeze_metadata(value: ResourceMeta, label: str) -> ResourceMeta:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    copied: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise TypeError(f"{label}[{key!r}] must be a JSON scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label}[{key!r}] must be finite")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ResourceRef:
    """Stable identity for a mutable host resource.

    ``namespace`` is logical, not physical. Backends decide whether
    ``content:foo`` is Postgres, S3, local disk, or an environment-local file.
    """

    namespace: str
    path: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or not self.namespace
            or ":" in self.namespace
            or "\0" in self.namespace
        ):
            raise ValueError(f"invalid resource namespace: {self.namespace!r}")
        if not isinstance(self.path, str) or not self.path or "\0" in self.path:
            raise ValueError(f"invalid resource path: {self.path!r}")

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
class ResourceTransactionRef:
    """Stable origin identity for one turn-scoped resource transaction."""

    id: str
    session_id: str
    turn_id: str
    turn_index: int

    def __post_init__(self) -> None:
        _require_nonempty_string(self.id, "resource transaction ref id")
        _require_nonempty_string(
            self.session_id,
            "resource transaction ref session_id",
        )
        _require_nonempty_string(self.turn_id, "resource transaction ref turn_id")
        _require_index(self.turn_index, "resource transaction ref turn_index")


@dataclass(frozen=True, slots=True)
class ResourceMutation:
    """One durable mutation, including its original transaction ownership."""

    ref: ResourceRef
    op: ResourceMutationOp
    transaction: ResourceTransactionRef | None = None
    before_version: str | None = None
    after_version: str | None = None
    metadata: ResourceMeta = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ResourceRef):
            raise TypeError("resource mutation ref must be a ResourceRef")
        if self.op not in {"create", "write", "replace", "delete"}:
            raise ValueError(f"invalid resource mutation op: {self.op!r}")
        if self.transaction is not None and not isinstance(
            self.transaction,
            ResourceTransactionRef,
        ):
            raise TypeError(
                "resource mutation transaction must be a ResourceTransactionRef"
            )
        _require_nonempty_string(
            self.before_version,
            "resource mutation before_version",
            optional=True,
        )
        _require_nonempty_string(
            self.after_version,
            "resource mutation after_version",
            optional=True,
        )
        if self.op == "create" and self.before_version is not None:
            raise ValueError("create mutation cannot have a before_version")
        if self.op in {"create", "write"} and self.after_version is None:
            raise ValueError(f"{self.op} mutation requires an after_version")
        if self.op == "replace" and (
            self.before_version is None or self.after_version is None
        ):
            raise ValueError(
                "replace mutation requires before_version and after_version"
            )
        if self.op == "delete" and self.after_version is not None:
            raise ValueError("delete mutation cannot have an after_version")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "resource mutation metadata"),
        )


@dataclass(frozen=True, slots=True)
class ResourceTxnContext:
    """Trajectory identity attached to one resource transaction."""

    session_id: str
    turn_id: str
    turn_index: int
    rationale: str = ""
    author: WriterAuthor = "agent"

    def __post_init__(self) -> None:
        _require_nonempty_string(self.session_id, "resource transaction session_id")
        _require_nonempty_string(self.turn_id, "resource transaction turn_id")
        _require_index(self.turn_index, "resource transaction turn_index")
        if not isinstance(self.rationale, str):
            raise TypeError("resource transaction rationale must be a string")
        if self.author not in {"agent", "human", "indexer"}:
            raise ValueError(f"invalid resource transaction author: {self.author!r}")


@dataclass(frozen=True, slots=True)
class ResourceRecoveryContext:
    """Committed transactions owned by one session and eligible for recovery."""

    session_id: str
    committed_transactions: tuple[ResourceTransactionRef, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.session_id, "resource recovery session_id")
        if not isinstance(self.committed_transactions, tuple) or not all(
            isinstance(transaction, ResourceTransactionRef)
            for transaction in self.committed_transactions
        ):
            raise TypeError(
                "committed_transactions must be a tuple of ResourceTransactionRef"
            )
        if any(
            transaction.session_id != self.session_id
            for transaction in self.committed_transactions
        ):
            raise ValueError(
                "committed resource transactions must belong to the recovery session"
            )
        transaction_ids = [
            transaction.id for transaction in self.committed_transactions
        ]
        if len(set(transaction_ids)) != len(transaction_ids):
            raise ValueError("committed resource transactions must be unique")


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome returned by a ResourceWriter mutation."""

    path: str
    path_class: PathClass
    error: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.path, "write result path")
        if self.path_class not in {"managed", "unmanaged", "constitution"}:
            raise ValueError(f"invalid write result path_class: {self.path_class!r}")
        _require_nonempty_string(self.error, "write result error", optional=True)


@runtime_checkable
class ResourceReader(Protocol):
    """Backend-neutral read authority for logical ``ResourceRef`` values.

    Workspace tools use ``ResourceWriter`` because its path surface carries
    workspace authority. Code that dereferences trajectory ``content_ref``
    values uses this logical-resource Protocol instead.
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
    """Recoverable turn-scoped resource mutation transaction.

    ``prepare`` durably stages and validates mutations without making them
    visible. ``apply`` makes them visible but still reversible so the effect
    backend can snapshot the resulting world. The caller then records the
    returned mutations in the authoritative Turn and calls ``commit`` after
    its append. Recovery commits applied transactions referenced by durable
    Turns and abandons every other prepared/applied transaction.
    """

    async def read(self, ref: ResourceRef) -> bytes | None:
        """Read the transaction's current staged view."""

        ...

    async def create(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation:
        """Create a resource and fail if it already exists."""

        ...

    async def replace(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation: ...

    async def delete(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> ResourceMutation: ...

    async def prepare(self) -> Sequence[ResourceMutation]: ...

    async def apply(self) -> None: ...

    async def commit(self) -> None: ...

    async def abandon(self) -> None: ...


@runtime_checkable
class ResourceWriter(Protocol):
    """Host-provided workspace mutation and read-before-write boundary."""

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


@runtime_checkable
class TransactionalResourceWriter(ResourceWriter, Protocol):
    """ResourceWriter that can create turn-scoped transactions."""

    async def begin_txn(self, context: ResourceTxnContext) -> ResourceTxn: ...

    async def recover(self, context: ResourceRecoveryContext) -> None:
        """Finalize committed prepared txns and discard uncommitted staging."""
        ...


@runtime_checkable
class EnvironmentForkableResourceWriter(Protocol):
    """Workspace writer that can rebind itself to a forked environment.

    This operation may validate or construct a binding but must not allocate
    independently owned external resources. The paired ``EnvironmentForkLease``
    owns all provisional environment state so failed SDK fork construction has
    one complete cleanup boundary.
    """

    async def fork_for_environment(
        self,
        *,
        workspace_root: str,
        child_session_id: str,
    ) -> ResourceWriter: ...


__all__ = [
    "EnvironmentForkableResourceWriter",
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
    "ResourceRecoveryContext",
    "ResourceRef",
    "ResourceStore",
    "ResourceTransactionRef",
    "ResourceTxn",
    "ResourceTxnContext",
    "ResourceWriter",
    "TransactionalResourceWriter",
    "WriteResult",
    "WriterAuthor",
]
