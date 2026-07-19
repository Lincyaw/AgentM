"""Versioned resource and atom active-set catalog ports.

Catalog identity is composition identity: which atom modules, manifests, and
configs formed a session. It is not the same boundary as ``ResourceWriter``,
which mutates user/workspace resources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.manifest import AtomInstallPriority


CatalogMeta = Mapping[str, str | int | float | bool | None]
CatalogQuerySort = Literal["asc", "desc"]


@dataclass(frozen=True, slots=True)
class ResourceVersion:
    """Immutable SDK resource version.

    This is for versioning SDK composition inputs such as atom source,
    manifests, generated prompts, or scenario specs. It is intentionally
    separate from ``ResourceWriter``, which mutates user/workspace resources.
    """

    resource_id: str
    version_id: str
    digest: str
    media_type: str | None = None
    size_bytes: int = 0
    metadata: CatalogMeta = field(default_factory=dict)


@runtime_checkable
class VersionedResourceStore(Protocol):
    """Content-addressed store for immutable SDK resources."""

    async def put(
        self,
        *,
        resource_id: str,
        content: bytes,
        media_type: str | None = None,
        metadata: CatalogMeta | None = None,
    ) -> ResourceVersion:
        ...

    async def resolve(
        self,
        resource_id: str,
        *,
        version_id: str | None = None,
    ) -> ResourceVersion | None:
        ...

    async def read(self, version: ResourceVersion) -> bytes:
        ...

    async def alias(self, alias: str, version: ResourceVersion) -> None:
        ...

    async def resolve_alias(self, alias: str) -> ResourceVersion | None:
        ...

    async def list_versions(self, resource_id: str) -> list[ResourceVersion]:
        ...


@dataclass(frozen=True, slots=True)
class AtomActivation:
    """One atom as it appears in a resolved session active set."""

    name: str
    module_path: str
    version: ResourceVersion | None = None
    priority: int = AtomInstallPriority.NORMAL
    requires: tuple[str, ...] = ()
    registers: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    provided_capabilities: tuple[str, ...] = ()
    config_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveSetFingerprint:
    """Stable fingerprint for the atom set installed into a session."""

    algorithm: str
    digest: str
    atoms: tuple[AtomActivation, ...] = ()
    metadata: CatalogMeta = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogActiveSetInput:
    """Resolved active-set data to persist into an atom catalog."""

    session_id: str
    atoms: tuple[AtomActivation, ...]
    root_session_id: str | None = None
    parent_session_id: str | None = None
    scenario: str | None = None
    provider: str | None = None
    created_at: float = 0.0
    metadata: CatalogMeta = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogActiveSetRecord:
    """One queryable active-set record.

    The catalog query surface is intentionally separate from trajectory query:
    it indexes SDK composition identity, not session event order.
    """

    session_id: str
    fingerprint: ActiveSetFingerprint
    root_session_id: str | None = None
    parent_session_id: str | None = None
    scenario: str | None = None
    provider: str | None = None
    created_at: float = 0.0
    metadata: CatalogMeta = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogQuery:
    """Portable predicate shape for atom/catalog indexes."""

    session_id: str | None = None
    root_session_id: str | None = None
    parent_session_id: str | None = None
    scenario: str | None = None
    provider: str | None = None
    atom_name: str | None = None
    module_path: str | None = None
    register: str | None = None
    require: str | None = None
    digest: str | None = None
    version_id: str | None = None
    limit: int | None = None
    sort: CatalogQuerySort = "asc"


@runtime_checkable
class AtomCatalog(Protocol):
    """Catalog for resolved atom composition identity."""

    async def record_active_set(
        self,
        active_set: CatalogActiveSetInput,
    ) -> ActiveSetFingerprint:
        ...

    async def get_active_set(
        self,
        session_id: str,
    ) -> ActiveSetFingerprint | None:
        ...


@runtime_checkable
class AtomCatalogQuery(Protocol):
    """Optional indexed query surface for active-set catalog records."""

    async def query_active_sets(
        self,
        query: CatalogQuery,
    ) -> list[CatalogActiveSetRecord]:
        ...


__all__ = [
    "ActiveSetFingerprint",
    "AtomActivation",
    "AtomCatalog",
    "AtomCatalogQuery",
    "CatalogActiveSetInput",
    "CatalogActiveSetRecord",
    "CatalogMeta",
    "CatalogQuery",
    "CatalogQuerySort",
    "ResourceVersion",
    "VersionedResourceStore",
]
