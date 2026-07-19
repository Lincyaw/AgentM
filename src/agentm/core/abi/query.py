"""Trajectory and observability query ports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agentm.core.abi.store import SessionMeta
from agentm.core.abi.trajectory import Turn


@dataclass(frozen=True, slots=True)
class QueryMeta:
    """Flat backend-neutral metadata for query rows."""

    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    """Read-side session identity row."""

    id: str
    parent_session_id: str | None = None
    root_session_id: str | None = None
    purpose: str = "root"
    cwd: str = ""
    created_at: float = 0.0
    config: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_meta(cls, meta: SessionMeta) -> "SessionIdentity":
        root = meta.config.get("root_session_id")
        return cls(
            id=meta.id,
            parent_session_id=meta.parent_id,
            root_session_id=root if isinstance(root, str) else None,
            purpose=meta.purpose,
            cwd=meta.cwd,
            created_at=meta.created_at,
            config=dict(meta.config),
        )


@dataclass(frozen=True, slots=True)
class SessionFilter:
    """Portable filters for session identity queries."""

    session_id: str | None = None
    parent_session_id: str | None = None
    root_session_id: str | None = None
    purpose: str | None = None
    since: float | None = None
    until: float | None = None
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Read-side event row from an observability backend."""

    session_id: str
    name: str
    timestamp: float = 0.0
    payload: Mapping[str, object] = field(default_factory=dict)
    meta: QueryMeta = field(default_factory=QueryMeta)


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """Read-side span row from an observability backend."""

    session_id: str
    name: str
    span_id: str = ""
    parent_span_id: str | None = None
    start_time: float = 0.0
    end_time: float | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    meta: QueryMeta = field(default_factory=QueryMeta)


@runtime_checkable
class TrajectoryQueryStore(Protocol):
    """Read-side session and turn query view.

    Implementations may query a local ``TrajectoryStore``, Postgres,
    ClickHouse, or another backend. Observability events and spans are a
    separate port because many authoritative trajectory stores do not own
    telemetry data.
    """

    def sessions(
        self,
        filter: SessionFilter | None = None,
    ) -> Iterable[SessionIdentity]:
        ...

    def turns(self, session_id: str) -> Iterable[Turn]:
        ...


@runtime_checkable
class ObservabilityQueryStore(Protocol):
    """Read-side event/span query view."""

    def events(self, session_id: str) -> Iterable[EventRecord]:
        ...

    def spans(self, session_id: str) -> Iterable[SpanRecord]:
        ...


@runtime_checkable
class TraceQueryStore(TrajectoryQueryStore, ObservabilityQueryStore, Protocol):
    """Combined query view implemented by backends that own both datasets."""


__all__ = [
    "EventRecord",
    "ObservabilityQueryStore",
    "QueryMeta",
    "SessionFilter",
    "SessionIdentity",
    "SpanRecord",
    "TraceQueryStore",
    "TrajectoryQueryStore",
]
