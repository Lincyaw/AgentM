# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""Composition adapter for independent trace query data planes."""

from __future__ import annotations

from collections.abc import Iterable

from agentm.core.abi.query import (
    EventRecord,
    ObservabilityQueryStore,
    SessionFilter,
    SessionIdentity,
    SpanRecord,
    TrajectoryQueryStore,
)
from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from agentm.core.abi.store import TrajectoryDiagnostic


class CompositeTraceQueryStore:
    """Compose one trajectory source with one observability source."""

    __slots__ = ("_observability", "_trajectory")

    def __init__(
        self,
        trajectory: TrajectoryQueryStore,
        observability: ObservabilityQueryStore,
    ) -> None:
        if not isinstance(trajectory, TrajectoryQueryStore):
            raise TypeError("CompositeTraceQueryStore requires a TrajectoryQueryStore")
        if not isinstance(observability, ObservabilityQueryStore):
            raise TypeError(
                "CompositeTraceQueryStore requires an ObservabilityQueryStore"
            )
        self._trajectory = trajectory
        self._observability = observability

    def sessions(
        self,
        filter: SessionFilter | None = None,
    ) -> Iterable[SessionIdentity]:
        return self._trajectory.sessions(filter)

    def turns(self, session_id: str) -> Iterable[Turn]:
        return self._trajectory.turns(session_id)

    def checkpoints(self, session_id: str) -> Iterable[TurnCheckpoint]:
        return self._trajectory.checkpoints(session_id)

    def diagnostics(self, session_id: str) -> Iterable[TrajectoryDiagnostic]:
        return self._trajectory.diagnostics(session_id)

    def events(self, session_id: str) -> Iterable[EventRecord]:
        return self._observability.events(session_id)

    def spans(self, session_id: str) -> Iterable[SpanRecord]:
        return self._observability.spans(session_id)


__all__ = ["CompositeTraceQueryStore"]
