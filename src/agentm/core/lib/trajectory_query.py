"""Read-side query adapter over a ``TrajectoryStore``."""

from __future__ import annotations

from collections.abc import Iterable

from agentm.core.abi.query import (
    SessionFilter,
    SessionIdentity,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import Turn, TurnCheckpoint


class TrajectoryStoreQueryAdapter:
    """Local query view for in-memory and JSONL trajectory stores."""

    __slots__ = ("_store",)

    def __init__(self, store: TrajectoryStore) -> None:
        self._store = store

    def sessions(
        self,
        filter: SessionFilter | None = None,
    ) -> Iterable[SessionIdentity]:
        rows = [SessionIdentity.from_meta(meta) for meta in self._store.list_sessions()]
        if filter is None:
            return rows
        rows = [row for row in rows if _session_matches(row, filter)]
        if filter.limit is not None:
            rows = rows[: filter.limit]
        return rows

    def turns(self, session_id: str) -> Iterable[Turn]:
        _, turns = self._store.load(session_id)
        return turns

    def checkpoints(self, session_id: str) -> Iterable[TurnCheckpoint]:
        checkpoint = self._store.load_checkpoint(session_id)
        return () if checkpoint is None else (checkpoint,)


def _session_matches(row: SessionIdentity, filter: SessionFilter) -> bool:
    if filter.session_id is not None and row.id != filter.session_id:
        return False
    if (
        filter.parent_session_id is not None
        and row.parent_session_id != filter.parent_session_id
    ):
        return False
    if (
        filter.root_session_id is not None
        and _root_session_id(row) != filter.root_session_id
    ):
        return False
    if filter.purpose is not None and row.purpose != filter.purpose:
        return False
    if filter.since is not None and row.created_at < filter.since:
        return False
    if filter.until is not None and row.created_at > filter.until:
        return False
    return True


def _root_session_id(row: SessionIdentity) -> str:
    return row.root_session_id or row.id


__all__ = ["TrajectoryStoreQueryAdapter"]
