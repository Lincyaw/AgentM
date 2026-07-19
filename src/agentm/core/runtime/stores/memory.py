"""In-memory TrajectoryStore — non-persistent, for tests and ephemeral runs."""

from __future__ import annotations


from collections.abc import Sequence

from agentm.core.abi.store import SessionMeta
from agentm.core.abi.trajectory import Turn, TurnCheckpoint, TurnRef
from agentm.core.lib.trajectory_store import (
    turn_prefix_cut,
    validate_checkpoint_commit,
    validate_turn_append,
    validate_turn_checkpoint,
    validate_turn_sequence,
)


class InMemoryTrajectoryStore:
    """A ``TrajectoryStore`` backed by a plain dict.

    Non-persistent: all state is lost on process exit.  Returns shallow
    copies of the stored Turn lists so callers cannot add/remove turns
    from internal state.  Turn objects themselves are frozen dataclasses
    and shared between copies.
    """

    __slots__ = ("_checkpoints", "_sessions")

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[SessionMeta, list[Turn]]] = {}
        self._checkpoints: dict[str, TurnCheckpoint] = {}

    def create_session(self, meta: SessionMeta) -> None:
        self.create_session_with_turns(meta, ())

    def create_session_with_turns(
        self, meta: SessionMeta, turns: Sequence[Turn]
    ) -> None:
        if meta.id in self._sessions:
            raise ValueError(f"session already exists: {meta.id}")
        copied = list(turns)
        validate_turn_sequence(copied)
        self._sessions[meta.id] = (meta, copied)

    def save_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        existing = self._checkpoints.get(session_id)
        validate_turn_checkpoint(record[1], checkpoint, existing=existing)
        self._checkpoints[session_id] = checkpoint

    def load_checkpoint(self, session_id: str) -> TurnCheckpoint | None:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._checkpoints.get(session_id)

    def append(self, session_id: str, turn: Turn) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        validate_turn_append(record[1], turn)
        validate_checkpoint_commit(self._checkpoints.get(session_id), turn)
        record[1].append(turn)
        self._checkpoints.pop(session_id, None)

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        meta, turns = record
        return (meta, list(turns))

    def load_prefix(
        self,
        session_id: str,
        up_to: TurnRef,
    ) -> tuple[SessionMeta, list[Turn]]:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        meta, turns = record
        cut = turn_prefix_cut(turns, up_to)
        return (meta, list(turns[: cut + 1]))

    def session_children(self, session_id: str) -> list[str]:
        return [
            sid
            for sid, (meta, _) in self._sessions.items()
            if meta.parent_id == session_id
        ]

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def list_sessions(self) -> list[SessionMeta]:
        return [meta for meta, _ in self._sessions.values()]

__all__ = ["InMemoryTrajectoryStore"]
