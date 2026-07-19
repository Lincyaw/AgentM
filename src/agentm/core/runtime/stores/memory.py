"""In-memory TrajectoryStore — non-persistent, for tests and ephemeral runs."""

from __future__ import annotations


from collections.abc import Sequence

from agentm.core.abi.store import SessionMeta
from agentm.core.abi.trajectory import Turn, TurnRef


class InMemoryTrajectoryStore:
    """A ``TrajectoryStore`` backed by a plain dict.

    Non-persistent: all state is lost on process exit.  Returns shallow
    copies of the stored Turn lists so callers cannot add/remove turns
    from internal state.  Turn objects themselves are frozen dataclasses
    and shared between copies.
    """

    __slots__ = ("_sessions",)

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[SessionMeta, list[Turn]]] = {}

    def create_session(self, meta: SessionMeta) -> None:
        self.create_session_with_turns(meta, ())

    def create_session_with_turns(
        self, meta: SessionMeta, turns: Sequence[Turn]
    ) -> None:
        if meta.id in self._sessions:
            raise ValueError(f"session already exists: {meta.id}")
        copied = list(turns)
        _validate_turn_sequence(copied)
        self._sessions[meta.id] = (meta, copied)

    def append(self, session_id: str, turn: Turn) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        expected = len(record[1])
        if turn.index != expected:
            raise ValueError(f"turn index {turn.index} does not follow {expected - 1}")
        record[1].append(turn)

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
        cut = _prefix_cut(turns, up_to)
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


def _validate_turn_sequence(turns: Sequence[Turn]) -> None:
    for expected, turn in enumerate(turns):
        if turn.index != expected:
            raise ValueError(
                f"turn index {turn.index} does not match expected {expected}"
            )


def _prefix_cut(turns: list[Turn], up_to: TurnRef) -> int:
    """Return the list index of the turn identified by ``up_to``.

    ``up_to`` is a turn index (int) or a turn id (str).  Raises
    ``KeyError`` if no matching turn exists.
    """
    if isinstance(up_to, int):
        for i, turn in enumerate(turns):
            if turn.index == up_to:
                return i
        raise KeyError(up_to)
    for i, turn in enumerate(turns):
        if turn.id == up_to:
            return i
    raise KeyError(up_to)


__all__ = ["InMemoryTrajectoryStore"]
