"""In-memory append-only sequence of committed Turns with one active Execution slot."""

from __future__ import annotations

from typing import Sequence

from agentm.core.v2.abi.trajectory import (
    Outcome,
    Turn,
    TurnMeta,
    TurnRef,
)
from agentm.core.v2.abi.trigger import Trigger
from agentm.core.v2.runtime.execution import Execution, StateError


class Trajectory:
    """An append-only sequence of committed Turns plus a single active Execution slot."""

    __slots__ = ("_turns", "_active")

    def __init__(self, turns: list[Turn] | None = None) -> None:
        self._turns: list[Turn] = list(turns) if turns else []
        self._active: Execution | None = None

    @property
    def turns(self) -> Sequence[Turn]:
        return tuple(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def is_executing(self) -> bool:
        return self._active is not None

    def begin(self, trigger: Trigger) -> Execution:
        """Start a new turn, returning its mutable Execution."""
        if self._active is not None:
            raise StateError("an execution is already active")
        self._active = Execution(index=len(self._turns), trigger=trigger)
        return self._active

    def commit(self, outcome: Outcome, meta: TurnMeta) -> Turn:
        """Freeze the active Execution into a Turn and append it."""
        if self._active is None:
            raise StateError("no active execution to commit")
        turn = self._active.commit(outcome, meta)
        self._turns.append(turn)
        self._active = None
        return turn

    def abandon(self) -> None:
        """Discard the active Execution (crash/interrupt path).  No-op if none active."""
        if self._active is None:
            return
        self._active.abandon()
        self._active = None

    def find_turn(self, ref: TurnRef) -> Turn | None:
        """Look up a committed turn by index (int) or id (str)."""
        if isinstance(ref, int):
            return next((t for t in self._turns if t.index == ref), None)
        return next((t for t in self._turns if t.id == ref), None)

    def prefix(self, ref: TurnRef) -> Trajectory:
        """Return a new Trajectory with committed turns up to and including ``ref``."""
        for pos, turn in enumerate(self._turns):
            match = turn.index == ref if isinstance(ref, int) else turn.id == ref
            if match:
                return Trajectory(self._turns[: pos + 1])
        raise KeyError(ref)


__all__ = [
    "StateError",
    "Trajectory",
]
