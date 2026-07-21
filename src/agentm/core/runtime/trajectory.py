# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
"""In-memory append-only sequence of committed Turns with one active Execution slot."""

from __future__ import annotations

from typing import Sequence

from agentm.core.abi.trajectory import (
    Outcome,
    Turn,
    TurnRef,
    TurnMeta,
)
from agentm.core.abi.trigger import Trigger
from agentm.core.runtime.execution import Execution, StateError


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

    def begin(
        self,
        trigger: Trigger,
        *,
        run_id: str,
        run_step: int,
    ) -> Execution:
        """Start a new turn, returning its mutable Execution."""
        if self._active is not None:
            raise StateError("an execution is already active")
        self._active = Execution(
            index=len(self._turns),
            trigger=trigger,
            run_id=run_id,
            run_step=run_step,
        )
        return self._active

    def prepare_commit(self, outcome: Outcome, meta: TurnMeta) -> Turn:
        """Freeze the active execution without making the Turn visible yet."""
        if self._active is None:
            raise StateError("no active execution to commit")
        return self._active.commit(outcome, meta)

    def finalize_commit(self, turn: Turn) -> None:
        """Publish a prepared Turn after its durable append succeeds."""
        if self._active is None:
            raise StateError("no active execution to finalize")
        self._turns.append(turn)
        self._active = None

    def commit(self, outcome: Outcome, meta: TurnMeta) -> Turn:
        """Freeze and immediately publish the active Execution."""
        turn = self.prepare_commit(outcome, meta)
        self.finalize_commit(turn)
        return turn

    def abandon(self) -> None:
        """Discard the active Execution (crash/interrupt path).  No-op if none active."""
        if self._active is None:
            return
        self._active.abandon()
        self._active = None

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
