"""Mutable pending-turn state — the in-flight counterpart of a committed Turn."""

from __future__ import annotations

import time
from uuid import uuid4

from agentm.core.abi.messages import AgentMessage, AssistantMessage
from agentm.core.abi.trajectory import (
    Outcome,
    Round,
    ToolRecord,
    Turn,
    TurnMeta,
)
from agentm.core.abi.trigger import Trigger


class StateError(RuntimeError):
    """Raised on an invalid trajectory/execution state transition."""


class Execution:
    """The mutable state of a turn while it is being executed."""

    __slots__ = ("_index", "_id", "_trigger", "_rounds", "_active", "_injected")

    def __init__(self, index: int, trigger: Trigger) -> None:
        self._index = index
        self._id = uuid4().hex
        self._trigger = trigger
        self._rounds: list[Round] = []
        self._active = True
        self._injected: list[tuple[int, list[AgentMessage]]] = []

    @property
    def index(self) -> int:
        return self._index

    @property
    def id(self) -> str:
        return self._id

    @property
    def trigger(self) -> Trigger:
        return self._trigger

    @property
    def rounds(self) -> list[Round]:
        return self._rounds

    @property
    def active(self) -> bool:
        return self._active

    @property
    def injected(self) -> list[tuple[int, list[AgentMessage]]]:
        return self._injected

    def add_injected(self, messages: list[AgentMessage]) -> None:
        if not self._active:
            raise StateError("cannot add injected messages to an inactive execution")
        # Tag with current round count — inject goes AFTER this round
        round_index = len(self._rounds) - 1
        self._injected.append((round_index, messages))

    def add_round(
        self, response: AssistantMessage, tool_results: list[ToolRecord]
    ) -> None:
        """Append a completed LLM call + tool executions to this execution."""
        if not self._active:
            raise StateError("cannot add a round to an inactive execution")
        self._rounds.append(Round(response=response, tool_results=tuple(tool_results)))

    def commit(self, outcome: Outcome, meta: TurnMeta) -> Turn:
        """Freeze this execution into an immutable Turn."""
        if not self._active:
            raise StateError("cannot commit an inactive execution")
        self._active = False
        final_outcome = outcome
        all_injected = [msg for _, msgs in self._injected for msg in msgs]
        if all_injected:
            final_outcome = Outcome(
                action=outcome.action,
                cause=outcome.cause,
                injected=tuple(all_injected),
            )
        return Turn(
            index=self._index,
            id=self._id,
            trigger=self._trigger,
            rounds=tuple(self._rounds),
            outcome=final_outcome,
            timestamp=time.time(),
            meta=meta,
        )

    def abandon(self) -> None:
        """Discard this execution.  Idempotent."""
        self._active = False


__all__ = [
    "Execution",
    "StateError",
]
