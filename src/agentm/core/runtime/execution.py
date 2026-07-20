"""Mutable pending-turn state — the in-flight counterpart of a committed Turn."""

from __future__ import annotations

import time
from collections.abc import Sequence
from uuid import uuid4

from agentm.core.abi.messages import AgentMessage, AssistantMessage
from agentm.core.abi.trajectory import (
    InjectedMessages,
    Outcome,
    Round,
    ToolRecord,
    Turn,
    TurnCheckpoint,
    TurnMeta,
)
from agentm.core.abi.trigger import Trigger, TriggerMetadata


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

    def checkpoint(
        self,
        meta: TurnMeta,
        *,
        trigger_metadata: TriggerMetadata | None = None,
        pending_response: AssistantMessage | None = None,
        pending_tool_results: Sequence[ToolRecord] = (),
    ) -> TurnCheckpoint:
        """Snapshot materialized progress without ending the execution."""
        if not self._active:
            raise StateError("cannot checkpoint an inactive execution")
        if pending_tool_results and pending_response is None:
            raise StateError("pending tool results require their assistant response")
        rounds = tuple(self._rounds)
        if pending_response is not None:
            rounds = (
                *rounds,
                Round(
                    response=pending_response,
                    tool_results=tuple(pending_tool_results),
                ),
            )
        anchored = tuple(
            InjectedMessages(after_round=round_index, messages=tuple(messages))
            for round_index, messages in self._injected
        )
        return TurnCheckpoint(
            index=self._index,
            id=self._id,
            trigger=self._trigger,
            rounds=rounds,
            injected=anchored,
            updated_at=time.time(),
            meta=meta,
            trigger_metadata=trigger_metadata,
        )

    def commit(self, outcome: Outcome, meta: TurnMeta) -> Turn:
        """Freeze this execution into an immutable Turn."""
        if not self._active:
            raise StateError("cannot commit an inactive execution")
        self._active = False
        anchored = tuple(
            InjectedMessages(after_round=round_index, messages=tuple(messages))
            for round_index, messages in self._injected
        )
        final_outcome = (
            Outcome(cause=outcome.cause, injected=anchored) if anchored else outcome
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
