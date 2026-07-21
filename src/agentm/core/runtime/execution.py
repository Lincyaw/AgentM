"""Mutable pending-turn state — the in-flight counterpart of a committed Turn."""

from __future__ import annotations

import time
from collections.abc import Sequence
from uuid import uuid4

from agentm.core.abi.messages import AgentMessage, AssistantMessage
from agentm.core.abi.trajectory import (
    Outcome,
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

    __slots__ = (
        "_index",
        "_id",
        "_run_id",
        "_run_step",
        "_trigger",
        "_response",
        "_tool_results",
        "_active",
        "_injected",
    )

    def __init__(
        self,
        index: int,
        trigger: Trigger,
        *,
        run_id: str,
        run_step: int,
    ) -> None:
        self._index = index
        self._id = uuid4().hex
        self._run_id = run_id
        self._run_step = run_step
        self._trigger = trigger
        self._response: AssistantMessage | None = None
        self._tool_results: tuple[ToolRecord, ...] = ()
        self._active = True
        self._injected: list[AgentMessage] = []

    @property
    def index(self) -> int:
        return self._index

    @property
    def id(self) -> str:
        return self._id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_step(self) -> int:
        return self._run_step

    @property
    def trigger(self) -> Trigger:
        return self._trigger

    @property
    def response(self) -> AssistantMessage | None:
        return self._response

    @property
    def tool_results(self) -> tuple[ToolRecord, ...]:
        return self._tool_results

    @property
    def active(self) -> bool:
        return self._active

    @property
    def injected(self) -> list[AgentMessage]:
        return self._injected

    def add_injected(self, messages: list[AgentMessage]) -> None:
        if not self._active:
            raise StateError("cannot add injected messages to an inactive execution")
        self._injected.extend(messages)

    def set_result(
        self, response: AssistantMessage, tool_results: list[ToolRecord]
    ) -> None:
        """Materialize this turn's single model response and tool results."""
        if not self._active:
            raise StateError("cannot set result on an inactive execution")
        if self._response is not None:
            raise StateError("execution result is already materialized")
        self._response = response
        self._tool_results = tuple(tool_results)

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
        if pending_response is None:
            response = self._response
            tool_results = self._tool_results
        else:
            response = pending_response
            tool_results = tuple(pending_tool_results)
        return TurnCheckpoint(
            index=self._index,
            id=self._id,
            run_id=self._run_id,
            run_step=self._run_step,
            trigger=self._trigger,
            response=response,
            tool_results=tool_results,
            injected=tuple(self._injected),
            updated_at=time.time(),
            meta=meta,
            trigger_metadata=trigger_metadata,
        )

    def commit(self, outcome: Outcome, meta: TurnMeta) -> Turn:
        """Freeze this execution into an immutable Turn."""
        if not self._active:
            raise StateError("cannot commit an inactive execution")
        self._active = False
        anchored = tuple(self._injected)
        final_outcome = (
            Outcome(cause=outcome.cause, injected=anchored) if anchored else outcome
        )
        return Turn(
            index=self._index,
            id=self._id,
            run_id=self._run_id,
            run_step=self._run_step,
            trigger=self._trigger,
            response=self._response,
            tool_results=self._tool_results,
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
