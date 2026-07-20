# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
"""Trigger queue with per-trigger completion and host-idle tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from agentm.core.abi.messages import JsonValue
from agentm.core.abi.trigger import (
    ContinueTrigger,
    Trigger,
    TriggerEnvelope,
    TriggerMetadata,
    TriggerPriority,
    trigger_priority_rank,
)


class QueueClosed(Exception):
    """Raised by ``wait()`` after the queue has been closed."""


class TriggerTerminated(RuntimeError):
    """Raised when an accepted trigger terminates without a successful result."""

    def __init__(self, cause: object) -> None:
        self.cause = cause
        super().__init__(f"trigger terminated: {type(cause).__name__}: {cause}")


_T = TypeVar("_T")


class TriggerReceipt(Generic[_T]):
    """Awaitable terminal receipt for one accepted trigger."""

    __slots__ = ("_future",)

    def __init__(self) -> None:
        self._future: asyncio.Future[_T] = asyncio.get_running_loop().create_future()

    async def wait(self) -> _T:
        return await asyncio.shield(self._future)

    def _succeed(self, result: _T) -> None:
        if not self._future.done():
            self._future.set_result(result)

    def _fail(self, exc: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(exc)
            # Retrieve the exception so fire-and-forget trigger submissions do not
            # produce "Future exception was never retrieved" warnings.
            self._future.exception()


@dataclass(frozen=True, slots=True)
class _AcceptedTrigger:
    envelope: TriggerEnvelope
    receipt: TriggerReceipt[object]


@dataclass(frozen=True, slots=True)
class _CloseSentinel:
    source: str = "close"


class TriggerQueue:
    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, object]] = (
            asyncio.PriorityQueue()
        )
        self._pending_work = 0
        self._in_flight: _AcceptedTrigger | None = None
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._seq = 0

    def push(
        self,
        trigger: Trigger,
        *,
        priority: TriggerPriority = "next",
        target_session_id: str | None = None,
        target_agent_id: str | None = None,
        origin: str | None = None,
        mode: str = "prompt",
        is_meta: bool = False,
        skip_commands: bool = False,
        meta: dict[str, JsonValue] | None = None,
    ) -> TriggerReceipt[object]:
        envelope = TriggerEnvelope(
            trigger=trigger,
            metadata=TriggerMetadata(
                priority=priority,
                target_session_id=target_session_id,
                target_agent_id=target_agent_id,
                origin=origin,
                mode=mode,
                is_meta=is_meta,
                skip_commands=skip_commands,
                meta=meta or {},
            ),
        )
        return self.push_envelope(envelope)

    def push_envelope(self, envelope: TriggerEnvelope) -> TriggerReceipt[object]:
        if self._closed:
            raise QueueClosed
        receipt: TriggerReceipt[object] = TriggerReceipt()
        self._put(_AcceptedTrigger(envelope, receipt), envelope.priority)
        self._idle.clear()
        return receipt

    def kick(self) -> TriggerReceipt[object]:
        return self.push(cast(Trigger, ContinueTrigger()), mode="internal")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._put(_CloseSentinel(), "later")

    def terminate(self, exc: BaseException) -> None:
        """Close the queue and fail every accepted trigger atomically."""
        self._closed = True
        if self._in_flight is not None:
            self.fail(exc)
        self.fail_pending(exc)

    async def wait(self) -> Trigger:
        return (await self.wait_envelope()).trigger

    async def wait_envelope(self) -> TriggerEnvelope:
        _, _, item = await self._queue.get()
        if isinstance(item, _CloseSentinel):
            raise QueueClosed
        accepted = cast(_AcceptedTrigger, item)
        if self._in_flight is not None:
            raise RuntimeError(
                "trigger consumer received while another trigger is in flight"
            )
        self._in_flight = accepted
        self._idle.clear()
        return accepted.envelope

    def complete(self, result: object) -> None:
        accepted = self._take_in_flight()
        accepted.receipt._succeed(result)
        self._update_idle()

    def fail(self, exc: BaseException) -> None:
        accepted = self._take_in_flight()
        accepted.receipt._fail(exc)
        self._update_idle()

    def fail_pending(self, exc: BaseException) -> None:
        while True:
            try:
                _, _, item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, _AcceptedTrigger):
                item.receipt._fail(exc)
        self._update_idle()

    def _take_in_flight(self) -> _AcceptedTrigger:
        accepted = self._in_flight
        if accepted is None:
            raise RuntimeError("no trigger is in flight")
        self._in_flight = None
        return accepted

    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def has_in_flight(self) -> bool:
        return self._in_flight is not None

    @property
    def current_envelope(self) -> TriggerEnvelope | None:
        if self._in_flight is None:
            return None
        return self._in_flight.envelope

    def note_work_started(self) -> None:
        self._pending_work += 1
        self._idle.clear()

    def note_work_finished(self) -> None:
        if self._pending_work > 0:
            self._pending_work -= 1
        self._update_idle()

    @property
    def has_pending_work(self) -> bool:
        return self._pending_work > 0

    async def wait_quiescent(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    def _update_idle(self) -> None:
        if self._pending_work == 0 and self._in_flight is None and self._queue.empty():
            self._idle.set()
        else:
            self._idle.clear()

    def _put(self, item: object, priority: TriggerPriority) -> None:
        self._queue.put_nowait((trigger_priority_rank(priority), self._seq, item))
        self._seq += 1


__all__ = [
    "QueueClosed",
    "TriggerQueue",
    "TriggerReceipt",
    "TriggerTerminated",
]
