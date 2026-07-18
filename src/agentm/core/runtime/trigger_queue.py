"""TriggerQueue — unified entry point for all input sources.

Replaces v1's SessionInbox.  Every input source (user, background tool,
monitor, subagent) pushes a Trigger; the control loop consumes them via
``wait()``.  Background work is tracked so callers can block until the
session is quiescent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agentm.core.abi.trigger import ContinueTrigger, Trigger


class QueueClosed(Exception):
    """Raised by ``wait()`` after the queue has been closed."""


@dataclass(frozen=True, slots=True)
class _KickTrigger:
    source: str = "kick"


@dataclass(frozen=True, slots=True)
class _CloseSentinel:
    source: str = "close"


class TriggerQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._pending_work = 0
        self._quiescent = asyncio.Event()
        self._quiescent.set()

    def push(self, trigger: Trigger) -> None:
        self._queue.put_nowait(trigger)
        self._quiescent.clear()

    def kick(self) -> None:
        self._queue.put_nowait(_KickTrigger())
        self._quiescent.clear()

    def close(self) -> None:
        self._queue.put_nowait(_CloseSentinel())

    async def wait(self) -> Trigger:
        item = await self._queue.get()
        if isinstance(item, _CloseSentinel):
            raise QueueClosed
        self._update_quiescent()
        if isinstance(item, _KickTrigger):
            return ContinueTrigger()
        return item  # type: ignore[return-value]

    def is_empty(self) -> bool:
        return self._queue.empty()

    def note_work_started(self) -> None:
        self._pending_work += 1
        self._quiescent.clear()

    def note_work_finished(self) -> None:
        if self._pending_work > 0:
            self._pending_work -= 1
        self._update_quiescent()

    @property
    def has_pending_work(self) -> bool:
        return self._pending_work > 0

    async def wait_quiescent(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._quiescent.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True

    def _update_quiescent(self) -> None:
        if self._pending_work == 0 and self._queue.empty():
            self._quiescent.set()
        else:
            self._quiescent.clear()


__all__ = [
    "QueueClosed",
    "TriggerQueue",
]
