"""Cancellation signal ABI.

SDK hosts, provider adapters, tools, and operation backends only need a small
cooperative cancellation surface: poll ``is_set()`` or await ``wait()``. The
runtime may back this with one event, multiple composed events, or a host-side
controller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

CancelReason = Literal[
    "user_cancel",
    "submit_interrupt",
    "shutdown",
    "sibling_error",
    "task_stop",
    "unknown",
]


@runtime_checkable
class CancelSignal(Protocol):
    """Cooperative cancellation signal shared across SDK boundaries."""

    def is_set(self) -> bool:
        """Return whether cancellation has been requested."""
        ...

    async def wait(self) -> object:
        """Block until cancellation is requested."""
        ...


@runtime_checkable
class CancelSource(CancelSignal, Protocol):
    """Mutable cancellation source owned by the component that can abort work."""

    def set(self, reason: CancelReason | str | None = "unknown") -> None:
        """Request cancellation and preserve the cause across boundaries."""
        ...


@runtime_checkable
class ResettableCancelSource(CancelSource, Protocol):
    """Mutable cancellation source that can be re-used across turns."""

    def clear(self) -> None:
        """Clear cancellation for a later turn."""
        ...


class EventCancelSource:
    """Small asyncio-backed cancellation source with optional reason."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: CancelReason | str | None = None

    @property
    def reason(self) -> CancelReason | str | None:
        return self._reason

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> object:
        return await self._event.wait()

    def set(self, reason: CancelReason | str | None = "unknown") -> None:
        self._reason = reason
        self._event.set()

    def clear(self) -> None:
        self._reason = None
        self._event.clear()


class CompositeCancelSignal:
    """Read-only cancellation signal composed from independent owners."""

    def __init__(self, *signals: CancelSignal | None) -> None:
        self._signals: tuple[CancelSignal, ...] = tuple(
            signal for signal in signals if signal is not None
        )

    @property
    def reason(self) -> CancelReason | str | None:
        for signal in self._signals:
            if signal.is_set():
                return cancel_reason(signal) or "unknown"
        return None

    @property
    def signals(self) -> Sequence[CancelSignal]:
        return self._signals

    def is_set(self) -> bool:
        return any(signal.is_set() for signal in self._signals)

    async def wait(self) -> object:
        if self.is_set() or not self._signals:
            return None
        waiters = [
            asyncio.create_task(signal.wait())
            for signal in self._signals
        ]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
        return None


def cancel_reason(signal: CancelSignal | None) -> CancelReason | str | None:
    """Return a signal's reason when the implementation exposes one."""

    if signal is None:
        return None
    reason = getattr(signal, "reason", None)
    if callable(reason):
        try:
            value = reason()
        except TypeError:
            return None
    else:
        value = reason
    return value if isinstance(value, str) else None


__all__ = [
    "CancelReason",
    "CancelSignal",
    "CancelSource",
    "CompositeCancelSignal",
    "EventCancelSource",
    "ResettableCancelSource",
    "cancel_reason",
]
