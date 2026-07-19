"""Cancellation signal ABI.

SDK hosts, provider adapters, tools, and operation backends only need a small
cooperative cancellation surface: poll ``is_set()`` or await ``wait()``. The
runtime may back this with one event, multiple composed events, or a host-side
controller.
"""

from __future__ import annotations

import asyncio
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

    def set(self) -> None:
        """Request cancellation."""
        ...


@runtime_checkable
class ReasonedCancelSignal(CancelSignal, Protocol):
    """Cancellation signal that exposes why it fired."""

    @property
    def reason(self) -> CancelReason | str | None:
        """Return the cancellation reason, when available."""
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
    "EventCancelSource",
    "ReasonedCancelSignal",
    "ResettableCancelSource",
    "cancel_reason",
]
