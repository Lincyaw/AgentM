"""Simplified EventBus for v2.

Key difference from v1: events are frozen dataclasses.  Handlers
express intent through return values, not mutation.  The bus wraps each
emission in an Envelope that carries the dispatch id — the event object
itself is never written to.

No readonly-field snapshot/restore machinery needed.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class Event:
    """Base for all v2 events.  Frozen — handlers cannot mutate."""

    pass


@dataclass(frozen=True, slots=True)
class Envelope:
    """Wraps an Event emission with a unique dispatch id."""

    dispatch_id: str
    channel: str
    event: Any


@dataclass(slots=True)
class _Subscription:
    priority: int
    seq: int
    handler: Handler
    owner: str | None = None


def _sub_key(sub: _Subscription) -> tuple[int, int]:
    return (sub.priority, sub.seq)


class BusPriority:
    PRE: int = 100
    NORMAL: int = 500
    POST: int = 900


@dataclass(slots=True)
class EventBus:
    """Channel-keyed pub/sub with priority-ordered dispatch."""

    _handlers: dict[str, list[_Subscription]] = field(default_factory=dict)
    _next_seq: int = 0
    _frozen_clear: bool = False

    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
        owner: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe ``handler`` to ``channel``; return an unsubscribe fn."""

        import bisect

        sub = _Subscription(
            priority=priority, seq=self._next_seq, handler=handler, owner=owner
        )
        self._next_seq += 1
        subs = self._handlers.setdefault(channel, [])
        bisect.insort(subs, sub, key=_sub_key)

        def unsubscribe() -> None:
            channel_subs = self._handlers.get(channel)
            if channel_subs is None:
                return
            for idx, existing in enumerate(channel_subs):
                if existing is sub:
                    del channel_subs[idx]
                    return

        return unsubscribe

    async def emit(self, channel: str, event: Any) -> list[Any]:
        """Dispatch ``event`` to all handlers on ``channel`` in priority order.

        Returns handler return values.  Handler exceptions are logged and
        swallowed; the corresponding slot holds ``None``.
        """

        subs = self._handlers.get(channel)
        if not subs:
            return []

        # Snapshot to guard against handlers mutating subscriptions
        snapshot = list(subs)
        results: list[Any] = []
        for sub in snapshot:
            try:
                value = sub.handler(event)
                if inspect.isawaitable(value):
                    value = await value
            except Exception:
                logger.exception(
                    "v2 event handler raised on channel {!r}; suppressing.",
                    channel,
                )
                value = None
            results.append(value)
        return results

    def emit_sync(self, channel: str, event: Any) -> list[Any]:
        """Synchronous dispatch — skips async handlers."""

        subs = self._handlers.get(channel)
        if not subs:
            return []

        # Snapshot to guard against handlers mutating subscriptions
        snapshot = list(subs)
        results: list[Any] = []
        for sub in snapshot:
            try:
                value = sub.handler(event)
                if inspect.isawaitable(value):
                    if hasattr(value, "close"):
                        value.close()
                    logger.warning(
                        "async handler on {!r} skipped during emit_sync",
                        channel,
                    )
                    value = None
            except Exception:
                logger.exception(
                    "v2 event handler raised on channel {!r}; suppressing.",
                    channel,
                )
                value = None
            results.append(value)
        return results

    def freeze_clear(self) -> None:
        """Block clear() from wiping handlers.  Called after atom install."""
        self._frozen_clear = True

    def clear(self) -> None:
        """Clear all handlers.  Blocked after freeze_clear()."""
        if self._frozen_clear:
            logger.warning(
                "EventBus.clear() ignored — bus is frozen; "
                "only Session.shutdown() may clear"
            )
            return
        self._handlers.clear()

    def _force_clear(self) -> None:
        """Unconditional clear — for Session.shutdown() only."""
        self._frozen_clear = False
        self._handlers.clear()


class EventBusObserver:
    """Observer protocol for bus-level instrumentation.

    The observability atom subclasses this to record dispatch spans
    and handler invocations without coupling to bus internals.
    """

    def on_emit_start(self, channel: str, event: Any) -> None: ...
    def on_handler_start(self, channel: str, handler: Handler, event: Any) -> None: ...
    def on_handler_done(
        self, channel: str, handler: Handler, event: Any, result: Any
    ) -> None: ...
    def on_emit_end(self, channel: str, event: Any, results: list[Any]) -> None: ...


__all__ = [
    "BusPriority",
    "Envelope",
    "Event",
    "EventBus",
    "EventBusObserver",
    "Handler",
]
