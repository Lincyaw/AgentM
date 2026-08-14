"""Immutable-event dispatch bus.

Events are frozen dataclasses. Handlers express intent through return values,
not mutation. Each emission gets a dispatch id passed to observers; the event
object itself is never written to.

No readonly-field snapshot/restore machinery needed.
"""
# code-health: ignore-file[AM022] -- heterogeneous event dispatch boundary

from __future__ import annotations

import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]
EventReducer = Callable[[Any, Any], Any]


def _keep_event(event: Any, _: Any) -> Any:
    return event


@dataclass(frozen=True, slots=True)
class Event:
    """Base for frozen events that handlers cannot mutate."""


@dataclass(slots=True)
class _Subscription:
    priority: int
    seq: int
    handler: Handler
    owner: str | None = None


@dataclass(slots=True)
class _ObserverRecord:
    observer: EventBusObserver
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
    _observers: list[_ObserverRecord] = field(default_factory=list)
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

    def remove_owner(self, owner: str) -> int:
        """Drop every subscription and observer made by ``owner``; return how many.

        Each subscription already carries the atom that made it, so replacing
        one atom with a newer version does not need the unsubscribe closures
        that atom never kept. An observer sees every dispatch on the bus, so
        one left behind by an uninstalled atom is the loudest kind of leak.
        """

        removed = 0
        for channel, subs in list(self._handlers.items()):
            kept = [sub for sub in subs if sub.owner != owner]
            removed += len(subs) - len(kept)
            if kept:
                self._handlers[channel] = kept
            else:
                del self._handlers[channel]
        kept_observers = [record for record in self._observers if record.owner != owner]
        removed += len(self._observers) - len(kept_observers)
        self._observers = kept_observers
        return removed

    def add_observer(
        self,
        observer: EventBusObserver,
        *,
        owner: str | None = None,
    ) -> Callable[[], None]:
        """Attach a bus observer; return an unsubscribe function."""

        record = _ObserverRecord(observer=observer, owner=owner)
        self._observers.append(record)

        def unsubscribe() -> None:
            try:
                self._observers.remove(record)
            except ValueError:
                return

        return unsubscribe

    async def emit(self, channel: str, event: Any) -> list[Any]:
        """Dispatch ``event`` to all handlers on ``channel`` in priority order.

        Returns handler return values.  Handler exceptions are logged and
        swallowed; the corresponding slot holds ``None``.
        """

        dispatch_id = uuid.uuid4().hex
        self._observer_emit_start(channel, event, dispatch_id)
        subs = self._handlers.get(channel)
        if not subs:
            self._observer_emit_end(channel, event, [], dispatch_id)
            return []

        # Snapshot to guard against handlers mutating subscriptions
        snapshot = list(subs)
        results: list[Any] = []
        for sub in snapshot:
            start_ns = time.perf_counter_ns()
            self._observer_handler_start(
                channel, sub.handler, event, dispatch_id, sub.owner
            )
            error: BaseException | None = None
            try:
                value = sub.handler(event)
                if inspect.isawaitable(value):
                    value = await value
            except Exception as exc:
                error = exc
                logger.exception(
                    "event handler raised on channel {!r}; suppressing.",
                    channel,
                )
                value = None
            duration_ns = time.perf_counter_ns() - start_ns
            self._observer_handler_done(
                channel,
                sub.handler,
                event,
                value,
                error,
                duration_ns,
                dispatch_id,
                sub.owner,
            )
            results.append(value)
        self._observer_emit_end(channel, event, results, dispatch_id)
        return results

    async def emit_decision(self, channel: str, event: Any) -> list[Any]:
        """Dispatch a return-value-carrying event and propagate handler errors.

        Every handler receives the original immutable event. Use this for
        gates, rewrites, replacements, and decisions where treating a crashed
        handler as ``None`` would silently change policy into abstention.
        """

        _, results = await self.emit_reduced(channel, event, _keep_event)
        return results

    async def emit_reduced(
        self,
        channel: str,
        event: Any,
        reducer: EventReducer,
    ) -> tuple[Any, list[Any]]:
        """Dispatch a transform event and feed each result to the next handler.

        Frozen events remain immutable: ``reducer`` creates the next event from
        the current event and one handler result. Observation-only handlers may
        return ``None``. Unlike ``emit``, a handler exception PROPAGATES to the
        caller: a transform chain is decision-carrying, and a crashed link must
        not silently degrade into an abstention. Reducer errors propagate too.
        """

        dispatch_id = uuid.uuid4().hex
        initial_event = event
        self._observer_emit_start(channel, initial_event, dispatch_id)
        subs = self._handlers.get(channel)
        if not subs:
            self._observer_emit_end(channel, initial_event, [], dispatch_id)
            return event, []

        snapshot = list(subs)
        results: list[Any] = []
        try:
            for sub in snapshot:
                handler_event = event
                start_ns = time.perf_counter_ns()
                self._observer_handler_start(
                    channel,
                    sub.handler,
                    handler_event,
                    dispatch_id,
                    sub.owner,
                )
                value: Any = None
                error: BaseException | None = None
                try:
                    value = sub.handler(handler_event)
                    if inspect.isawaitable(value):
                        value = await value
                except Exception as exc:
                    error = exc
                    logger.exception(
                        "event handler raised on transform channel {!r}; propagating.",
                        channel,
                    )
                    raise
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    duration_ns = time.perf_counter_ns() - start_ns
                    self._observer_handler_done(
                        channel,
                        sub.handler,
                        handler_event,
                        None if error is not None else value,
                        error,
                        duration_ns,
                        dispatch_id,
                        sub.owner,
                    )
                results.append(value)
                if value is not None:
                    event = reducer(event, value)
        finally:
            self._observer_emit_end(channel, initial_event, results, dispatch_id)
        return event, results

    def emit_sync(self, channel: str, event: Any) -> list[Any]:
        """Synchronous dispatch — skips async handlers."""

        dispatch_id = uuid.uuid4().hex
        self._observer_emit_start(channel, event, dispatch_id)
        subs = self._handlers.get(channel)
        if not subs:
            self._observer_emit_end(channel, event, [], dispatch_id)
            return []

        # Snapshot to guard against handlers mutating subscriptions
        snapshot = list(subs)
        results: list[Any] = []
        for sub in snapshot:
            start_ns = time.perf_counter_ns()
            self._observer_handler_start(
                channel, sub.handler, event, dispatch_id, sub.owner
            )
            error: BaseException | None = None
            try:
                value = sub.handler(event)
                if inspect.isawaitable(value):
                    if inspect.iscoroutine(value):
                        value.close()
                    logger.warning(
                        "async handler on {!r} skipped during emit_sync",
                        channel,
                    )
                    value = None
            except Exception as exc:
                error = exc
                logger.exception(
                    "event handler raised on channel {!r}; suppressing.",
                    channel,
                )
                value = None
            duration_ns = time.perf_counter_ns() - start_ns
            self._observer_handler_done(
                channel,
                sub.handler,
                event,
                value,
                error,
                duration_ns,
                dispatch_id,
                sub.owner,
            )
            results.append(value)
        self._observer_emit_end(channel, event, results, dispatch_id)
        return results

    def freeze_clear(self) -> None:
        """Block clear() from wiping handlers.  Called after atom install."""
        self._frozen_clear = True

    def copy(self) -> EventBus:
        """Snapshot subscriptions and observers for composition transactions."""

        copied = EventBus()
        copied._handlers = {
            channel: list(subscriptions)
            for channel, subscriptions in self._handlers.items()
        }
        copied._observers = list(self._observers)
        copied._next_seq = self._next_seq
        copied._frozen_clear = self._frozen_clear
        return copied

    def replace_from(self, other: EventBus) -> None:
        """Restore subscriptions and observers from ``other``."""

        self._handlers = {
            channel: list(subscriptions)
            for channel, subscriptions in other._handlers.items()
        }
        self._observers = list(other._observers)
        self._next_seq = other._next_seq
        self._frozen_clear = other._frozen_clear

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
        self._observers.clear()

    def _observer_emit_start(self, channel: str, event: Any, dispatch_id: str) -> None:
        for record in list(self._observers):
            try:
                record.observer.on_emit_start(channel, event, dispatch_id)
            except Exception:
                logger.debug("event bus observer on_emit_start failed")

    def _observer_handler_start(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        dispatch_id: str,
        owner: str | None,
    ) -> None:
        for record in list(self._observers):
            try:
                record.observer.on_handler_start(
                    channel, handler, event, dispatch_id, owner
                )
            except Exception:
                logger.debug("event bus observer on_handler_start failed")

    def _observer_handler_done(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
        dispatch_id: str,
        owner: str | None,
    ) -> None:
        for record in list(self._observers):
            try:
                record.observer.on_handler_done(
                    channel,
                    handler,
                    event,
                    result,
                    error,
                    duration_ns,
                    dispatch_id,
                    owner,
                )
            except Exception:
                logger.debug("event bus observer on_handler_done failed")

    def _observer_emit_end(
        self,
        channel: str,
        event: Any,
        results: list[Any],
        dispatch_id: str,
    ) -> None:
        for record in list(self._observers):
            try:
                record.observer.on_emit_end(channel, event, results, dispatch_id)
            except Exception:
                logger.debug("event bus observer on_emit_end failed")


class EventBusObserver:
    """Observer protocol for bus-level instrumentation.

    The observability atom subclasses this to record dispatch spans
    and handler invocations without coupling to bus internals.
    """

    def on_emit_start(self, channel: str, event: Any, dispatch_id: str) -> None: ...
    def on_handler_start(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        dispatch_id: str,
        owner: str | None,
    ) -> None: ...
    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
        dispatch_id: str,
        owner: str | None,
    ) -> None: ...
    def on_emit_end(
        self, channel: str, event: Any, results: list[Any], dispatch_id: str
    ) -> None: ...


__all__ = [
    "BusPriority",
    "Event",
    "EventBus",
    "EventBusObserver",
    "EventReducer",
    "Handler",
]
