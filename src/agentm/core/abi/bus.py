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
class BusSegment:
    """One context's own subscriptions, dispatched as part of a bus.

    An atom subscribes into its own segment and the session dispatches to it
    because the segment is *linked* into the bus.  Detaching the atom unlinks
    the segment, so its handlers stop being reached without anything being
    removed from them — and a rollback that relinks it puts every handler back
    exactly where it dispatched from, because position is ``(priority, seq)``
    and both were fixed when the subscription was made.

    ``seq`` comes from the bus rather than from the segment: dispatch order is
    a property of the whole bus, and two segments numbering themselves
    independently would interleave by accident.
    """

    _bus: EventBus
    owner: str | None = None
    handlers: dict[str, list[_Subscription]] = field(default_factory=dict)
    observers: list[_ObserverRecord] = field(default_factory=list)

    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]:
        """Subscribe into this segment; return an unsubscribe fn."""

        import bisect

        sub = _Subscription(
            priority=priority,
            seq=self._bus._take_seq(),
            handler=handler,
            owner=self.owner,
        )
        subs = self.handlers.setdefault(channel, [])
        bisect.insort(subs, sub, key=_sub_key)

        def unsubscribe() -> None:
            channel_subs = self.handlers.get(channel)
            if channel_subs is None:
                return
            for idx, existing in enumerate(channel_subs):
                if existing is sub:
                    del channel_subs[idx]
                    return

        return unsubscribe

    def add_observer(self, observer: EventBusObserver) -> Callable[[], None]:
        """Attach an observer into this segment; return an unsubscribe fn."""

        record = _ObserverRecord(observer=observer, owner=self.owner)
        self.observers.append(record)

        def unsubscribe() -> None:
            try:
                self.observers.remove(record)
            except ValueError:
                return

        return unsubscribe

    def take(self) -> BusSegment:
        """Move every attachment into a fresh segment, leaving this empty."""

        moved = BusSegment(_bus=self._bus, owner=self.owner)
        moved.handlers = self.handlers
        moved.observers = self.observers
        self.handlers = {}
        self.observers = []
        return moved

    def give(self, other: BusSegment) -> None:
        """Take every attachment back from a segment ``take`` moved it into."""

        for channel, subs in other.handlers.items():
            kept = self.handlers.setdefault(channel, [])
            kept[:0] = subs
            kept.sort(key=_sub_key)
        self.observers[:0] = other.observers
        other.handlers = {}
        other.observers = []


@dataclass(slots=True)
class EventBus:
    """Channel-keyed pub/sub with priority-ordered dispatch.

    What it dispatches is the union of its own subscriptions and those of every
    linked ``BusSegment``, ordered by ``(priority, seq)`` across all of them —
    the same total order a single table gave, since ``seq`` is allocated here.
    """

    _handlers: dict[str, list[_Subscription]] = field(default_factory=dict)
    _observers: list[_ObserverRecord] = field(default_factory=list)
    _next_seq: int = 0
    _frozen_clear: bool = False
    _linked: list[BusSegment] = field(default_factory=list)

    # --- Segments ---

    def _take_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    def segment(self, owner: str | None = None) -> BusSegment:
        """Mint a segment that dispatches in this bus's order once linked."""

        return BusSegment(_bus=self, owner=owner)

    def link(self, segment: BusSegment) -> None:
        """Dispatch to ``segment``'s attachments as well as this bus's own."""

        if any(existing is segment for existing in self._linked):
            return
        self._linked.append(segment)

    def unlink(self, segment: BusSegment) -> None:
        """Stop dispatching to ``segment``; safe to repeat."""

        self._linked = [
            existing for existing in self._linked if existing is not segment
        ]

    def subscriptions(self, channel: str) -> list[_Subscription]:
        """Everything ``channel`` dispatches to, in dispatch order."""

        own = self._handlers.get(channel)
        if not self._linked:
            return list(own) if own else []
        merged: list[_Subscription] = list(own) if own else []
        for segment in self._linked:
            linked = segment.handlers.get(channel)
            if linked:
                merged.extend(linked)
        merged.sort(key=_sub_key)
        return merged

    def channels(self) -> list[str]:
        """Every channel with at least one subscription anywhere on the bus."""

        names = set(self._handlers)
        for segment in self._linked:
            names.update(segment.handlers)
        return sorted(names)

    def all_observers(self) -> list[_ObserverRecord]:
        """Every observer, this bus's own first and then each linked segment's."""

        if not self._linked:
            return list(self._observers)
        records = list(self._observers)
        for segment in self._linked:
            records.extend(segment.observers)
        return records

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
            priority=priority, seq=self._take_seq(), handler=handler, owner=owner
        )
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
        # Already a fresh list, which is also the guard against a handler
        # mutating subscriptions while this dispatch walks them.
        snapshot = self.subscriptions(channel)
        if not snapshot:
            self._observer_emit_end(channel, event, [], dispatch_id)
            return []

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
        snapshot = self.subscriptions(channel)
        if not snapshot:
            self._observer_emit_end(channel, initial_event, [], dispatch_id)
            return event, []

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
        # Already a fresh list, which is also the guard against a handler
        # mutating subscriptions while this dispatch walks them.
        snapshot = self.subscriptions(channel)
        if not snapshot:
            self._observer_emit_end(channel, event, [], dispatch_id)
            return []

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
        """Snapshot subscriptions and observers for composition transactions.

        This bus's *own* subscriptions.  Which segments are linked is not part
        of it: a segment is linked and unlinked by whoever owns it, together
        with that owner's other two link lists, and a restore that put a list
        of segments back would undo linking somebody else did in between while
        the two lists that move with it stayed as they were.
        """

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
        """Restore subscriptions and observers from ``other``, links untouched.

        ``_next_seq`` is deliberately not restored.  It is an allocator, not
        state: a sequence number it has handed out is held by the subscription
        that got it, and a linked segment's subscriptions outlive this call.
        Rewinding it re-issues numbers that are still in use, two live
        subscriptions tie on ``(priority, seq)``, and the one subscribed later
        dispatches first.  Handing out a number nobody holds costs nothing.
        """

        self._handlers = {
            channel: list(subscriptions)
            for channel, subscriptions in other._handlers.items()
        }
        self._observers = list(other._observers)
        self._frozen_clear = other._frozen_clear

    def clear(self) -> None:
        """Clear all handlers, linked segments included.  Blocked after freeze_clear().

        A linked segment's handlers are handlers this bus dispatches, so
        leaving one linked would leave behind exactly what this promises to
        remove.  A segment is linked or not as a whole, so its observers go
        with it; this bus's own observers stay, as they always have.

        Unlinking here is not coordinated with whoever linked the segments: a
        session holds the same links in its context list and its service
        registry, and clearing the bus does not touch those.  This is the bus's
        own operation, and a composed session freezes it at ``start()`` for
        that reason.
        """
        if self._frozen_clear:
            logger.warning(
                "EventBus.clear() ignored — bus is frozen; "
                "only Session.shutdown() may clear"
            )
            return
        self._handlers.clear()
        self._linked.clear()

    def _force_clear(self) -> None:
        """Unconditional clear — for Session.shutdown() only."""
        self._frozen_clear = False
        self._handlers.clear()
        self._observers.clear()
        self._linked.clear()

    def _observer_emit_start(self, channel: str, event: Any, dispatch_id: str) -> None:
        for record in self.all_observers():
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
        for record in self.all_observers():
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
        for record in self.all_observers():
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
        for record in self.all_observers():
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
    "BusSegment",
    "Event",
    "EventBus",
    "EventBusObserver",
    "EventReducer",
    "Handler",
]
