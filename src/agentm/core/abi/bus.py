"""Channel-keyed event bus and dispatch primitives.

Split from ``events.py`` so the event dataclasses stay transport-agnostic
while the bus machinery (subscriptions, priority dispatch, observers)
lives in its own module. Both modules are part of the ABI surface; atoms
import bus symbols through ``agentm.core.abi``.
"""

from __future__ import annotations

import bisect
import inspect
import time
import uuid
from copy import deepcopy
from collections.abc import Awaitable, Callable
from dataclasses import (
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
)
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from loguru import logger

from .events import BusPriority, Event

if TYPE_CHECKING:
    from .events import (
        AgentEndEvent,
        AgentStartEvent,
        BeforeSendToLlmEvent,
        ContextEvent,
        DecideTurnActionEvent,
        StreamDeltaEvent,
        ToolCallEvent,
        ToolResultEvent,
        TurnEndEvent,
        TurnStartEvent,
    )


# A handler may be sync or async; it returns anything (the bus collects).
Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]
ObserverCallback = Callable[[str, Any], None]


class EventBusObserver(Protocol):
    """Optional sidecar that observes every ``EventBus.emit`` dispatch.

    The bus invokes these methods synchronously inside ``emit``/``emit_sync``;
    any exception they raise is logged and swallowed so observers cannot
    affect handler outputs.
    """

    def on_emit_start(self, channel: str, event: Any) -> None: ...

    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
        owner: str | None,
    ) -> None: ...

    def on_emit_end(self, channel: str, event: Any, results: list[Any]) -> None: ...


ObserverRegistration = ObserverCallback | EventBusObserver


@dataclass(frozen=True, slots=True)
class _Subscription:
    """One handler registration with its dispatch-order key.

    ``priority`` is the tier (lower runs earlier). ``seq`` is a monotonic
    counter assigned at subscribe time so two same-priority handlers are
    ordered by registration (FIFO within tier). Sorting is purely on
    ``(priority, seq)``; ``handler`` is the payload the bus invokes.

    ``owner`` is the atom module-path that registered this handler (stamped
    by ``_ExtensionAPIImpl.on`` from its own identity). It travels with the
    registration — not the handler object — so observation attribution and
    reload position-restore work for bound methods and builtins alike.
    """

    priority: int
    seq: int
    handler: Handler
    owner: str | None = None


def _sub_key(sub: _Subscription) -> tuple[int, int]:
    return (sub.priority, sub.seq)


def _mutable_fields(event: Event) -> frozenset[str]:
    hook = getattr(type(event), "HOOK", None)
    return frozenset(getattr(hook, "mutable_fields", ()))


@dataclass(frozen=True, slots=True)
class _ReadonlyFieldSnapshot:
    original: Any
    frozen: Any
    restore: Any


def _freeze_value(value: Any, seen: set[int] | None = None) -> Any:
    """Build an equality-safe structural fingerprint for event payloads.

    Provider and client objects often use identity equality, so comparing them
    with a deep copy creates false mutation reports. Containers and dataclasses
    are traversed; opaque leaves retain their original identity.
    """

    if value is None or isinstance(value, (bool, int, str, bytes)):
        return ("literal", type(value), value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, Enum):
        return ("enum", type(value), value.name)

    identity = id(value)
    active = seen if seen is not None else set()
    if identity in active:
        return ("cycle", identity)
    active.add(identity)
    try:
        if isinstance(value, list):
            return ("list", tuple(_freeze_value(item, active) for item in value))
        if isinstance(value, tuple):
            return ("tuple", tuple(_freeze_value(item, active) for item in value))
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (
                        _freeze_value(key, active),
                        _freeze_value(item, active),
                    )
                    for key, item in value.items()
                ),
            )
        if isinstance(value, (set, frozenset)):
            return (
                type(value).__name__,
                frozenset(_freeze_value(item, active) for item in value),
            )
        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type(value),
                tuple(
                    (
                        item.name,
                        _freeze_value(getattr(value, item.name), active),
                    )
                    for item in dataclass_fields(value)
                ),
            )
        return ("opaque", type(value), identity)
    finally:
        active.remove(identity)


def _snapshot_readonly_fields(
    event: Any,
) -> dict[str, _ReadonlyFieldSnapshot] | None:
    if not isinstance(event, Event):
        return None
    mutable = _mutable_fields(event)
    event_fields = {item.name for item in dataclass_fields(event)}
    unknown = mutable - event_fields
    if unknown:
        raise RuntimeError(
            f"{type(event).__name__}.HOOK declares unknown mutable fields: "
            f"{sorted(unknown)}"
        )
    snapshots: dict[str, _ReadonlyFieldSnapshot] = {}
    for item in dataclass_fields(event):
        if item.name == "dispatch_id" or item.name in mutable:
            continue
        original = getattr(event, item.name)
        try:
            restore = deepcopy(original)
        except Exception as exc:
            # Opaque leaves are identity-compared. If a handler replaces the
            # field, retaining the original object is sufficient to restore it.
            logger.debug(
                "EventBus could not clone readonly field {}.{}; "
                "falling back to identity restoration: {}",
                type(event).__name__,
                item.name,
                exc,
            )
            restore = original
        snapshots[item.name] = _ReadonlyFieldSnapshot(
            original=original,
            frozen=_freeze_value(original),
            restore=restore,
        )
    return snapshots


def _readonly_mutation_error(
    event: Any,
    before: dict[str, _ReadonlyFieldSnapshot] | None,
) -> str | None:
    if before is None:
        return None
    changed: list[str] = []
    for name, snapshot in before.items():
        try:
            current = getattr(event, name)
            is_changed = (
                current is not snapshot.original
                or _freeze_value(current) != snapshot.frozen
            )
        except Exception as exc:
            raise RuntimeError(
                f"could not compare readonly field {type(event).__name__}.{name}"
            ) from exc
        if is_changed:
            changed.append(name)
    if not changed:
        return None
    for name in changed:
        setattr(event, name, before[name].restore)
    return (
        f"handler mutated undeclared readonly fields on "
        f"{type(event).__name__}: {changed}; the fields were restored. Declare them in "
        "HookContract.mutable_fields or stop mutating them"
    )


@dataclass(slots=True)
class EventBus:
    """Minimal channel-keyed pub/sub. See module docstring for the contract."""

    _handlers: dict[str, list[_Subscription]] = field(default_factory=dict)
    # Per-channel handler-only list, regenerated lazily after any
    # ``on``/``unsubscribe`` mutation. Avoids rebuilding ``[s.handler for s
    # in subs]`` on every emit — load-bearing on hot channels like
    # ``stream_delta`` (one emission per provider chunk).
    _handler_cache: dict[str, list[tuple[Handler, str | None]]] = field(
        default_factory=dict
    )
    _observer: EventBusObserver | None = None
    _observer_callbacks: list[ObserverRegistration] = field(default_factory=list)
    _strict_sync_handlers: bool = False
    _strict_event_mutations: bool = True
    _next_seq: int = 0

    def set_observer(self, observer: EventBusObserver | None) -> None:
        """Install (or clear) a single observer. The bus invokes its hooks
        from inside ``emit``; observer exceptions are logged and swallowed.
        Only one observer at a time — second call replaces the first.
        """
        self._observer = observer

    def add_observer(self, callback: ObserverRegistration) -> Callable[[], None]:
        """Observe every emit and return an idempotent unsubscribe fn."""

        self._observer_callbacks.append(callback)

        def unsubscribe() -> None:
            try:
                self._observer_callbacks.remove(callback)
            except ValueError:
                return

        return unsubscribe

    def set_strict_sync(self, strict: bool) -> None:
        """If True, ``emit_sync`` raises ``RuntimeError`` when it encounters
        an async handler instead of silently skipping it. Use during
        development to surface mistakes; off by default for production.
        """
        self._strict_sync_handlers = strict

    def set_strict_event_mutations(self, strict: bool) -> None:
        """Fail dispatch when a handler mutates an undeclared event field.

        Typed events are checked by default. The bus deep-snapshots every
        field not listed in ``HookContract.mutable_fields`` and restores it
        before failing. For observation-only events, every payload field is
        therefore read-only. Disable only for targeted profiling or
        compatibility diagnostics.
        """
        self._strict_event_mutations = strict

    # Typed overloads for kernel-owned channels. Runtime-level channels
    # (``before_agent_start``, ``session_shutdown``, ``before_compact``,
    # ``after_compact``, ``child_session_*``, ``cost_budget_exceeded``,
    # ``plan_submitted``, ``session_ready``) fall through to the ``str``
    # fallback to preserve the layer rule (kernel does not import runtime).
    # Extensions may also invent their own channels — the ``str`` fallback
    # also preserves that escape hatch.
    @overload
    def on(
        self,
        channel: Literal["agent_start"],
        handler: Callable[[AgentStartEvent], Any]
        | Callable[[AgentStartEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["agent_end"],
        handler: Callable[[AgentEndEvent], Any]
        | Callable[[AgentEndEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["decide_turn_action"],
        handler: Callable[[DecideTurnActionEvent], Any]
        | Callable[[DecideTurnActionEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["turn_start"],
        handler: Callable[[TurnStartEvent], Any]
        | Callable[[TurnStartEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["turn_end"],
        handler: Callable[[TurnEndEvent], Any]
        | Callable[[TurnEndEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["tool_call"],
        handler: Callable[[ToolCallEvent], Any]
        | Callable[[ToolCallEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["tool_result"],
        handler: Callable[[ToolResultEvent], Any]
        | Callable[[ToolResultEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["before_send_to_llm"],
        handler: Callable[[BeforeSendToLlmEvent], Any]
        | Callable[[BeforeSendToLlmEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["context"],
        handler: Callable[[ContextEvent], Any]
        | Callable[[ContextEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["stream_delta"],
        handler: Callable[[StreamDeltaEvent], Any]
        | Callable[[StreamDeltaEvent], Awaitable[Any]],
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
        owner: str | None = None,
    ) -> Callable[[], None]: ...
    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
        owner: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe ``handler`` to ``channel``; return an unsubscribe fn.

        Calling the returned function removes this exact handler. Calling it
        a second time is a no-op (idempotent). ``priority`` selects the
        dispatch tier — see :class:`BusPriority`. Within a tier, registration
        order is preserved (FIFO). ``owner`` records the registering atom so
        observers and the reloader can attribute the subscription; it is
        stamped automatically by ``_ExtensionAPIImpl.on`` and need not be
        supplied by other callers.
        """

        sub = _Subscription(
            priority=priority, seq=self._next_seq, handler=handler, owner=owner
        )
        self._next_seq += 1
        bisect.insort(
            self._handlers.setdefault(channel, []),
            sub,
            key=_sub_key,
        )
        self._handler_cache.pop(channel, None)

        def unsubscribe() -> None:
            handlers = self._handlers.get(channel)
            if handlers is None:
                return
            for idx, existing in enumerate(handlers):
                if existing is sub:
                    del handlers[idx]
                    self._handler_cache.pop(channel, None)
                    return

        return unsubscribe

    def subscriptions_for(self, channel: str) -> list[_Subscription]:
        """Return a fresh shallow copy of subscriptions on ``channel`` in
        dispatch order. Used by runtime internals (atom_reloader) that need
        to inspect or rearrange the live order — e.g. for the reload-time
        within-tier FIFO position-preservation pass. The returned list is a
        copy; mutating it does not affect the bus.
        """
        return list(self._handlers.get(channel, ()))

    def replace_subscriptions(
        self, channel: str, subscriptions: list[_Subscription]
    ) -> None:
        """Replace the bus's subscription list for ``channel`` with the
        caller-supplied list. Companion to :meth:`subscriptions_for` for
        runtime internals that need to splice handlers back in a specific
        order. The list is stored by reference; do not mutate after handing
        it over.
        """
        if subscriptions:
            self._handlers[channel] = subscriptions
        else:
            self._handlers.pop(channel, None)
        self._handler_cache.pop(channel, None)

    def channels(self) -> list[str]:
        """Return every channel name with at least one registered handler.
        Order is dict-insertion (not stable across runs)."""
        return list(self._handlers.keys())

    @overload
    async def emit(
        self, channel: Literal["agent_start"], event: AgentStartEvent
    ) -> list[Any]: ...
    @overload
    async def emit(
        self, channel: Literal["agent_end"], event: AgentEndEvent
    ) -> list[Any]: ...
    @overload
    async def emit(
        self,
        channel: Literal["decide_turn_action"],
        event: DecideTurnActionEvent,
    ) -> list[Any]: ...
    @overload
    async def emit(
        self, channel: Literal["turn_start"], event: TurnStartEvent
    ) -> list[Any]: ...
    @overload
    async def emit(
        self, channel: Literal["turn_end"], event: TurnEndEvent
    ) -> list[Any]: ...
    @overload
    async def emit(
        self, channel: Literal["tool_call"], event: ToolCallEvent
    ) -> list[Any]: ...
    @overload
    async def emit(
        self, channel: Literal["tool_result"], event: ToolResultEvent
    ) -> list[Any]: ...
    @overload
    async def emit(
        self, channel: Literal["before_send_to_llm"], event: BeforeSendToLlmEvent
    ) -> list[Any]: ...
    @overload
    async def emit(
        self, channel: Literal["context"], event: ContextEvent
    ) -> list[Any]: ...
    @overload
    async def emit(self, channel: str, event: Any) -> list[Any]: ...
    async def emit(self, channel: str, event: Any) -> list[Any]:
        """Dispatch ``event`` to all handlers on ``channel`` in order.

        Returns the list of handler return values (in registration order).
        Exceptions raised by individual handlers are logged and swallowed;
        the corresponding return slot holds ``None``.

        **Dispatch id.** On entry, the bus assigns a fresh
        ``uuid.uuid4().hex`` to ``event.dispatch_id`` when ``event`` is an
        :class:`Event` instance. Re-emitting the same instance therefore
        produces a fresh id — the field default on :class:`Event` is only
        there for events that are constructed but never emitted (e.g. test
        fixtures). Non-``Event`` payloads (raw dicts, extension-invented
        shapes) are dispatched as-is; consumers that need a join key on
        those payloads should use :class:`Event` subclasses.
        """

        handlers = self._handler_cache.get(channel)
        if handlers is None:
            registered = self._handlers.get(channel)
            handlers = [(sub.handler, sub.owner) for sub in (registered or ())]
            self._handler_cache[channel] = handlers
        observer_callbacks = tuple(self._observer_callbacks)
        observer = self._observer
        observe_handlers = observer is not None or any(
            not callable(callback) for callback in observer_callbacks
        )
        if isinstance(event, Event):
            event.dispatch_id = uuid.uuid4().hex
        if not handlers and observer is None and not observer_callbacks:
            return []
        self._safe_observe("on_emit_start", channel, event)
        results: list[Any] = []
        for h, owner in handlers:
            err: BaseException | None = None
            start_ns = time.perf_counter_ns() if observe_handlers else 0
            if observe_handlers:
                self._safe_observe("on_handler_start", channel, h, event)
            readonly_before = (
                _snapshot_readonly_fields(event)
                if self._strict_event_mutations
                else None
            )
            try:
                value = h(event)
                if inspect.isawaitable(value):
                    value = await value
            except Exception as exc:
                logger.exception(
                    f"Event handler raised on channel {channel!r}; suppressing."
                )
                err = exc
                value = None
            mutation_error = _readonly_mutation_error(event, readonly_before)
            if observe_handlers:
                self._safe_observe(
                    "on_handler_done",
                    channel,
                    h,
                    event,
                    value,
                    err,
                    time.perf_counter_ns() - start_ns,
                    owner,
                )
            if mutation_error is not None:
                raise RuntimeError(mutation_error)
            results.append(value)
        self._safe_observe("on_emit_end", channel, event, results)
        return results

    def _safe_observe(self, method: str, *args: Any) -> None:
        for callback in tuple(self._observer_callbacks):
            try:
                if callable(callback):
                    if method == "on_emit_start":
                        channel, event = args[0], args[1]
                        callback(channel, event)
                    continue
                observer_method = getattr(callback, method, None)
                if observer_method is None:
                    continue
                observer_method(*args)
            except Exception:
                if callable(callback):
                    logger.exception("EventBus observer callback raised; suppressing.")
                else:
                    logger.exception(f"EventBus observer.{method} raised; suppressing.")
        observer = self._observer
        if observer is None:
            return
        try:
            observer_method = getattr(observer, method, None)
            if observer_method is None:
                return
            observer_method(*args)
        except Exception:
            logger.exception(f"EventBus observer.{method} raised; suppressing.")

    def emit_sync(self, channel: str, event: Any) -> list[Any]:
        """Synchronous emit — runs only sync handlers; coroutine-returning
        handlers are logged-and-skipped.

        Exists so sync code paths (``ExtensionAPI.register_tool`` and friends,
        which cannot ``await``) can still publish events on the bus without
        forcing the API surface async. Observer hooks fire normally.

        Dispatch-id assignment mirrors :meth:`emit`: ``event.dispatch_id`` is
        overwritten with a fresh ``uuid.uuid4().hex`` when ``event`` is an
        :class:`Event` instance.
        """
        handlers = self._handler_cache.get(channel)
        if handlers is None:
            registered = self._handlers.get(channel)
            handlers = [(sub.handler, sub.owner) for sub in (registered or ())]
            self._handler_cache[channel] = handlers
        observer_callbacks = tuple(self._observer_callbacks)
        observer = self._observer
        observe_handlers = observer is not None or any(
            not callable(callback) for callback in observer_callbacks
        )
        if isinstance(event, Event):
            event.dispatch_id = uuid.uuid4().hex
        if not handlers and observer is None and not observer_callbacks:
            return []
        async_violation: tuple[str, Any] | None = None
        self._safe_observe("on_emit_start", channel, event)
        results: list[Any] = []
        for h, owner in handlers:
            err: BaseException | None = None
            start_ns = time.perf_counter_ns() if observe_handlers else 0
            if observe_handlers:
                self._safe_observe("on_handler_start", channel, h, event)
            readonly_before = (
                _snapshot_readonly_fields(event)
                if self._strict_event_mutations
                else None
            )
            try:
                value = h(event)
                if inspect.isawaitable(value):
                    if hasattr(value, "close"):
                        value.close()
                    if self._strict_sync_handlers and async_violation is None:
                        async_violation = (channel, h)
                    else:
                        logger.warning(
                            f"Async handler on channel {channel!r} skipped during emit_sync; use a sync handler or subscribe via an async-only channel."
                        )
                    value = None
            except Exception as exc:
                logger.exception(
                    f"Event handler raised on channel {channel!r}; suppressing."
                )
                err = exc
                value = None
            mutation_error = _readonly_mutation_error(event, readonly_before)
            if observe_handlers:
                self._safe_observe(
                    "on_handler_done",
                    channel,
                    h,
                    event,
                    value,
                    err,
                    time.perf_counter_ns() - start_ns,
                    owner,
                )
            if mutation_error is not None:
                raise RuntimeError(mutation_error)
            results.append(value)
        self._safe_observe("on_emit_end", channel, event, results)
        if async_violation is not None:
            ch, handler = async_violation
            raise RuntimeError(
                f"async handler {handler!r} on sync-only channel {ch!r}; "
                "use a sync handler or disable strict_sync mode"
            )
        return results

    def clear(self) -> None:
        """Remove every subscription on every channel."""

        self._handlers.clear()
        self._handler_cache.clear()


__all__ = [
    "BusPriority",
    "EventBus",
    "EventBusObserver",
    "Handler",
    "ObserverCallback",
    "ObserverRegistration",
]
