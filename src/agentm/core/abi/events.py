"""Kernel event bus and event taxonomy.

Implements §3.5 (Extension Bus) of
`.claude/designs/pluggable-architecture.md`. Conceptually a port of pi-mono's
``packages/coding-agent/src/core/event-bus.ts`` (33 lines) to async Python,
plus a minimal seed of typed event dataclasses used by ``loop.py``.

Dispatch contract:
- **Serial per channel.** Handlers registered on the same channel run in
  registration order; a later handler observes mutations made by earlier
  handlers. There is no ordering guarantee across channels.
- **Failures are isolated.** An exception from one handler is logged and
  swallowed; remaining handlers still run. One bad extension must not break
  the loop.
- **Sync and async handlers** are both accepted; coroutine returns are
  awaited.
- ``emit`` returns a list of handler return values (in registration order),
  so callers can inspect them for cancel/replace decisions.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

from .messages import AgentMessage, AssistantMessage
from .stream import Model
from .tool import Tool, ToolResult

logger = logging.getLogger(__name__)


# --- Event types ------------------------------------------------------------


class Event:
    """Marker base class for all kernel events.

    Concrete events are dataclasses; this class exists so callers can write
    ``isinstance(x, Event)`` if they want a structural check.
    """


@dataclass(slots=True, frozen=True)
class AgentStartEvent(Event):
    """Emitted once at the start of ``AgentLoop.run``."""

    messages: list[AgentMessage]


@dataclass(slots=True, frozen=True)
class AgentEndEvent(Event):
    """Emitted once when ``AgentLoop.run`` returns.

    ``stop_reason`` is one of: ``end_turn``, ``max_turns``, ``aborted``,
    ``error``.
    """

    messages: list[AgentMessage]
    stop_reason: str


@dataclass(slots=True)
class BeforeAgentEndEvent(Event):
    """Fires after a text-only assistant turn but before ``agent_end``.

    Handlers may return ``{\"cancel\": True, \"append\": list[AgentMessage]}``
    to keep the loop alive and inject new user-visible context for the next
    turn.
    """

    messages: list[AgentMessage]
    stop_reason: str


@dataclass(slots=True, frozen=True)
class TurnStartEvent(Event):
    """Emitted at the start of each loop turn (one LLM call)."""

    turn_index: int


@dataclass(slots=True, frozen=True)
class TurnEndEvent(Event):
    """Emitted after a turn's assistant message is fully assembled."""

    turn_index: int
    message: AssistantMessage


@dataclass(slots=True)
class ToolCallEvent(Event):
    """Emitted before a tool is executed.

    **Mutability contract**: this event is intentionally **not frozen**.
    Handlers may mutate ``args`` in place — later handlers and the actual
    tool ``execute`` will see the mutations. Handlers may also return
    ``{"block": True, "reason": str}`` from their handler function; the loop
    will short-circuit and emit a synthetic error tool result instead of
    running the tool.
    """

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolResultEvent(Event):
    """Emitted after a tool returns.

    **Replacement contract**: handlers may return a replacement
    :class:`ToolResult`; the last non-None replacement wins (see
    ``loop._collect_replacement``).
    """

    tool_call_id: str
    tool_name: str
    result: ToolResult


@dataclass(slots=True)
class BeforeSendToLlmEvent(Event):
    """Fires after context handlers have rewritten messages, immediately
    before the StreamFn is invoked. Handlers MAY mutate ``messages`` in
    place to make a final adjustment. This is the last hook before bytes
    leave the harness.

    The ``messages`` list is the same instance the loop will pass to
    ``StreamFn``; mutate it cautiously.

    Distinct from :class:`ContextEvent`: ``context`` may rewrite-by-return
    (replacement list) or in-place; this event fires *after* that resolution
    on the final ``messages`` list, so handlers like ``cost_budget`` see
    exactly what is about to hit the wire.
    """

    messages: list[AgentMessage]
    model: Model
    tools: list[Tool]
    system: str | None


@dataclass(slots=True, frozen=True)
class LlmRequestStartEvent(Event):
    """Emitted right before the loop drains ``stream_fn``."""

    turn_index: int
    message_count: int
    tool_count: int
    system_chars: int
    model_id: str | None


@dataclass(slots=True, frozen=True)
class LlmRequestEndEvent(Event):
    """Emitted after the loop finishes draining ``stream_fn`` (success or
    error). ``error`` is ``repr(exc)`` on failure, ``None`` on success.
    """

    turn_index: int
    chunk_count: int
    duration_ns: int
    error: str | None = None


@dataclass(slots=True, frozen=True)
class StreamDeltaEvent(Event):
    """One raw chunk from the provider stream, forwarded by the loop.

    Used by presenters (TUI, JSON tap) that want token-by-token output.
    The kernel still assembles the full ``AssistantMessage`` itself and
    publishes it via ``turn_end``; ``stream_delta`` is purely additive.

    ``delta`` is the same ``AssistantStreamEvent`` instance the loop
    received from ``stream_fn`` — typically a ``TextDelta``,
    ``ToolCallStart``, or ``MessageEnd``. Subscribers should pattern-match
    on the delta's type.
    """

    turn_index: int
    delta: Any  # AssistantStreamEvent — typed Any here to avoid pulling
    # the ``stream`` module into the events surface for everyone.


@dataclass(slots=True)
class ContextEvent(Event):
    """Emitted before each LLM call with the current message list.

    **Mutability contract**: ``messages`` is the list that will be sent to
    the model. Handlers may mutate it in place, and/or return a replacement
    ``list[AgentMessage]``; the last non-None replacement wins.
    """

    messages: list[AgentMessage]


@dataclass(slots=True, frozen=True)
class DiagnosticEvent(Event):
    """Non-fatal diagnostic from any subsystem during session construction
    or runtime. Emitted on the ``"diagnostic"`` channel. Used for failures
    that the recovery-floor invariant requires us to survive — scenario
    yaml load errors, skill/prompt loader hiccups, etc. The CLI subscribes
    and prints; only ``"error"`` level affects the exit code.
    """

    level: Literal["info", "warning", "error"]
    source: str
    message: str


# --- Bus --------------------------------------------------------------------


# A handler may be sync or async; it returns anything (the bus collects).
Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]


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
        result: Any,
        error: BaseException | None,
        duration_ns: int,
    ) -> None: ...

    def on_emit_end(
        self, channel: str, event: Any, results: list[Any]
    ) -> None: ...


@dataclass(slots=True)
class EventBus:
    """Minimal channel-keyed pub/sub. See module docstring for the contract."""

    _handlers: dict[str, list[Handler]] = field(default_factory=dict)
    _observer: EventBusObserver | None = None
    _strict_sync_handlers: bool = False

    def set_observer(self, observer: EventBusObserver | None) -> None:
        """Install (or clear) a single observer. The bus invokes its hooks
        from inside ``emit``; observer exceptions are logged and swallowed.
        Only one observer at a time — second call replaces the first.
        """
        self._observer = observer

    def set_strict_sync(self, strict: bool) -> None:
        """If True, ``emit_sync`` raises ``RuntimeError`` when it encounters
        an async handler instead of silently skipping it. Use during
        development to surface mistakes; off by default for production.
        """
        self._strict_sync_handlers = strict

    # Typed overloads for kernel-owned channels. Harness-level channels
    # (``before_agent_start``, ``session_shutdown``, ``before_compact``,
    # ``after_compact``, ``child_session_*``, ``cost_budget_exceeded``,
    # ``plan_submitted``, ``session_ready``) fall through to the ``str``
    # fallback to preserve the layer rule (kernel does not import harness).
    # Extensions may also invent their own channels — the ``str`` fallback
    # also preserves that escape hatch.
    @overload
    def on(
        self,
        channel: Literal["agent_start"],
        handler: Callable[[AgentStartEvent], Any]
        | Callable[[AgentStartEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["agent_end"],
        handler: Callable[[AgentEndEvent], Any]
        | Callable[[AgentEndEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["before_agent_end"],
        handler: Callable[[BeforeAgentEndEvent], Any]
        | Callable[[BeforeAgentEndEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["turn_start"],
        handler: Callable[[TurnStartEvent], Any]
        | Callable[[TurnStartEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["turn_end"],
        handler: Callable[[TurnEndEvent], Any]
        | Callable[[TurnEndEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["tool_call"],
        handler: Callable[[ToolCallEvent], Any]
        | Callable[[ToolCallEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["tool_result"],
        handler: Callable[[ToolResultEvent], Any]
        | Callable[[ToolResultEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["before_send_to_llm"],
        handler: Callable[[BeforeSendToLlmEvent], Any]
        | Callable[[BeforeSendToLlmEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["context"],
        handler: Callable[[ContextEvent], Any]
        | Callable[[ContextEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(
        self,
        channel: Literal["stream_delta"],
        handler: Callable[[StreamDeltaEvent], Any]
        | Callable[[StreamDeltaEvent], Awaitable[Any]],
    ) -> Callable[[], None]: ...
    @overload
    def on(self, channel: str, handler: Handler) -> Callable[[], None]: ...
    def on(self, channel: str, handler: Handler) -> Callable[[], None]:
        """Subscribe ``handler`` to ``channel``; return an unsubscribe fn.

        Calling the returned function removes this exact handler. Calling it
        a second time is a no-op (idempotent).
        """

        self._handlers.setdefault(channel, []).append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers.get(channel)
            if handlers is None:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

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
        self, channel: Literal["before_agent_end"], event: BeforeAgentEndEvent
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
        """

        registered = self._handlers.get(channel)
        if not registered and self._observer is None:
            return []
        handlers = list(registered or ())
        observer = self._observer
        self._safe_observe("on_emit_start", channel, event)
        results: list[Any] = []
        for h in handlers:
            err: BaseException | None = None
            start_ns = time.perf_counter_ns() if observer is not None else 0
            try:
                value = h(event)
                if inspect.isawaitable(value):
                    value = await value
            except Exception as exc:
                logger.exception(
                    "Event handler raised on channel %r; suppressing.", channel
                )
                err = exc
                value = None
            if observer is not None:
                self._safe_observe(
                    "on_handler_done",
                    channel,
                    h,
                    value,
                    err,
                    time.perf_counter_ns() - start_ns,
                )
            results.append(value)
        self._safe_observe("on_emit_end", channel, event, results)
        return results

    def _safe_observe(self, method: str, *args: Any) -> None:
        observer = self._observer
        if observer is None:
            return
        try:
            getattr(observer, method)(*args)
        except Exception:
            logger.exception("EventBus observer.%s raised; suppressing.", method)

    def emit_sync(self, channel: str, event: Any) -> list[Any]:
        """Synchronous emit — runs only sync handlers; coroutine-returning
        handlers are logged-and-skipped.

        Exists so sync code paths (``ExtensionAPI.register_tool`` and friends,
        which cannot ``await``) can still publish events on the bus without
        forcing the API surface async. Observer hooks fire normally.
        """
        registered = self._handlers.get(channel)
        if not registered and self._observer is None:
            return []
        handlers = list(registered or ())
        observer = self._observer
        self._safe_observe("on_emit_start", channel, event)
        results: list[Any] = []
        async_violation: tuple[str, Any] | None = None
        for h in handlers:
            err: BaseException | None = None
            start_ns = time.perf_counter_ns() if observer is not None else 0
            try:
                value = h(event)
                if inspect.isawaitable(value):
                    if hasattr(value, "close"):
                        value.close()
                    if self._strict_sync_handlers and async_violation is None:
                        async_violation = (channel, h)
                    else:
                        logger.warning(
                            "Async handler on channel %r skipped during emit_sync; "
                            "use a sync handler or subscribe via an async-only channel.",
                            channel,
                        )
                    value = None
            except Exception as exc:
                logger.exception(
                    "Event handler raised on channel %r; suppressing.", channel
                )
                err = exc
                value = None
            if observer is not None:
                self._safe_observe(
                    "on_handler_done",
                    channel,
                    h,
                    value,
                    err,
                    time.perf_counter_ns() - start_ns,
                )
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


__all__ = [
    "AgentEndEvent",
    "AgentStartEvent",
    "BeforeAgentEndEvent",
    "BeforeSendToLlmEvent",
    "ContextEvent",
    "DiagnosticEvent",
    "Event",
    "EventBus",
    "EventBusObserver",
    "Handler",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "StreamDeltaEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
