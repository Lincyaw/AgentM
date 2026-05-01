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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, overload

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


@dataclass(slots=True)
class ContextEvent(Event):
    """Emitted before each LLM call with the current message list.

    **Mutability contract**: ``messages`` is the list that will be sent to
    the model. Handlers may mutate it in place, and/or return a replacement
    ``list[AgentMessage]``; the last non-None replacement wins.
    """

    messages: list[AgentMessage]


# --- Bus --------------------------------------------------------------------


# A handler may be sync or async; it returns anything (the bus collects).
Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]


@dataclass(slots=True)
class EventBus:
    """Minimal channel-keyed pub/sub. See module docstring for the contract."""

    _handlers: dict[str, list[Handler]] = field(default_factory=dict)

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

        handlers = list(self._handlers.get(channel, ()))
        results: list[Any] = []
        for h in handlers:
            try:
                value = h(event)
                if inspect.isawaitable(value):
                    value = await value
            except Exception:
                logger.exception(
                    "Event handler raised on channel %r; suppressing.", channel
                )
                value = None
            results.append(value)
        return results

    def clear(self) -> None:
        """Remove every subscription on every channel."""

        self._handlers.clear()


__all__ = [
    "AgentEndEvent",
    "AgentStartEvent",
    "BeforeSendToLlmEvent",
    "ContextEvent",
    "Event",
    "EventBus",
    "Handler",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
