"""Kernel event bus and event taxonomy.

Implements §3.5 (Extension Bus) of
`.claude/designs/pluggable-architecture.md`. Conceptually a port of pi-mono's
``packages/coding-agent/src/core/event-bus.ts`` (33 lines) to async Python,
plus a minimal seed of typed event dataclasses used by ``loop.py``.

Per-turn termination semantics follow the sum-type protocol described in
``.claude/designs/agent-loop.md``: each turn ends with one
:class:`LoopAction`, computed by the kernel from the assistant message and
tool outcomes, then optionally overridden by extensions on the
``decide_turn_action`` channel. ``TerminationCause`` subclasses carry rich
data (no string literals); extensions and observability pattern-match on
the concrete type.

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

import bisect
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal, Protocol, overload

from .messages import AgentMessage, AssistantMessage
from .stream import Model
from .tool import Tool, ToolOutcome, ToolResult

logger = logging.getLogger(__name__)


class BusPriority:
    """Symbolic dispatch tiers for ``EventBus.on(priority=...)``.

    Lower number runs earlier. Atoms declare their intended tier so the
    dispatch order is decoupled from registration/install order. See
    ``.claude/designs/handler-priority.md`` for the full contract.

    - ``PRE`` — security gates, validation, decisions that must run first.
    - ``NORMAL`` — default; business-logic atoms.
    - ``POST`` — audit, tracing, observability; sees final state.

    Numeric values between tiers are legal (e.g. ``priority=300``) but
    discouraged in atom code; the symbolic constants are the supported door.
    """

    PRE: Final[int] = 100
    NORMAL: Final[int] = 500
    POST: Final[int] = 900


# --- Event types ------------------------------------------------------------


class Event:
    """Marker base class for all kernel events.

    Concrete events are dataclasses; this class exists so callers can write
    ``isinstance(x, Event)`` if they want a structural check.
    """


# --- Termination causes -----------------------------------------------------


@dataclass(slots=True, frozen=True)
class TerminationCause:
    """Sealed sum-type base. Concrete subclasses describe WHO decided to
    terminate and WHY.

    ``final`` controls whether extensions can override the kernel default via
    :class:`Inject`. When ``final`` is True, the loop emits
    :class:`DecideTurnActionEvent` for observability but ignores any
    :class:`LoopAction` overrides — used for kernel-imposed terminations such
    as ``MaxTurnsExhausted`` and ``SignalAborted`` where the cause cannot be
    safely contradicted.
    """

    final: ClassVar[bool] = False


@dataclass(slots=True, frozen=True)
class ModelEndTurn(TerminationCause):
    """The assistant message had no tool_calls — model voluntarily finished."""


@dataclass(slots=True, frozen=True)
class ToolTerminated(TerminationCause):
    """A tool returned :class:`ToolTerminate`.

    ``tool_name`` and ``reason`` come from the terminal tool itself so
    downstream consumers can distinguish *which* terminal tool fired.
    """

    tool_name: str
    reason: str


@dataclass(slots=True, frozen=True)
class MaxTurnsExhausted(TerminationCause):
    """Loop ran to its turn cap. Cannot be overridden."""

    final: ClassVar[bool] = True


@dataclass(slots=True, frozen=True)
class SignalAborted(TerminationCause):
    """External :class:`asyncio.Event` signalled abort. Cannot be overridden."""

    final: ClassVar[bool] = True


@dataclass(slots=True, frozen=True)
class ProviderTruncated(TerminationCause):
    """Provider stopped streaming due to its own limit (max_tokens or error)."""

    kind: Literal["max_tokens", "error"]


@dataclass(slots=True, frozen=True)
class ProviderProtocolViolation(TerminationCause):
    """Provider stop_reason said ``tool_use`` but no tool_calls were extracted.

    Indicates a parser/provider disagreement worth surfacing distinctly
    rather than silently mapping to ``ModelEndTurn``.
    """

    detail: str


@dataclass(slots=True, frozen=True)
class BudgetExhausted(TerminationCause):
    """A budget cap was reached. ``detail`` names which budget — e.g.
    ``"cost"`` (harness session-level cost cap) or ``"max_tool_calls"``
    (kernel-loop tool-call cap). Cannot be overridden; extensions can cap
    budgets but not un-cap them once tripped.
    """

    final: ClassVar[bool] = True

    detail: str = ""


# --- Loop actions -----------------------------------------------------------


@dataclass(slots=True, frozen=True)
class LoopAction:
    """Sealed sum-type base for the loop's next action."""


@dataclass(slots=True, frozen=True)
class Step(LoopAction):
    """Continue to the next turn with current messages.

    Default action after a successful tools-and-results round when no tool
    asked to terminate.
    """


@dataclass(slots=True, frozen=True)
class Stop(LoopAction):
    """Terminate the loop with the given cause."""

    cause: TerminationCause


@dataclass(slots=True, frozen=True)
class Inject(LoopAction):
    """Continue to next turn after appending these messages.

    Used by extensions to override a default :class:`Stop` (e.g. inject a
    continuation prompt instead of terminating). Multiple ``Inject`` returns
    in the same turn are concatenated in registration order.
    """

    messages: list[AgentMessage]


# --- Event payloads ---------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AgentStartEvent(Event):
    """Emitted once at the start of ``AgentLoop.run``."""

    CHANNEL: ClassVar[Literal["agent_start"]] = "agent_start"
    messages: list[AgentMessage]


@dataclass(slots=True, frozen=True)
class AgentEndEvent(Event):
    """Emitted once when ``AgentLoop.run`` returns.

    ``cause`` is a :class:`TerminationCause` instance — pattern-match on the
    concrete subclass to identify why the loop stopped. Replaces the previous
    string-typed ``stop_reason`` field; consumers that just want a label can
    take ``type(cause).__name__``.
    """

    CHANNEL: ClassVar[Literal["agent_end"]] = "agent_end"
    messages: list[AgentMessage]
    cause: TerminationCause


@dataclass(slots=True)
class TurnObservation:
    """Snapshot of one turn's outcome.

    Given to :class:`DecideTurnActionEvent` handlers along with the kernel's
    default :class:`LoopAction`. ``assistant_message`` is ``None`` only on
    kernel-imposed termination paths (``SignalAborted`` / ``MaxTurnsExhausted``)
    that fire the hook for observability symmetry but skip the message
    pipeline; in that case ``tool_outcomes`` is also empty.
    """

    turn_index: int
    assistant_message: AssistantMessage | None
    tool_outcomes: list[ToolOutcome]
    default_action: LoopAction


@dataclass(slots=True)
class DecideTurnActionEvent(Event):
    """Fires after every turn, before the loop advances or terminates.

    Handlers may return a :class:`LoopAction` (or ``None`` for "no opinion")
    to override the kernel's default. Resolution lattice (see
    ``.claude/designs/agent-loop.md``):

    1. If the kernel default is ``Stop(cause)`` and ``cause.final`` is True,
       no override is honored — the hook fires for logging/observability
       only.
    2. Among handler returns, any :class:`Inject` wins (messages from all
       Inject returns are concatenated in registration order); else any
       :class:`Stop` overrides ``Step``; else the default applies.
    """

    CHANNEL: ClassVar[Literal["decide_turn_action"]] = "decide_turn_action"
    observation: TurnObservation


@dataclass(slots=True, frozen=True)
class TurnStartEvent(Event):
    """Emitted at the start of each loop turn (one LLM call)."""

    CHANNEL: ClassVar[Literal["turn_start"]] = "turn_start"
    turn_index: int


@dataclass(slots=True, frozen=True)
class TurnEndEvent(Event):
    """Emitted after a turn's assistant message is fully assembled.

    ``messages`` is the full live trajectory snapshot **including** the
    just-emitted ``message`` and every prior turn's tool calls / tool
    results. Consumers that need to slice "what was new this turn" do so
    against this snapshot rather than ``api.session.get_messages()``,
    which only reflects entries the kernel has persisted to the
    SessionManager — the kernel persists in one batch after ``prompt()``
    returns, so mid-loop reads of the session view are stale.
    """

    CHANNEL: ClassVar[Literal["turn_end"]] = "turn_end"
    turn_index: int
    message: AssistantMessage
    messages: tuple[AgentMessage, ...] = ()


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

    CHANNEL: ClassVar[Literal["tool_call"]] = "tool_call"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolResultEvent(Event):
    """Emitted after a tool returns.

    **Replacement contract**: handlers may return a replacement
    :class:`ToolResult`; the last non-None replacement wins (see
    ``loop._collect_tool_result_replacement``).
    """

    CHANNEL: ClassVar[Literal["tool_result"]] = "tool_result"
    tool_call_id: str
    tool_name: str
    result: ToolResult


@dataclass(slots=True, frozen=True)
class ToolErrorEvent(Event):
    """Emitted by the loop when a tool call cannot produce a normal result.

    The kernel does NOT synthesize the user-visible English error string
    itself; it constructs an empty :class:`ToolResult` (``is_error=True``,
    ``content=[]``) and emits this event so a default builtin atom
    (``tool_error_messages``) can write the human-readable text into
    ``result.content``. Extensions that want to localize, re-format, or
    suppress error text replace the default atom on this channel.

    The ``result`` field is the same instance the loop will pass through
    :class:`ToolResultEvent` and into the message trajectory; handlers
    mutate ``result.content`` in place. The dataclass itself is frozen
    because the *kind* / *tool_name* / *reason* are facts the kernel has
    already decided; they describe the cause, not a recommendation.

    ``kind`` discriminates the three kernel-imposed error paths:
    - ``"execution_failed"`` — ``tool.execute`` raised an exception.
    - ``"unknown_tool"``     — the assistant called a name not in the
                                 tool index.
    - ``"blocked"``          — a ``tool_call`` handler returned
                                 ``{"block": True, "reason": ...}``.
    """

    CHANNEL: ClassVar[Literal["tool_error"]] = "tool_error"
    kind: Literal["execution_failed", "unknown_tool", "blocked"]
    tool_name: str
    reason: str
    result: ToolResult
    exception: BaseException | None = None


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

    CHANNEL: ClassVar[Literal["before_send_to_llm"]] = "before_send_to_llm"
    messages: list[AgentMessage]
    model: Model
    tools: list[Tool]
    system: str | None


@dataclass(slots=True, frozen=True)
class LlmRequestStartEvent(Event):
    """Emitted right before the loop drains ``stream_fn``."""

    CHANNEL: ClassVar[Literal["llm_request_start"]] = "llm_request_start"
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

    CHANNEL: ClassVar[Literal["llm_request_end"]] = "llm_request_end"
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

    CHANNEL: ClassVar[Literal["stream_delta"]] = "stream_delta"
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

    CHANNEL: ClassVar[Literal["context"]] = "context"
    messages: list[AgentMessage]


@dataclass(slots=True, frozen=True)
class DiagnosticEvent(Event):
    """Non-fatal diagnostic from any subsystem during session construction
    or runtime. Emitted on the ``"diagnostic"`` channel. Used for failures
    that the recovery-floor invariant requires us to survive — scenario
    yaml load errors, skill/prompt loader hiccups, etc. The CLI subscribes
    and prints; only ``"error"`` level affects the exit code.
    """

    CHANNEL: ClassVar[Literal["diagnostic"]] = "diagnostic"
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


@dataclass(frozen=True, slots=True)
class _Subscription:
    """One handler registration with its dispatch-order key.

    ``priority`` is the tier (lower runs earlier). ``seq`` is a monotonic
    counter assigned at subscribe time so two same-priority handlers are
    ordered by registration (FIFO within tier). Sorting is purely on
    ``(priority, seq)``; ``handler`` is the payload the bus invokes.
    """

    priority: int
    seq: int
    handler: Handler


def _sub_key(sub: _Subscription) -> tuple[int, int]:
    return (sub.priority, sub.seq)


@dataclass(slots=True)
class EventBus:
    """Minimal channel-keyed pub/sub. See module docstring for the contract."""

    _handlers: dict[str, list[_Subscription]] = field(default_factory=dict)
    # Per-channel handler-only list, regenerated lazily after any
    # ``on``/``unsubscribe`` mutation. Avoids rebuilding ``[s.handler for s
    # in subs]`` on every emit — load-bearing on hot channels like
    # ``stream_delta`` (one emission per provider chunk).
    _handler_cache: dict[str, list[Handler]] = field(default_factory=dict)
    _observer: EventBusObserver | None = None
    _strict_sync_handlers: bool = False
    _next_seq: int = 0

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
    ) -> Callable[[], None]: ...
    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Callable[[], None]:
        """Subscribe ``handler`` to ``channel``; return an unsubscribe fn.

        Calling the returned function removes this exact handler. Calling it
        a second time is a no-op (idempotent). ``priority`` selects the
        dispatch tier — see :class:`BusPriority`. Within a tier, registration
        order is preserved (FIFO).
        """

        sub = _Subscription(priority=priority, seq=self._next_seq, handler=handler)
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
                if existing.handler is handler:
                    del handlers[idx]
                    self._handler_cache.pop(channel, None)
                    return

        return unsubscribe

    def subscriptions_for(self, channel: str) -> list[_Subscription]:
        """Return a fresh shallow copy of subscriptions on ``channel`` in
        dispatch order. Used by harness internals (atom_reloader) that need
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
        harness internals that need to splice handlers back in a specific
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
        """

        handlers = self._handler_cache.get(channel)
        if handlers is None:
            registered = self._handlers.get(channel)
            handlers = [sub.handler for sub in (registered or ())]
            self._handler_cache[channel] = handlers
        if not handlers and self._observer is None:
            return []
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
        handlers = self._handler_cache.get(channel)
        if handlers is None:
            registered = self._handlers.get(channel)
            handlers = [sub.handler for sub in (registered or ())]
            self._handler_cache[channel] = handlers
        if not handlers and self._observer is None:
            return []
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
    "BeforeSendToLlmEvent",
    "BudgetExhausted",
    "BusPriority",
    "ContextEvent",
    "DecideTurnActionEvent",
    "DiagnosticEvent",
    "Event",
    "EventBus",
    "EventBusObserver",
    "Handler",
    "Inject",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "LoopAction",
    "MaxTurnsExhausted",
    "ModelEndTurn",
    "ProviderProtocolViolation",
    "ProviderTruncated",
    "SignalAborted",
    "Step",
    "Stop",
    "StreamDeltaEvent",
    "TerminationCause",
    "ToolCallEvent",
    "ToolErrorEvent",
    "ToolResultEvent",
    "ToolTerminated",
    "TurnEndEvent",
    "TurnObservation",
    "TurnStartEvent",
]
