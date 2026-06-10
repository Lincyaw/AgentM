"""Kernel event bus and event taxonomy.

Implements §3.5 (Extension Bus) of
`.claude/designs/pluggable-architecture.md`. A small async event bus
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
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, overload

from .messages import AgentMessage, AssistantMessage
from .stream import Model
from .tool import Tool, ToolOutcome, ToolResult

if TYPE_CHECKING:
    # ``AgentSessionConfig`` is imported only for type hints —
    # ``session_config`` itself imports from ``agentm.core.abi`` (this
    # package's ``__init__``), so a runtime import would close the cycle.
    # All annotations on this module already use ``from __future__ import
    # annotations`` (PEP 563), so the type-only import is sufficient.
    from .session_config import AgentSessionConfig

    # ``SessionTelemetry`` is the per-session OTel handle (tracer + logger
    # + lifecycle-pairing tracker). Type-only import: every
    # :meth:`Event.to_otel` call site uses duck-typed attribute access on
    # the runtime object, so the kernel does not import the runtime module
    # at module-load time.
    from agentm.core.runtime.otel_export import SessionTelemetry

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


@dataclass(slots=True)
class Event:
    """Base class for all kernel events.

    Not frozen — the bus reassigns ``dispatch_id`` on every emit.
    """

    dispatch_id: str = field(
        default_factory=lambda: uuid.uuid4().hex, kw_only=True
    )

    def to_otel(self, telemetry: "SessionTelemetry") -> None:
        """No-op base; concrete overrides live in ``runtime.event_otel``."""
        del telemetry


# --- Termination causes -----------------------------------------------------


@dataclass(slots=True, frozen=True)
class TerminationCause:
    """Sealed sum-type: ``final=True`` causes cannot be overridden by extensions."""

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
class NoPendingInput(TerminationCause):
    """Resume-without-prompt: no pending input. Non-final so extensions can Inject."""


@dataclass(slots=True, frozen=True)
class BudgetExhausted(TerminationCause):
    """A budget cap was reached (``detail`` names which). Cannot be overridden."""

    final: ClassVar[bool] = True

    detail: str = ""


# --- Loop actions -----------------------------------------------------------


@dataclass(slots=True, frozen=True)
class LoopAction:
    """Sealed sum-type base for the loop's next action."""


@dataclass(slots=True, frozen=True)
class Step(LoopAction):
    """Continue to the next turn."""


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


@dataclass(slots=True)
class AgentStartEvent(Event):
    """Emitted once at the start of ``AgentLoop.run``."""

    CHANNEL: ClassVar[Literal["agent_start"]] = "agent_start"
    messages: list[AgentMessage]


@dataclass(slots=True)
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
    """One turn's outcome, given to DecideTurnActionEvent handlers."""

    turn_index: int
    assistant_message: AssistantMessage | None
    tool_outcomes: list[ToolOutcome]
    default_action: LoopAction
    turn_id: int = 0


@dataclass(slots=True)
class DecideTurnActionEvent(Event):
    """Fires after every turn. Handlers may return a LoopAction to override the default."""

    CHANNEL: ClassVar[Literal["decide_turn_action"]] = "decide_turn_action"
    observation: TurnObservation


@dataclass(slots=True)
class TurnStartEvent(Event):
    """Emitted at the start of each loop turn (one LLM call)."""

    CHANNEL: ClassVar[Literal["turn_start"]] = "turn_start"
    turn_index: int
    turn_id: int = 0


@dataclass(slots=True)
class TurnEndEvent(Event):
    """Emitted after a turn's assistant message is assembled. ``messages`` is the full snapshot."""

    CHANNEL: ClassVar[Literal["turn_end"]] = "turn_end"
    turn_index: int
    message: AssistantMessage
    messages: tuple[AgentMessage, ...] = ()
    turn_id: int = 0


@dataclass(slots=True)
class ToolCallEvent(Event):
    """Before tool execution. Handlers may mutate ``args`` or return ``{"block": True}``."""

    CHANNEL: ClassVar[Literal["tool_call"]] = "tool_call"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolResultEvent(Event):
    """After tool execution. Handlers may return a replacement ToolResult (last wins)."""

    CHANNEL: ClassVar[Literal["tool_result"]] = "tool_result"
    tool_call_id: str
    tool_name: str
    result: ToolResult


@dataclass(slots=True)
class ToolErrorEvent(Event):
    """Tool error: handlers populate ``result.content`` with the user-visible message."""

    CHANNEL: ClassVar[Literal["tool_error"]] = "tool_error"
    kind: Literal["execution_failed", "unknown_tool", "blocked"]
    tool_name: str
    reason: str
    result: ToolResult
    exception: BaseException | None = None


@dataclass(slots=True)
class BeforeSendToLlmEvent(Event):
    """Last hook before StreamFn. Handlers may mutate ``messages`` in place."""

    CHANNEL: ClassVar[Literal["before_send_to_llm"]] = "before_send_to_llm"
    messages: list[AgentMessage]
    model: Model
    tools: list[Tool]
    system: str | None


@dataclass(slots=True)
class LlmRequestStartEvent(Event):
    """Emitted right before the loop drains ``stream_fn``."""

    CHANNEL: ClassVar[Literal["llm_request_start"]] = "llm_request_start"
    turn_index: int
    message_count: int
    tool_count: int
    system_chars: int
    model_id: str | None
    turn_id: int = 0
    # Full system-prompt text. None means the loop chose not to persist
    # the body — only ``system_chars`` (the length) is recorded on the
    # ``chat`` span. Populated when ``AGENTM_TRACE_SYSTEM_PROMPT=1`` so
    # auditors can diff what was actually sent turn-by-turn (the prompt
    # is rebuilt each turn and may drift, which breaks prefix cache).
    system_text: str | None = None


@dataclass(slots=True)
class LlmRequestEndEvent(Event):
    """Emitted after the loop finishes draining ``stream_fn`` (success or
    error). ``error`` is ``repr(exc)`` on failure, ``None`` on success.
    """

    CHANNEL: ClassVar[Literal["llm_request_end"]] = "llm_request_end"
    turn_index: int
    chunk_count: int
    duration_ns: int
    error: str | None = None
    turn_id: int = 0


@dataclass(slots=True)
class StreamDeltaEvent(Event):
    """One raw chunk from the provider stream, forwarded for presenters."""

    CHANNEL: ClassVar[Literal["stream_delta"]] = "stream_delta"
    turn_index: int
    delta: Any  # AssistantStreamEvent — typed Any here to avoid pulling
    # the ``stream`` module into the events surface for everyone.
    turn_id: int = 0


@dataclass(slots=True)
class ContextEvent(Event):
    """Emitted before each LLM call with the current message list.

    **Mutability contract**: ``messages`` is the list that will be sent to
    the model. Handlers may mutate it in place, and/or return a replacement
    ``list[AgentMessage]``; the last non-None replacement wins.
    """

    CHANNEL: ClassVar[Literal["context"]] = "context"
    messages: list[AgentMessage]


@dataclass(slots=True)
class DiagnosticEvent(Event):
    """Non-fatal diagnostic; ``"error"`` level affects the exit code."""

    CHANNEL: ClassVar[Literal["diagnostic"]] = "diagnostic"
    level: Literal["info", "warning", "error"]
    source: str
    message: str


# --- Runtime-level event payloads ------------------------------------------


@dataclass(slots=True)
class BeforeAgentStartEvent(Event):
    """Before the kernel loop. Handlers may return ``{"system": "..."}`` to replace the prompt."""

    CHANNEL: ClassVar[Literal["before_agent_start"]] = "before_agent_start"
    messages: list[AgentMessage]
    system: str | None


@dataclass(slots=True)
class SessionShutdownEvent(Event):
    """Fires on ``AgentSession.shutdown``."""

    CHANNEL: ClassVar[Literal["session_shutdown"]] = "session_shutdown"
    cwd: str


@dataclass(slots=True)
class BeforeCompactEvent(Event):
    """Observation-only: fires before compaction. Handlers may mutate ``messages``."""

    CHANNEL: ClassVar[Literal["before_compact"]] = "before_compact"
    messages: list[AgentMessage]
    reason: str  # e.g. "auto_overflow", "manual", "scenario_request"


@dataclass(slots=True)
class AfterCompactEvent(Event):
    """Fires after compaction is committed to the SessionManager."""

    CHANNEL: ClassVar[Literal["after_compact"]] = "after_compact"
    summary: str
    kept_message_count: int
    discarded_message_count: int
    details: Any = None  # extension-specific (e.g. artifact index)


@dataclass(slots=True)
class ChildSessionStartEvent(Event):
    """Fires on the parent bus when a child AgentSession is created."""

    CHANNEL: ClassVar[Literal["child_session_start"]] = "child_session_start"
    child_session_id: str
    parent_session_id: str
    purpose: str  # e.g. "subagent:worker", caller-defined


@dataclass(slots=True)
class ChildSessionExtendingEvent(Event):
    """Handlers return ``[(module_path, config)]`` to add atoms to a child session."""

    CHANNEL: ClassVar[Literal["child_session_extending"]] = "child_session_extending"
    parent_session_id: str
    child_config: "AgentSessionConfig"


@dataclass(slots=True)
class ChildSessionEndEvent(Event):
    """Fires on the parent bus when a child AgentSession terminates."""

    CHANNEL: ClassVar[Literal["child_session_end"]] = "child_session_end"
    child_session_id: str
    parent_session_id: str
    final_message_count: int
    error: str | None = None


@dataclass(slots=True)
class CostBudgetExceededEvent(Event):
    """Budget limit crossed; next prompt short-circuits."""

    CHANNEL: ClassVar[Literal["cost_budget_exceeded"]] = "cost_budget_exceeded"
    used: float
    limit: float
    currency: str = "usd"


@dataclass(slots=True)
class PlanSubmittedEvent(Event):
    """A plan was submitted; ``plan_id`` is the session-entry id."""

    CHANNEL: ClassVar[Literal["plan_submitted"]] = "plan_submitted"
    plan_id: str
    plan_text: str


@dataclass(slots=True)
class MessagePersistedEvent(Event):
    """A durable message was appended to the loop's trajectory."""

    CHANNEL: ClassVar[Literal["message_persisted"]] = "message_persisted"
    message: AgentMessage
    source: Literal["assistant", "tool_result", "injected"]
    turn_index: int
    turn_id: int


@dataclass(slots=True)
class MessageAppendedEvent(Event):
    """SessionManager entry appended; ``record`` is the JSON-ready dict."""

    CHANNEL: ClassVar[Literal["message_appended"]] = "message_appended"
    record: dict[str, Any]


@dataclass(slots=True)
class SessionHeaderEmittedEvent(Event):
    """New session header minted; ``record`` is the JSON-ready dict."""

    CHANNEL: ClassVar[Literal["session_header_emitted"]] = "session_header_emitted"
    record: dict[str, Any]


@dataclass(slots=True)
class EntryAppendedEvent(Event):
    """A session entry was appended (sync-only handlers)."""

    CHANNEL: ClassVar[Literal["entry_appended"]] = "entry_appended"
    session_id: str
    entry_type: str
    entry_id: str
    parent_id: str | None
    payload: Any


@dataclass(slots=True)
class SessionReadyEvent(Event):
    """All extensions loaded, provider picked, before first prompt."""

    CHANNEL: ClassVar[Literal["session_ready"]] = "session_ready"
    cwd: str
    session_id: str
    tool_names: tuple[str, ...]
    command_names: tuple[str, ...]
    extension_module_paths: tuple[str, ...]
    model: Model | None
    root_session_id: str
    task_id: str | None = None
    persona: str | None = None
    task_class: str | None = None
    eval_run_id: str | None = None
    eval_task_id: str | None = None


@dataclass(slots=True)
class ResolveSubagentEvent(Event):
    """Request persona metadata for a named sub-agent type."""

    CHANNEL: ClassVar[Literal["resolve_subagent"]] = "resolve_subagent"
    name: str


@dataclass(slots=True)
class ExtensionInstallEvent(Event):
    """Extension install lifecycle: ``phase`` is ``start``/``end``/``error``."""

    CHANNEL: ClassVar[Literal["extension_install"]] = "extension_install"
    module_path: str
    config: dict[str, Any]
    phase: Literal["start", "end", "error"]
    duration_ns: int = 0
    error: str | None = None
    trigger: Literal["bootstrap", "agent", "human", "propose_change_approved"] = (
        "bootstrap"
    )


@dataclass(slots=True)
class ExtensionReloadEvent(Event):
    """Fires after a transactional reload succeeds or hits rollback failure."""

    CHANNEL: ClassVar[Literal["extension_reload"]] = "extension_reload"
    name: str
    old_hash: str | None
    new_hash: str
    trigger: Literal["agent", "human", "propose_change_approved"]
    tier: int
    error: str | None = None
    is_self_modify: bool = False


@dataclass(slots=True)
class BeforeInstallAtomEvent(Event):
    """Veto hook between internal safety gates and the on-disk write of
    ``api.install_atom``. Handlers may return ``{"block": True, "reason":
    "..."}`` to refuse — first truthy block wins. ``config`` is mutable
    in place; ``source`` and ``name`` are read-only here.
    """

    CHANNEL: ClassVar[Literal["before_install_atom"]] = "before_install_atom"
    name: str
    module_path: str
    target_path: str
    source: str
    config: dict[str, Any]
    tier: int
    trigger: Literal["agent", "human", "propose_change_approved"]


@dataclass(slots=True)
class BeforeUnloadAtomEvent(Event):
    """Veto hook before ``api.unload_atom`` removes an atom. Same contract
    as :class:`BeforeInstallAtomEvent` — return ``{"block": True, "reason":
    "..."}`` to refuse; first truthy block wins.
    """

    CHANNEL: ClassVar[Literal["before_unload_atom"]] = "before_unload_atom"
    name: str
    module_path: str
    tier: int
    trigger: Literal["agent", "human", "propose_change_approved"]


@dataclass(slots=True)
class CommandDispatchedEvent(Event):
    """Fires when ``AgentSession.prompt`` dispatches a slash-command to a
    code-registered handler (i.e. a :class:`CommandSpec` in the session's
    ``commands`` registry). Observation only; the dispatch has already
    happened by the time this fires. Use ``input`` events instead if you
    need to rewrite the text BEFORE dispatch.

    ``args`` is the rest-of-line argument string passed to the command
    handler. ``owner`` is the module path of the atom that registered the
    command, mirroring the ``api_register`` attribution.
    """

    CHANNEL: ClassVar[Literal["command_dispatched"]] = "command_dispatched"
    name: str
    args: str
    owner: str


@dataclass(slots=True)
class ExtensionUnloadEvent(Event):
    """Fires after a successful unload of an installed atom.

    Mirrors ``ExtensionReloadEvent`` so subscribers (TUI, observability)
    can treat install/reload/unload as one family. Provider extensions
    cannot be unloaded; constitution-path atoms cannot be unloaded.
    """

    CHANNEL: ClassVar[Literal["extension_unload"]] = "extension_unload"
    name: str
    module_path: str
    trigger: Literal["agent", "human", "propose_change_approved"]
    tier: int
    error: str | None = None


@dataclass(slots=True)
class ApiRegisterEvent(Event):
    """Fires synchronously from ``ExtensionAPI`` register methods.

    Emitted via ``bus.emit_sync`` so it works from inside sync ``install``
    bodies. Lets subscribers see what each extension contributes.
    """

    CHANNEL: ClassVar[Literal["api_register"]] = "api_register"
    kind: Literal["tool", "command", "provider", "renderer"]
    name: str
    extension: str
    payload: Any


@dataclass(slots=True)
class ApiSendUserMessageEvent(Event):
    """Fires when an extension calls ``api.send_user_message``."""

    CHANNEL: ClassVar[Literal["api_send_user_message"]] = "api_send_user_message"
    extension: str
    content: Any


@dataclass(slots=True)
class ResourcesDiscoverEvent(Event):
    """Fires when an extension wants peers to contribute resource paths."""

    CHANNEL: ClassVar[Literal["resources_discover"]] = "resources_discover"
    cwd: str
    reason: Literal["startup", "reload"]


@dataclass(slots=True)
class ResourceWriteEvent(Event):
    """Fires when a managed resource write lands as a git commit."""

    CHANNEL: ClassVar[Literal["resource_write"]] = "resource_write"
    path: str
    pre_sha: str
    post_sha: str
    rationale: str
    author: Literal["agent", "human", "indexer"]


# --- Bus --------------------------------------------------------------------


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
    ) -> None: ...

    def on_emit_end(
        self, channel: str, event: Any, results: list[Any]
    ) -> None: ...


ObserverRegistration = ObserverCallback | EventBusObserver


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
    _observer_callbacks: list[ObserverRegistration] = field(default_factory=list)
    _strict_sync_handlers: bool = False
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
            handlers = [sub.handler for sub in (registered or ())]
            self._handler_cache[channel] = handlers
        observer_callbacks = tuple(self._observer_callbacks)
        observer = self._observer
        observe_handlers = observer is not None or any(
            not callable(callback) for callback in observer_callbacks
        )
        if not handlers and observer is None and not observer_callbacks:
            return []
        if isinstance(event, Event):
            event.dispatch_id = uuid.uuid4().hex
        self._safe_observe("on_emit_start", channel, event)
        results: list[Any] = []
        for h in handlers:
            err: BaseException | None = None
            start_ns = time.perf_counter_ns() if observe_handlers else 0
            if observe_handlers:
                self._safe_observe("on_handler_start", channel, h, event)
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
            if observe_handlers:
                self._safe_observe(
                    "on_handler_done",
                    channel,
                    h,
                    event,
                    value,
                    err,
                    time.perf_counter_ns() - start_ns,
                )
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
                    logger.exception("EventBus observer.%s raised; suppressing.", method)
        observer = self._observer
        if observer is None:
            return
        try:
            observer_method = getattr(observer, method, None)
            if observer_method is None:
                return
            observer_method(*args)
        except Exception:
            logger.exception("EventBus observer.%s raised; suppressing.", method)

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
            handlers = [sub.handler for sub in (registered or ())]
            self._handler_cache[channel] = handlers
        observer_callbacks = tuple(self._observer_callbacks)
        observer = self._observer
        observe_handlers = observer is not None or any(
            not callable(callback) for callback in observer_callbacks
        )
        if not handlers and observer is None and not observer_callbacks:
            return []
        if isinstance(event, Event):
            event.dispatch_id = uuid.uuid4().hex
        async_violation: tuple[str, Any] | None = None
        self._safe_observe("on_emit_start", channel, event)
        results: list[Any] = []
        for h in handlers:
            err: BaseException | None = None
            start_ns = time.perf_counter_ns() if observe_handlers else 0
            if observe_handlers:
                self._safe_observe("on_handler_start", channel, h, event)
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
            if observe_handlers:
                self._safe_observe(
                    "on_handler_done",
                    channel,
                    h,
                    event,
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
    "AfterCompactEvent",
    "AgentEndEvent",
    "AgentStartEvent",
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "BeforeAgentStartEvent",
    "BeforeCompactEvent",
    "BeforeInstallAtomEvent",
    "BeforeSendToLlmEvent",
    "BeforeUnloadAtomEvent",
    "BudgetExhausted",
    "BusPriority",
    "ChildSessionEndEvent",
    "ChildSessionExtendingEvent",
    "ChildSessionStartEvent",
    "CommandDispatchedEvent",
    "ContextEvent",
    "CostBudgetExceededEvent",
    "DecideTurnActionEvent",
    "DiagnosticEvent",
    "EntryAppendedEvent",
    "Event",
    "EventBus",
    "EventBusObserver",
    "ExtensionInstallEvent",
    "ExtensionReloadEvent",
    "ExtensionUnloadEvent",
    "Handler",
    "Inject",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "LoopAction",
    "MaxTurnsExhausted",
    "MessageAppendedEvent",
    "MessagePersistedEvent",
    "ModelEndTurn",
    "NoPendingInput",
    "ObserverCallback",
    "ObserverRegistration",
    "PlanSubmittedEvent",
    "ProviderProtocolViolation",
    "ProviderTruncated",
    "ResolveSubagentEvent",
    "ResourceWriteEvent",
    "ResourcesDiscoverEvent",
    "SessionHeaderEmittedEvent",
    "SessionReadyEvent",
    "SessionShutdownEvent",
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
