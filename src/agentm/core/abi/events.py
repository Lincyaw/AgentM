"""Kernel event taxonomy — typed event dataclasses.

Implements §3.5 (Extension Bus) of
`.claude/designs/pluggable-architecture.md`. Typed event dataclasses used
by ``loop.py`` and the event bus in ``bus.py``.

Per-turn termination semantics follow the sum-type protocol described in
``.claude/designs/agent-loop.md``: each turn ends with one
:class:`LoopAction`, computed by the kernel from the assistant message and
tool outcomes, then optionally overridden by extensions on the
``decide_turn_action`` channel. ``TerminationCause`` subclasses carry rich
data (no string literals); extensions and observability pattern-match on
the concrete type.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

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


@dataclass(slots=True, frozen=True)
class HookContract:
    """Machine-readable handler contract for an event channel.

    The event class remains the source of truth for channel name and payload.
    ``HOOK`` only documents how atom handlers may use the event: whether it is
    recommended for agent-authored atoms, whether payload mutation is supported,
    and whether handler return values are consumed by the runtime.
    """

    visibility: Literal["recommended", "advanced", "internal"] = "advanced"
    effects: tuple[str, ...] = ("observe",)
    return_contract: str | None = None
    mutation_contract: str | None = None
    mutable_fields: tuple[str, ...] = ()
    """Event fields handlers may mutate or replace before the emitter proceeds."""
    handler: Literal["sync_or_async", "sync_only"] = "sync_or_async"
    notes: tuple[str, ...] = ()
    """Additional caveats not already covered by the Event class docstring."""


# --- Event types ------------------------------------------------------------


@dataclass(slots=True)
class Event:
    """Base class for all kernel events.

    Carries a per-emission ``dispatch_id`` so observability consumers can
    correlate a single ``agentm.event.dispatch`` log record with every
    ``agentm.handler.invoke`` record fanned out from it. The field default
    keeps test fixtures and standalone constructions sane; the
    :class:`EventBus` **reassigns** ``dispatch_id`` on every ``emit`` /
    ``emit_sync`` entry, so a re-emitted instance produces a fresh id.

    Declared ``kw_only=True`` at the field level so subclass positional
    fields remain compatible without re-ordering. The base is **not**
    ``frozen`` — the bus needs to write ``dispatch_id`` on every emit, and
    Python dataclasses forbid mixing frozen / non-frozen across the
    inheritance chain — so every concrete Event subclass is also non-frozen.
    """

    dispatch_id: str = field(
        default_factory=lambda: uuid.uuid4().hex, kw_only=True
    )


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
class NoPendingInput(TerminationCause):
    """Resume-without-prompt found nothing to do.

    Surfaced by :meth:`AgentSession.tick` as the **default action** of the
    synthetic ``decide_turn_action`` it fires before any LLM call.
    ``final = False`` is load-bearing, not symmetric padding: it is what
    permits an extension handler on the same channel to return :class:`Inject` and
    override the default — without that override, the loop never runs and
    the resume produces no new turn. The cause only appears on
    :class:`AgentEndEvent` when no handler injected; in that case the
    trajectory is intentionally untouched (see ``tick`` docstring).
    """


@dataclass(slots=True, frozen=True)
class BudgetExhausted(TerminationCause):
    """A budget cap was reached. ``detail`` names which budget — e.g.
    ``"cost"`` (runtime session-level cost cap) or ``"max_tool_calls"``
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


@dataclass(slots=True)
class AgentStartEvent(Event):
    """Emitted once at the start of ``AgentLoop.run``."""

    CHANNEL: ClassVar[Literal["agent_start"]] = "agent_start"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
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
    turn_id: int = 0


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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "override_loop_action", "inject_messages"),
        return_contract="LoopAction | None",
    )
    observation: TurnObservation


@dataclass(slots=True)
class TurnStartEvent(Event):
    """Emitted at the start of each loop turn (one LLM call)."""

    CHANNEL: ClassVar[Literal["turn_start"]] = "turn_start"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe",),
    )
    turn_index: int
    turn_id: int = 0


@dataclass(slots=True)
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe",),
    )
    turn_index: int
    message: AssistantMessage
    messages: tuple[AgentMessage, ...] = ()
    turn_id: int = 0


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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "mutate_args", "block_tool_call"),
        return_contract=(
            "{\"block\": true, \"reason\": str, "
            "\"kind\"?: \"user_rejected\"} | None"
        ),
        mutation_contract="event.args may be mutated in place before execution.",
        mutable_fields=("args",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "mutate_result", "replace_result"),
        return_contract="ToolResult | None",
        mutation_contract=(
            "event.result may be mutated or replaced; a returned ToolResult "
            "takes precedence."
        ),
        mutable_fields=("result",),
    )
    tool_call_id: str
    tool_name: str
    result: ToolResult


@dataclass(slots=True)
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
    mutate ``result.content`` in place. The *kind* / *tool_name* / *reason*
    fields are runtime-checked as read-only because they are facts the kernel
    already decided; they describe the cause, not a recommendation.

    ``kind`` discriminates the kernel-imposed error paths:
    - ``"execution_failed"`` — ``tool.execute`` raised an exception.
    - ``"unknown_tool"``     — the assistant called a name not in the
                                 tool index.
    - ``"blocked"``          — a ``tool_call`` handler returned
                                 ``{"block": True, "reason": ...}``.
    - ``"user_rejected"``    — a ``tool_call`` handler returned
                                 ``{"block": True, "kind": "user_rejected", ...}``.
                                 Signals the model should try a different
                                 approach rather than retry.
    """

    CHANNEL: ClassVar[Literal["tool_error"]] = "tool_error"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "format_error_result"),
        mutation_contract=(
            "event.result.content may be mutated in place before it is "
            "surfaced to the model."
        ),
        mutable_fields=("result",),
    )
    kind: Literal["execution_failed", "unknown_tool", "blocked", "user_rejected"]
    tool_name: str
    reason: str
    result: ToolResult
    exception: BaseException | None = None


@dataclass(slots=True)
class BeforeSendToLlmEvent(Event):
    """Fires after context handlers have rewritten messages, immediately
    before the StreamFn is invoked. Handlers MAY mutate ``messages`` in
    place to make a final adjustment. This is the last hook before bytes
    leave the runtime.

    The ``messages`` list is the same instance the loop will pass to
    ``StreamFn``; mutate it cautiously.

    Distinct from :class:`ContextEvent`: ``context`` may rewrite-by-return
    (replacement list) or in-place; this event fires *after* that resolution
    on the final ``messages`` list, so handlers like ``cost_budget`` see
    exactly what is about to hit the wire.
    """

    CHANNEL: ClassVar[Literal["before_send_to_llm"]] = "before_send_to_llm"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=(
            "observe",
            "mutate_messages",
            "replace_model",
            "replace_tools",
            "mutate_system",
        ),
        mutation_contract=(
            "event.messages, event.model, event.tools, and event.system are the "
            "final provider request and may be replaced immediately before send."
        ),
        mutable_fields=("messages", "model", "tools", "system"),
    )
    messages: list[AgentMessage]
    model: Model
    tools: list[Tool]
    system: str | None


@dataclass(slots=True)
class LlmRequestStartEvent(Event):
    """Emitted right before the loop drains ``stream_fn``."""

    CHANNEL: ClassVar[Literal["llm_request_start"]] = "llm_request_start"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    turn_index: int
    chunk_count: int
    duration_ns: int
    error: str | None = None
    turn_id: int = 0


@dataclass(slots=True)
class StreamDeltaEvent(Event):
    """One raw chunk from the provider stream, forwarded by the loop.

    Used by presenters (TUI, JSON tap) that want token-by-token output.
    The kernel still assembles the full ``AssistantMessage`` itself and
    publishes it via ``turn_end``; ``stream_delta`` is purely additive.

    This is a high-volume channel; prefer ``turn_end`` unless token-level
    streaming is required.

    ``delta`` is the same ``AssistantStreamEvent`` instance the loop
    received from ``stream_fn`` — typically a ``TextDelta``,
    ``ToolCallStart``, or ``MessageEnd``. Subscribers should pattern-match
    on the delta's type.
    """

    CHANNEL: ClassVar[Literal["stream_delta"]] = "stream_delta"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="internal",
        effects=("observe",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "mutate_messages", "replace_messages"),
        return_contract="list[AgentMessage] | None",
        mutation_contract=(
            "event.messages may be mutated in place; a returned list replaces "
            "the current context."
        ),
        mutable_fields=("messages",),
    )
    messages: list[AgentMessage]


@dataclass(slots=True)
class DiagnosticEvent(Event):
    """Non-fatal diagnostic from any subsystem during session construction
    or runtime. Emitted on the ``"diagnostic"`` channel. Used for failures
    that the recovery-floor invariant requires us to survive — scenario
    yaml load errors, skill/prompt loader hiccups, etc. The CLI subscribes
    and prints; only ``"error"`` level affects the exit code.
    """

    CHANNEL: ClassVar[Literal["diagnostic"]] = "diagnostic"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe",),
    )
    level: Literal["info", "warning", "error"]
    source: str
    message: str


@dataclass(slots=True)
class BackgroundActivityEvent(Event):
    """Structured status for detached background work and monitor signals.

    Producers still use ``SessionInbox`` for model-visible completions and
    monitor fires. This event is the sibling presenter signal that lets a wire
    client render the same unit in its chrome while it is live.
    """

    CHANNEL: ClassVar[Literal["background_activity"]] = "background_activity"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    source: str
    activity_id: str
    label: str
    status: str
    session_id: str | None = None
    note: str | None = None
    terminal: bool = False


# --- Runtime-level event payloads ------------------------------------------
#
# The events below are emitted by runtime-level subsystems (compaction,
# child-session lifecycle, cost budget, plan-mode, install/reload/unload,
# resource writes). They live in the ABI module — alongside the kernel
# events — so atoms have a single canonical import for every event payload.
# Per the layer rule (kernel does not import runtime), the kernel ``EventBus``
# does not have typed ``emit`` overloads for these; they flow through the
# ``str`` fallback channel.


@dataclass(slots=True)
class BeforeAgentStartEvent(Event):
    """Fires at the top of ``AgentSession.prompt`` before the kernel loop runs.

    **System prompt** — mutate ``event.system`` in place. Handlers see each
    other's mutations, so chained appends work naturally. The runtime reads
    the final ``event.system`` after all handlers run.

    **Veto** — set ``event.veto = <TerminationCause>`` to abort the prompt
    before the loop runs. The runtime checks after all handlers; first
    non-None ``veto`` wins (but all handlers still run for observability).

    ``messages`` is the live list that will be passed to the loop — handlers
    should generally not rewrite it here (use ``context`` /
    ``before_send_to_llm`` for that).
    """

    CHANNEL: ClassVar[Literal["before_agent_start"]] = "before_agent_start"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "mutate_messages", "mutate_system", "veto_prompt"),
        mutation_contract=(
            "event.messages and event.system may be mutated or replaced; "
            "event.veto may be set."
        ),
        mutable_fields=("messages", "system", "veto"),
    )
    messages: list[AgentMessage]
    system: str | None
    veto: TerminationCause | None = None


@dataclass(slots=True)
class InputEvent(Event):
    """Fires when user text arrives, before the LLM driver runs.

    **Text rewriting**: mutate ``event.text`` in place to transform the
    user prompt (e.g., expand ``/template`` invocations). The runtime
    reads the final ``event.text`` after all handlers run.

    **Full handling** (slash commands): set ``event.handled = True`` and
    ``event.handled_messages = [...]`` to short-circuit the normal prompt
    flow. The runtime returns ``handled_messages`` directly to the
    ``prompt()`` caller without invoking the LLM.
    """

    CHANNEL: ClassVar[Literal["input"]] = "input"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe", "rewrite_input", "handle_input"),
        mutation_contract=(
            "event.text may be mutated; set handled and handled_messages to "
            "short-circuit normal prompt flow."
        ),
        mutable_fields=("text", "handled", "handled_messages"),
    )
    text: str
    handled: bool = False
    handled_messages: list[AgentMessage] | None = None


@dataclass(slots=True)
class SessionShutdownEvent(Event):
    """Fires when ``AgentSession.shutdown`` is called.

    Carries the session's cwd so cleanup handlers can locate session-scoped
    resources without holding a reference to the session itself.
    """

    CHANNEL: ClassVar[Literal["session_shutdown"]] = "session_shutdown"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe", "cleanup"),
    )
    cwd: str


@dataclass(slots=True)
class BeforeCompactEvent(Event):
    """Fires before an extension performs context compaction.

    Observation-only channel: the emitter (``llm_compaction``)
    discards handler return values. Subscribers may inspect or mutate
    ``messages`` in place to influence the buffer that compaction will see,
    but cannot cancel or replace the compaction itself.

    Mutability: ``messages`` is intentionally mutable (not frozen) so a
    handler can adjust the in-flight buffer before compaction kicks off.
    """

    CHANNEL: ClassVar[Literal["before_compact"]] = "before_compact"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe", "mutate_messages"),
        mutation_contract=(
            "event.messages may be adjusted before compaction consumes them."
        ),
        mutable_fields=("messages",),
    )
    messages: list[AgentMessage]
    reason: str  # e.g. "auto_overflow", "manual", "scenario_request"


@dataclass(slots=True)
class AfterCompactEvent(Event):
    """Fires after compaction is committed to the SessionManager."""

    CHANNEL: ClassVar[Literal["after_compact"]] = "after_compact"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    summary: str
    kept_message_count: int
    discarded_message_count: int
    details: Any = None  # extension-specific (e.g. artifact index)


@dataclass(slots=True)
class ChildSessionStartEvent(Event):
    """Fires on the parent bus when a child AgentSession is created."""

    CHANNEL: ClassVar[Literal["child_session_start"]] = "child_session_start"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    child_session_id: str
    parent_session_id: str
    purpose: str  # e.g. "subagent:worker", caller-defined


@dataclass(slots=True)
class ChildSessionExtendingEvent(Event):
    """Fires synchronously on the parent bus BEFORE the substrate spawns a
    child session, so extensions can contribute additional atoms to the
    child's load order.

    Handlers should return either ``None`` (no opinion) or a list of
    ``(module_path, config)`` tuples that the substrate will append to
    the child's ``AgentSessionConfig.extensions`` before the factory
    runs. Multiple handlers' contributions are concatenated in
    registration order. The substrate dedupes by ``module_path``: if an
    entry is already present in ``child_config.extensions`` (operator
    override) OR contributed by an earlier handler, later contributions
    of the same module are dropped — handlers don't need to dedupe
    themselves.

    The ``child_config`` is exposed read-only on the event; handlers
    MUST NOT mutate it. The substrate clones the extensions list before
    appending, so even if a misbehaving handler mutates the field in
    place the live config the factory sees is the substrate-controlled
    one.

    Emitted via ``bus.emit_sync`` because the spawn path needs the
    contributions to be settled before ``session_cls.create`` runs;
    async handlers are skipped (matching the rest of the ``emit_sync``
    contract).
    """

    CHANNEL: ClassVar[Literal["child_session_extending"]] = "child_session_extending"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe", "extend_child_session"),
        return_contract="list[tuple[str, dict[str, Any]]] | None",
        handler="sync_only",
    )
    parent_session_id: str
    child_config: "AgentSessionConfig"


@dataclass(slots=True)
class ChildSessionEndEvent(Event):
    """Fires on the parent bus when a child AgentSession terminates."""

    CHANNEL: ClassVar[Literal["child_session_end"]] = "child_session_end"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    child_session_id: str
    parent_session_id: str
    final_message_count: int
    error: str | None = None


@dataclass(slots=True)
class CostBudgetExceededEvent(Event):
    """Fires when the ``cost_budget`` extension's accumulator crosses the
    configured limit.

    ``AgentSession`` subscribes once at create-time and latches an internal
    flag; the next ``prompt`` short-circuits with an ``agent_end`` event
    carrying ``stop_reason='budget'``. Pure event-bus signalling — no
    exceptions cross handler boundaries.
    """

    CHANNEL: ClassVar[Literal["cost_budget_exceeded"]] = "cost_budget_exceeded"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    used: float
    limit: float
    currency: str = "usd"


@dataclass(slots=True)
class PlanSubmittedEvent(Event):
    """Fires when the ``tool_submit_plan`` tool runs to completion.

    Carries the plan id (entry id returned by ``ReadonlySession.append_entry``)
    so downstream extensions (``trajectory``, plan-mode controllers) can
    correlate the submission to its persisted entry.
    """

    CHANNEL: ClassVar[Literal["plan_submitted"]] = "plan_submitted"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    plan_id: str
    plan_text: str


@dataclass(slots=True)
class MessagePersistedEvent(Event):
    """Fires from inside ``AgentLoop`` immediately after the loop's local
    message list gains a new durable entry — assistant turn, tool_result, or
    extension-injected message. The runtime subscribes once and routes each
    event through ``SessionManager.append_message`` so the on-disk trajectory
    is updated in real time rather than batched at the end of ``run``.

    Whole-list replacements done by compaction / context-rewrite handlers
    via ``messages[:] = ...`` deliberately do NOT emit this event — those
    are ephemeral context rewrites, not durable additions.

    This can be high-volume in tool-heavy runs.
    """

    CHANNEL: ClassVar[Literal["message_persisted"]] = "message_persisted"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    message: AgentMessage
    source: Literal["assistant", "tool_result", "injected"]
    turn_index: int
    turn_id: int


@dataclass(slots=True)
class MessageAppendedEvent(Event):
    """Fires after :meth:`SessionManager._append_record` mutates in-memory
    state.

    Routes session-trajectory persistence through the observability sink:
    the observability atom subscribes and writes ``record`` straight into
    the merged per-session JSONL. Replaces the older synchronous
    ``open("a") + write + close`` per-message write path in SessionManager
    (see ``.claude/designs/single-event-log.md``).

    ``record`` is the JSON-ready dict shape SessionManager already used on
    disk (``{"type", "id", "parent_id", "timestamp", "payload"}``) — keeping
    it as a plain dict avoids dragging :class:`SessionEntry` into the
    ABI-event surface and lets the writer dump it without further
    massaging.
    """

    CHANNEL: ClassVar[Literal["message_appended"]] = "message_appended"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    record: dict[str, Any]


@dataclass(slots=True)
class SessionHeaderEmittedEvent(Event):
    """Fires when :meth:`SessionManager.new_session` mints a fresh session
    header.

    Header round-trips through the same merged log as
    :class:`MessageAppendedEvent`. ``record`` carries the JSON-ready
    SessionHeader dict; subsequent loads filter by ``kind=session.header``
    and take the most recent (the "rewrite-style behavior in the merged
    world becomes 'emit a new header'" reconciliation from the spec).
    """

    CHANNEL: ClassVar[Literal["session_header_emitted"]] = "session_header_emitted"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    record: dict[str, Any]


@dataclass(slots=True)
class EntryAppendedEvent(Event):
    """Fires after :meth:`ReadonlySession.append_entry` persists an entry.

    Lets extensions observe every write to the session entry tree —
    assistant messages, ``llmharness.audit_event`` / ``llmharness.verdict`` /
    ``llmharness.audit_graph_op`` entries, plan submissions, etc. — without
    polling ``get_branch()`` or tailing the on-disk JSONL.

    Emitted via :meth:`EventBus.emit_sync` from inside the sync
    ``append_entry`` codepath. Handlers must therefore be sync (an async
    handler is skipped with a diagnostic, matching the rest of the
    ``emit_sync`` contract). The event fires AFTER the entry has been
    durably written so handler crashes cannot corrupt session state.

    ``payload`` is the raw object passed to ``append_entry`` — observers
    that need a JSON-serialisable view should run it through
    :func:`agentm.core.lib.to_jsonable` themselves; we don't pre-serialise
    on the hot path since most subscribers (e.g. ``live_inspector``) need
    a custom shape anyway.
    """

    CHANNEL: ClassVar[Literal["entry_appended"]] = "entry_appended"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
        handler="sync_only",
    )
    session_id: str
    """The persisted session-manager header id (``ReadonlySession.get_session_id``),
    not the OTel span id. Distinct from the bus-owning session's
    ``api.session_id`` for embedded callers."""
    entry_type: str
    entry_id: str
    parent_id: str | None
    payload: Any


@dataclass(slots=True)
class SessionReadyEvent(Event):
    """Fires once after ``AgentSession.create`` has loaded every extension
    and the active provider has been picked, but before the first ``prompt``.

    This is the only timing point where every extension is guaranteed to see
    the *final* tool list, command set, and model. ``tool_filter`` and
    similar "post-install scrub" extensions hook here.
    """

    CHANNEL: ClassVar[Literal["session_ready"]] = "session_ready"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "initialize_after_tools_ready"),
    )
    cwd: str
    session_id: str
    tool_names: tuple[str, ...]
    command_names: tuple[str, ...]
    extension_module_paths: tuple[str, ...]
    model: Model | None
    root_session_id: str
    task_id: str | None = None
    persona: str | None = None
    # Per-task-evolution loop fields (see per-task-evolution-loop.md §4).
    # ``task_class`` ties this session's trace to a tunable task family;
    # ``eval_run_id`` and ``eval_task_id`` are populated only on eval-run
    # child sessions spawned by eval harnesses.
    task_class: str | None = None
    eval_run_id: str | None = None
    eval_task_id: str | None = None


@dataclass(slots=True)
class ResolveSubagentEvent(Event):
    """Request persona metadata for a named sub-agent type.

    The ``sub_agent`` atom emits this typed channel before spawning a child
    session. Scenario atoms may return a mapping with ``body``, ``tools``,
    ``input_schema``, ``budget_defaults``, and ``artifact_kinds`` entries.
    """

    CHANNEL: ClassVar[Literal["resolve_subagent"]] = "resolve_subagent"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe", "provide_subagent_metadata"),
        return_contract="dict[str, Any] | None",
    )
    name: str


@dataclass(slots=True)
class ExtensionInstallEvent(Event):
    """Fires twice per ``load_extension`` call: ``"start"`` precedes
    ``install(api, config)``; ``"end"`` follows a successful return;
    ``"error"`` follows a thrown exception.

    ``trigger`` distinguishes who initiated the install. ``"bootstrap"``
    is the default for installs done by ``AgentSession.create`` from a
    scenario or auto-discovery; the other values flow through
    ``api.install_atom``. Subscribers (e.g. the TUI) use this to decide
    whether to surface a "★ self-modify" toast.
    """

    CHANNEL: ClassVar[Literal["extension_install"]] = "extension_install"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "mutate_config", "block_install"),
        return_contract="{\"block\": true, \"reason\": str} | None",
        mutation_contract="event.config may be mutated before atom install.",
        mutable_fields=("config",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe", "block_unload"),
        return_contract="{\"block\": true, \"reason\": str} | None",
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe",),
    )
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
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="recommended",
        effects=("observe",),
        handler="sync_only",
        mutable_fields=("payload",),
    )
    kind: Literal["tool", "command", "provider", "renderer"]
    name: str
    extension: str
    payload: Any


@dataclass(slots=True)
class ApiSendUserMessageEvent(Event):
    """Fires when an extension calls ``api.send_user_message``."""

    CHANNEL: ClassVar[Literal["api_send_user_message"]] = "api_send_user_message"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    extension: str
    content: Any


@dataclass(slots=True)
class ResourcesDiscoverEvent(Event):
    """Fires when an extension wants peers to contribute resource paths."""

    CHANNEL: ClassVar[Literal["resources_discover"]] = "resources_discover"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe", "provide_resource_paths"),
        return_contract="list[str] | None",
    )
    cwd: str
    reason: Literal["startup", "reload"]


@dataclass(slots=True)
class ResourceWriteEvent(Event):
    """Fires when a managed resource write lands as a git commit."""

    CHANNEL: ClassVar[Literal["resource_write"]] = "resource_write"
    HOOK: ClassVar[HookContract] = HookContract(
        visibility="advanced",
        effects=("observe",),
    )
    path: str
    pre_sha: str
    post_sha: str
    rationale: str
    author: Literal["agent", "human", "indexer"]


def _collect_mutable_event_fields() -> dict[str, tuple[str, ...]]:
    """Build the static-analysis registry from each event's hook contract."""
    registry: dict[str, tuple[str, ...]] = {}
    for candidate in tuple(globals().values()):
        if (
            not isinstance(candidate, type)
            or candidate is Event
            or not issubclass(candidate, Event)
        ):
            continue
        hook = getattr(candidate, "HOOK", None)
        mutable_fields = getattr(hook, "mutable_fields", ())
        if mutable_fields:
            registry[candidate.__name__] = tuple(mutable_fields)
    return registry


MUTABLE_EVENT_FIELDS_BY_TYPE: Final[dict[str, tuple[str, ...]]] = (
    _collect_mutable_event_fields()
)


__all__ = [
    "AfterCompactEvent",
    "AgentEndEvent",
    "AgentStartEvent",
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "BackgroundActivityEvent",
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
    "ExtensionInstallEvent",
    "ExtensionReloadEvent",
    "ExtensionUnloadEvent",
    "HookContract",
    "Inject",
    "InputEvent",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "LoopAction",
    "MaxTurnsExhausted",
    "MessageAppendedEvent",
    "MessagePersistedEvent",
    "ModelEndTurn",
    "MUTABLE_EVENT_FIELDS_BY_TYPE",
    "NoPendingInput",
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
