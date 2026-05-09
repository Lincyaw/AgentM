"""Minimal agent loop tying messages, tools, stream, and event bus together.

Implements the seed ``AgentLoop`` referenced across §3 of
`.claude/designs/pluggable-architecture.md`. Per-turn termination semantics
follow the sum-type protocol in `.claude/designs/agent-loop.md`: each turn
ends with one :class:`LoopAction` computed by the kernel from the assistant
message and tool outcomes, optionally overridden by extensions on the
``decide_turn_action`` channel.

Loop sketch (one ``run`` invocation):

    emit "agent_start"
    while turn_index < max_turns:
        if signal.is_set(): _terminate(SignalAborted())
        emit "turn_start"
        emit "context"  (handlers may rewrite the messages list)
        emit "before_send_to_llm"
        stream LLM, assemble AssistantMessage from events
        emit "turn_end"
        append assistant message to messages
        if tool_calls:
            execute each, collect ToolOutcome[]
            append ToolResultMessage
        compute default LoopAction
        emit "decide_turn_action" (handlers may override)
        if Stop:    _terminate(cause)
        if Inject:  extend messages, continue
        if Step:    continue
    _terminate(MaxTurnsExhausted())

``_terminate`` always emits ``decide_turn_action`` then ``agent_end`` so
observability stays symmetric across every termination path; on ``final``
causes the override returns are ignored.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .events import (
    AgentEndEvent,
    AgentStartEvent,
    BeforeSendToLlmEvent,
    BudgetExhausted,
    ContextEvent,
    DecideTurnActionEvent,
    EventBus,
    Inject,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    LoopAction,
    MaxTurnsExhausted,
    ModelEndTurn,
    ProviderProtocolViolation,
    ProviderTruncated,
    SignalAborted,
    Step,
    Stop,
    StreamDeltaEvent,
    TerminationCause,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolTerminated,
    TurnEndEvent,
    TurnObservation,
    TurnStartEvent,
)
from .messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)
from .termination import (
    Aborted,
    EndTurn,
    MaxTokens,
    ProviderError,
    ToolUseExpected,
    VendorSpecific,
)
from .stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    StreamFn,
    ToolCallArgsParseError,
    TextDelta,
)
from .tool import (
    Tool,
    ToolContinue,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)


# --- Config -----------------------------------------------------------------


@dataclass(slots=True)
class LoopConfig:
    """Loop tuning knobs. Defaults are deliberately conservative."""

    max_turns: int = 32
    max_tool_calls: int | None = None


# --- Helpers ----------------------------------------------------------------


def _collect_replacement(returns: list[Any], key: str) -> Any | None:
    """Pick the last non-None value of ``key`` from a list of handler returns.

    Handlers on mutating/replaceable channels return either ``None`` (no
    opinion) or a dict-like object whose entries describe an override
    decision (e.g. ``{"block": True, "reason": "..."}`` on ``tool_call``,
    ``{"messages": [...]}`` on ``context``). The "last non-None wins" rule
    keeps the contract simple: extensions stack, and the most recently
    registered authoritative voice wins.
    """

    chosen: Any | None = None
    for value in returns:
        if value is None:
            continue
        if isinstance(value, dict) and key in value and value[key] is not None:
            chosen = value[key]
        elif not isinstance(value, dict) and key == "":
            # Allow non-dict replacements when key is empty (used for the
            # ToolResult-replacement case where handlers may return the new
            # ToolResult directly).
            chosen = value
    return chosen


def _collect_tool_result_replacement(returns: list[Any]) -> ToolResult | None:
    """Return the last non-None ``ToolResult`` from handler returns."""

    chosen: ToolResult | None = None
    for value in returns:
        if isinstance(value, ToolResult):
            chosen = value
    return chosen


def _collect_context_messages(
    returns: list[Any],
) -> list[AgentMessage] | None:
    """Return the last non-None replacement message list, if any."""

    chosen: list[AgentMessage] | None = None
    for value in returns:
        if isinstance(value, list):
            chosen = value
        elif isinstance(value, dict) and value.get("messages") is not None:
            messages = value["messages"]
            if isinstance(messages, list):
                chosen = messages
    return chosen


def _collect_error(returns: list[Any]) -> BaseException | None:
    """Return the last explicit exception object from handler returns."""

    chosen: BaseException | None = None
    for value in returns:
        if isinstance(value, BaseException):
            chosen = value
    return chosen


def _normalize_tool_output(out: ToolResult | ToolOutcome) -> ToolOutcome:
    """Wrap a bare :class:`ToolResult` in :class:`ToolContinue`.

    Tools may return either a ``ToolResult`` (legacy / simple case) or any
    :class:`ToolOutcome` subclass. The kernel normalizes everything to a
    ``ToolOutcome`` so the per-turn decision logic doesn't have to handle
    two shapes.
    """

    if isinstance(out, ToolOutcome):
        return out
    return ToolContinue(result=out)


def _outcome_result(outcome: ToolOutcome) -> ToolResult:
    """Extract the :class:`ToolResult` from any :class:`ToolOutcome` subclass."""

    if isinstance(outcome, ToolContinue):
        return outcome.result
    if isinstance(outcome, ToolTerminate):
        return outcome.result
    raise TypeError(f"unknown ToolOutcome subclass: {type(outcome).__name__}")


def _default_action(
    assistant_msg: AssistantMessage, tool_outcomes: list[ToolOutcome]
) -> LoopAction:
    """Compute the kernel's default :class:`LoopAction` for a turn.

    Order of precedence:
    1. Any tool returned :class:`ToolTerminate` → ``Stop(ToolTerminated(...))``
       (first wins so the cause maps to the *first* terminal tool call).
    2. No tool calls at all → dispatch on the provider's
       :class:`TerminationHint`:
       - :class:`MaxTokens` → ``Stop(ProviderTruncated(kind="max_tokens"))``
       - :class:`ProviderError` → ``Stop(ProviderTruncated(kind="error"))``
       - :class:`ToolUseExpected` → ``Stop(ProviderProtocolViolation)``
       - :class:`EndTurn` / :class:`Aborted` / :class:`VendorSpecific` /
         missing hint → ``Stop(ModelEndTurn())``
    3. Tools ran successfully and none asked to terminate → ``Step()``.

    If ``termination`` is unset (legacy provider adapters), fall back to the
    raw ``stop_reason`` string for graceful migration.
    """

    for out in tool_outcomes:
        if isinstance(out, ToolTerminate):
            # Recover the tool name from the surrounding loop frame: callers
            # in the loop pair outcomes with their originating tool_call so
            # this helper only needs the outcomes themselves. The loop wraps
            # the cause separately when it has the name handy.
            return Stop(ToolTerminated(tool_name="", reason=out.reason))

    if not tool_outcomes:
        hint = assistant_msg.termination
        if hint is None:
            # Back-compat path: providers that haven't migrated to
            # ``TerminationHint`` yet still populate ``stop_reason`` with the
            # legacy kernel vocabulary.
            raw = assistant_msg.stop_reason
            if raw == "max_tokens":
                return Stop(ProviderTruncated(kind="max_tokens"))
            if raw == "error":
                return Stop(ProviderTruncated(kind="error"))
            if raw == "tool_use":
                return Stop(
                    ProviderProtocolViolation(
                        detail=(
                            "provider reported tool_use but no tool_calls were "
                            "extracted"
                        )
                    )
                )
            return Stop(ModelEndTurn())

        if isinstance(hint, MaxTokens):
            return Stop(ProviderTruncated(kind="max_tokens"))
        if isinstance(hint, ProviderError):
            return Stop(ProviderTruncated(kind="error"))
        if isinstance(hint, ToolUseExpected):
            return Stop(
                ProviderProtocolViolation(
                    detail=(
                        "provider reported tool_use but no tool_calls were "
                        "extracted"
                    )
                )
            )
        # EndTurn / Aborted / VendorSpecific all collapse to a clean end-turn
        # at the kernel layer. ``Aborted`` is normally surfaced via the
        # signal path (``SignalAborted``) before we reach this helper; if a
        # provider reports it on the message itself we still terminate cleanly.
        if isinstance(hint, EndTurn | Aborted | VendorSpecific):
            return Stop(ModelEndTurn())
        # Exhaustive over the ``TerminationHint`` union; the fall-through
        # keeps mypy honest if a new variant is added without updating here.
        return Stop(ModelEndTurn())  # pragma: no cover

    return Step()


def _default_action_with_names(
    assistant_msg: AssistantMessage,
    paired_outcomes: list[tuple[str, ToolOutcome]],
) -> LoopAction:
    """Variant of :func:`_default_action` that knows each outcome's tool name.

    Lifts ``tool_name`` from the loop's per-call bookkeeping into
    :class:`ToolTerminated` so observers can identify *which* terminal tool
    fired without a separate lookup.
    """

    for tool_name, out in paired_outcomes:
        if isinstance(out, ToolTerminate):
            return Stop(ToolTerminated(tool_name=tool_name, reason=out.reason))
    return _default_action(
        assistant_msg, [out for _, out in paired_outcomes]
    )


def _resolve_action(default: LoopAction, returns: list[Any]) -> LoopAction:
    """Reconcile the kernel default with handler-supplied overrides.

    Resolution lattice (see ``.claude/designs/agent-loop.md``):
    1. If ``default`` is ``Stop(cause)`` and ``cause.final`` is True, the
       default wins regardless of what handlers returned.
    2. Among handler returns of type :class:`LoopAction`:
       - Any :class:`Inject` wins; messages from all Injects are
         concatenated in registration order.
       - Else any :class:`Stop` wins; if multiple Stops were returned, the
         last one's cause is used.
       - Else :class:`Step` wins if returned; otherwise the default applies.
    """

    if isinstance(default, Stop) and default.cause.final:
        return default

    overrides = [r for r in returns if isinstance(r, LoopAction)]

    inject_msgs: list[AgentMessage] = []
    stop_override: Stop | None = None
    has_step = False
    for o in overrides:
        if isinstance(o, Inject):
            inject_msgs.extend(o.messages)
        elif isinstance(o, Stop):
            stop_override = o
        elif isinstance(o, Step):
            has_step = True

    if inject_msgs:
        return Inject(messages=inject_msgs)
    if stop_override is not None:
        return stop_override
    if has_step:
        return Step()
    return default


def _assemble_assistant_message(
    events: list[AssistantStreamEvent], *, fallback_timestamp: float
) -> AssistantMessage:
    """Build an :class:`AssistantMessage` from a list of stream events.

    If a terminal :class:`MessageEnd` event is present, its embedded message
    is returned verbatim. Otherwise, deltas are concatenated into a best-
    effort assistant message with ``stop_reason="end_turn"`` so the loop can
    proceed deterministically.
    """

    for ev in reversed(events):
        if isinstance(ev, MessageEnd):
            return ev.message

    # Fallback: assemble from deltas.
    text_buf: list[str] = []
    for ev in events:
        if isinstance(ev, TextDelta):
            text_buf.append(ev.text)
    content: list[Any] = []
    if text_buf:
        content.append(TextContent(type="text", text="".join(text_buf)))
    return AssistantMessage(
        role="assistant",
        content=content,
        timestamp=fallback_timestamp,
        stop_reason="end_turn",
    )


def _extract_tool_calls(message: AssistantMessage) -> list[ToolCallBlock]:
    """Return every ``ToolCallBlock`` in ``message.content`` in order."""

    return [b for b in message.content if isinstance(b, ToolCallBlock)]


def _now() -> float:
    return time.time()


# --- Loop -------------------------------------------------------------------


class AgentLoop:
    """Minimal ReAct-style loop wired to a pluggable ``StreamFn`` and bus."""

    def __init__(
        self,
        *,
        stream_fn: StreamFn,
        bus: EventBus,
        config: LoopConfig | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self._bus = bus
        self._config = config if config is not None else LoopConfig()
        self._next_turn_id = 0

    def set_stream_fn(self, fn: StreamFn) -> None:
        """Replace the active provider for subsequent turns.

        This is the only supported way to swap LLM providers mid-session;
        callers must not mutate the kernel's private stream state directly.
        """

        self._stream_fn = fn

    async def run(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Drive the loop until termination.

        Returns the full updated message list (input messages plus all
        appended assistant and tool-result messages). The terminal
        :class:`AgentEndEvent` carries the :class:`TerminationCause` that
        explains why the loop stopped.
        """

        messages = list(messages)  # local copy; we won't mutate caller's list
        start_returns = await self._bus.emit(
            AgentStartEvent.CHANNEL, AgentStartEvent(messages=messages)
        )
        start_error = _collect_error(start_returns)
        if start_error is not None:
            raise start_error

        max_turns = self._config.max_turns
        max_tool_calls = self._config.max_tool_calls
        tool_calls_used = 0
        last_assistant: AssistantMessage | None = None
        last_turn_index = -1

        try:
            for turn_index in range(max_turns):
                last_turn_index = turn_index
                turn_id = self._next_turn_id
                self._next_turn_id += 1
                # Rebuild dispatch index per turn so atoms registered mid-prompt
                # via ``api.install_atom`` (or any other ``register_tool`` path)
                # become callable on the very next turn within the same prompt.
                tool_index = {t.name: t for t in tools}
                if signal is not None and signal.is_set():
                    return await self._terminate(
                        messages,
                        SignalAborted(),
                        last_assistant=last_assistant,
                        turn_index=turn_index,
                        turn_id=turn_id,
                    )
                if (
                    max_tool_calls is not None
                    and tool_calls_used >= max_tool_calls
                ):
                    return await self._terminate(
                        messages,
                        BudgetExhausted(detail="max_tool_calls"),
                        last_assistant=last_assistant,
                        turn_index=turn_index,
                        turn_id=turn_id,
                    )

                await self._bus.emit(
                    TurnStartEvent.CHANNEL,
                    TurnStartEvent(turn_index=turn_index, turn_id=turn_id),
                )

                # context event — handlers may rewrite message list
                ctx_event = ContextEvent(messages=messages)
                ctx_returns = await self._bus.emit(ContextEvent.CHANNEL, ctx_event)
                replacement = _collect_context_messages(ctx_returns)
                if replacement is not None:
                    messages = list(replacement)
                else:
                    # Handlers may have mutated ctx_event.messages in place.
                    messages = list(ctx_event.messages)

                # Final pre-flight hook: handlers see the exact list that
                # will be passed to StreamFn and may mutate it in place.
                before_send_event = BeforeSendToLlmEvent(
                    messages=messages,
                    model=model,
                    tools=tools,
                    system=system,
                )
                await self._bus.emit(BeforeSendToLlmEvent.CHANNEL, before_send_event)

                # Drain the LLM stream, emitting llm_request_start/end so
                # observers (cost trackers, observability) see request
                # boundaries without wrapping ``stream_fn`` themselves.
                await self._bus.emit(
                    LlmRequestStartEvent.CHANNEL,
                    LlmRequestStartEvent(
                        turn_index=turn_index,
                        message_count=len(messages),
                        tool_count=len(tools),
                        system_chars=len(system or ""),
                        model_id=getattr(model, "id", None),
                        turn_id=turn_id,
                    ),
                )
                stream_events: list[AssistantStreamEvent] = []
                stream_start_ns = time.perf_counter_ns()
                stream_error: str | None = None
                try:
                    async for ev in self._stream_fn(
                        messages=messages,
                        model=model,
                        tools=tools,
                        system=system,
                        signal=signal,
                    ):
                        stream_events.append(ev)
                        # Forward each chunk so presenters (TUI, JSON tap)
                        # can render token-by-token. The kernel still
                        # assembles the full message itself and publishes
                        # it via ``turn_end``; this channel is purely
                        # additive and ignored by everyone else.
                        await self._bus.emit(
                            StreamDeltaEvent.CHANNEL,
                            StreamDeltaEvent(turn_index=turn_index, delta=ev, turn_id=turn_id),
                        )
                        if isinstance(ev, ToolCallArgsParseError):
                            await self._bus.emit(ev.CHANNEL, ev)
                except Exception as exc:
                    stream_error = repr(exc)
                    await self._bus.emit(
                        LlmRequestEndEvent.CHANNEL,
                        LlmRequestEndEvent(
                            turn_index=turn_index,
                            chunk_count=len(stream_events),
                            duration_ns=time.perf_counter_ns() - stream_start_ns,
                            error=stream_error,
                            turn_id=turn_id,
                        ),
                    )
                    raise
                await self._bus.emit(
                    LlmRequestEndEvent.CHANNEL,
                    LlmRequestEndEvent(
                        turn_index=turn_index,
                        chunk_count=len(stream_events),
                        duration_ns=time.perf_counter_ns() - stream_start_ns,
                        error=None,
                        turn_id=turn_id,
                    ),
                )

                assistant_msg = _assemble_assistant_message(
                    stream_events, fallback_timestamp=_now()
                )
                last_assistant = assistant_msg
                messages.append(assistant_msg)
                await self._bus.emit(
                    TurnEndEvent.CHANNEL,
                    TurnEndEvent(
                        turn_index=turn_index,
                        message=assistant_msg,
                        messages=tuple(messages),
                        turn_id=turn_id,
                    ),
                )

                tool_calls = _extract_tool_calls(assistant_msg)
                paired_outcomes: list[tuple[str, ToolOutcome]] = []
                if tool_calls:
                    raw_outcomes = await self._execute_tool_calls(
                        messages=messages,
                        tool_calls=tool_calls,
                        tool_index=tool_index,
                        signal=signal,
                    )
                    if raw_outcomes is None:
                        # Signal tripped mid-tool-execution. Route through
                        # the standard kernel termination so decide_turn_action
                        # and agent_end fire exactly once each.
                        return await self._terminate(
                            messages,
                            SignalAborted(),
                            last_assistant=last_assistant,
                            turn_index=turn_index,
                            turn_id=turn_id,
                        )
                    paired_outcomes = raw_outcomes
                    tool_calls_used += len(paired_outcomes)

                action = await self._dispatch_decision(
                    turn_index=turn_index,
                    turn_id=turn_id,
                    assistant_msg=assistant_msg,
                    paired_outcomes=paired_outcomes,
                )

                if isinstance(action, Stop):
                    return await self._finish_with_cause(messages, action.cause)
                if isinstance(action, Inject):
                    messages.extend(action.messages)
                    continue
                # Step → fall through to next turn

            # Loop fell out without returning → exhausted max_turns.
            return await self._terminate(
                messages,
                MaxTurnsExhausted(),
                last_assistant=last_assistant,
                turn_index=last_turn_index,
                turn_id=max(0, self._next_turn_id - 1),
            )

        except asyncio.CancelledError:
            # Hard task cancellation (vs. cooperative ``signal`` abort).
            # We do not emit ``agent_end`` here: awaiting in a cancelled
            # task is unsafe (the next ``await`` re-raises CancelledError).
            # Cooperative aborts go through ``_terminate(SignalAborted)``
            # via the ``raw_outcomes is None`` branch above and preserve
            # the decide_turn_action → agent_end pairing.
            raise

    async def _execute_tool_calls(
        self,
        *,
        messages: list[AgentMessage],
        tool_calls: list[ToolCallBlock],
        tool_index: dict[str, Tool],
        signal: asyncio.Event | None,
    ) -> list[tuple[str, ToolOutcome]] | None:
        """Run each ``tool_call`` sequentially, collecting paired outcomes.

        Mutates ``messages`` to append a single :class:`ToolResultMessage`
        containing every result block.

        Returns ``None`` if ``signal`` trips mid-flight so the caller can
        route the abort through :meth:`_terminate` (preserving the
        decide_turn_action → agent_end pairing). If a tool raises
        :class:`asyncio.CancelledError`, the exception propagates: the
        agent task has been forcibly cancelled and emitting events from
        a cancelled context is unsafe.
        """

        result_blocks: list[ToolResultBlock] = []
        paired: list[tuple[str, ToolOutcome]] = []
        for tc in tool_calls:
            if signal is not None and signal.is_set():
                # Mid-flight cooperative abort: signal the caller via the
                # ``None`` sentinel so it can run the standard termination.
                return None

            tc_event = ToolCallEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                args=dict(tc.arguments),  # mutable copy for handlers
            )
            call_returns = await self._bus.emit(ToolCallEvent.CHANNEL, tc_event)
            blocked = _collect_replacement(call_returns, "block")
            outcome: ToolOutcome
            if blocked:
                reason = (
                    _collect_replacement(call_returns, "reason")
                    or "blocked by extension"
                )
                outcome = ToolContinue(
                    result=await self._make_error_result(
                        kind="blocked",
                        tool_name=tc.name,
                        reason=str(reason),
                        exception=None,
                    )
                )
            else:
                tool = tool_index.get(tc.name)
                if tool is None:
                    outcome = ToolContinue(
                        result=await self._make_error_result(
                            kind="unknown_tool",
                            tool_name=tc.name,
                            reason=tc.name,
                            exception=None,
                        )
                    )
                else:
                    try:
                        raw_out = await tool.execute(
                            tc_event.args, signal=signal
                        )
                    except asyncio.CancelledError:
                        # Hard cancellation: propagate without emitting
                        # any events — see the docstring above.
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # Uniform exception → error-result conversion.
                        outcome = ToolContinue(
                            result=await self._make_error_result(
                                kind="execution_failed",
                                tool_name=tc.name,
                                reason=str(exc),
                                exception=exc,
                            )
                        )
                    else:
                        outcome = _normalize_tool_output(raw_out)

            result = _outcome_result(outcome)
            res_event = ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=result,
            )
            res_returns = await self._bus.emit(ToolResultEvent.CHANNEL, res_event)
            replaced = _collect_tool_result_replacement(res_returns)
            final_result = replaced if replaced is not None else res_event.result

            # If the result was replaced, propagate the replacement into the
            # outcome so downstream consumers see the same payload the loop
            # records on the message list.
            if replaced is not None:
                outcome = (
                    ToolTerminate(result=final_result, reason=outcome.reason)
                    if isinstance(outcome, ToolTerminate)
                    else ToolContinue(result=final_result)
                )

            result_blocks.append(
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id=tc.id,
                    content=list(final_result.content),
                    is_error=final_result.is_error,
                )
            )
            paired.append((tc.name, outcome))

        messages.append(
            ToolResultMessage(
                role="tool_result",
                content=result_blocks,
                timestamp=_now(),
            )
        )
        return paired

    async def _make_error_result(
        self,
        *,
        kind: str,
        tool_name: str,
        reason: str,
        exception: BaseException | None,
    ) -> ToolResult:
        """Build the empty-content ``ToolResult`` for one of the three error
        paths and let the ``tool_error`` channel populate ``content``.

        The kernel deliberately does not synthesize the user-visible English
        string itself; that policy is owned by the ``tool_error_messages``
        builtin atom (or whatever extension replaces it). When no atom is
        installed, the result still carries a single ``TextContent`` with a
        bare fall-back so the trajectory remains debuggable.
        """

        # ``Literal`` already constrains the kernel call sites; the str hop
        # here is a defensive cast for the dataclass's frozen Literal field.
        narrowed: Any = kind
        result = ToolResult(content=[], is_error=True)
        await self._bus.emit(
            ToolErrorEvent.CHANNEL,
            ToolErrorEvent(
                kind=narrowed,
                tool_name=tool_name,
                reason=reason,
                result=result,
                exception=exception,
            ),
        )
        if not result.content:
            # Recovery floor: no atom subscribed (or all subscribers were
            # no-ops). Insert a minimal placeholder so observers and the
            # provider both see *something* legible.
            result.content.append(
                TextContent(
                    type="text",
                    text=f"tool_error: {kind} ({tool_name})",
                )
            )
        return result

    async def _dispatch_decision(
        self,
        *,
        turn_index: int,
        assistant_msg: AssistantMessage,
        turn_id: int,
        paired_outcomes: list[tuple[str, ToolOutcome]],
    ) -> LoopAction:
        """Compute the default action, fire the hook, resolve overrides."""

        default = _default_action_with_names(assistant_msg, paired_outcomes)
        observation = TurnObservation(
            turn_index=turn_index,
            assistant_message=assistant_msg,
            tool_outcomes=[out for _, out in paired_outcomes],
            default_action=default,
            turn_id=turn_id,
        )
        returns = await self._bus.emit(
            DecideTurnActionEvent.CHANNEL,
            DecideTurnActionEvent(observation=observation),
        )
        return _resolve_action(default, returns)

    async def _finish_with_cause(
        self, messages: list[AgentMessage], cause: TerminationCause
    ) -> list[AgentMessage]:
        """Emit ``agent_end`` for ``cause`` and return ``messages`` unchanged.

        Used by the in-loop terminal paths (``Stop`` from a decision or a
        cancel mid-tool-execution). The kernel-imposed paths (signal /
        max_turns) go through :meth:`_terminate` which also fires the
        decision hook for observability symmetry.
        """

        await self._bus.emit(
            AgentEndEvent.CHANNEL, AgentEndEvent(messages=messages, cause=cause)
        )
        return messages

    async def _terminate(
        self,
        messages: list[AgentMessage],
        cause: TerminationCause,
        *,
        last_assistant: AssistantMessage | None,
        turn_index: int,
        turn_id: int,
    ) -> list[AgentMessage]:
        """Kernel-imposed termination path.

        Fires :class:`DecideTurnActionEvent` with ``Stop(cause)`` as the
        default so observability sees every termination, then calls
        :meth:`_finish_with_cause`. ``cause.final`` is True for every caller
        of this helper, so any handler-supplied overrides are ignored —
        we still surface them through the bus so observers can record them.
        """

        observation = TurnObservation(
            turn_index=turn_index,
            assistant_message=last_assistant,
            tool_outcomes=[],
            default_action=Stop(cause),
            turn_id=turn_id,
        )
        await self._bus.emit(
            DecideTurnActionEvent.CHANNEL,
            DecideTurnActionEvent(observation=observation),
        )
        return await self._finish_with_cause(messages, cause)


__all__ = ["AgentLoop", "LoopConfig"]
