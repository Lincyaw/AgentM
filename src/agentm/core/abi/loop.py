"""Minimal agent loop tying messages, tools, stream, and event bus together.

Implements the seed ``AgentLoop`` referenced across §3 of
`.claude/designs/pluggable-architecture.md`. Per-turn termination semantics
follow the sum-type protocol in `.claude/designs/agent-loop.md`: each turn
ends with one :class:`LoopAction` computed by the kernel from the assistant
message and tool outcomes, optionally overridden by extensions on the
``decide_turn_action`` channel.

Loop sketch (one ``run`` invocation):

    emit "agent_start"
    while max_turns is None or turn_index < max_turns:   # None ⇒ no cap
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
import itertools
import json
from loguru import logger
import os
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TypeVar

from .bus import EventBus
from .events import (
    AgentEndEvent,
    AgentStartEvent,
    BeforeSendToLlmEvent,
    BudgetExhausted,
    ContextEvent,
    DecideTurnActionEvent,
    DiagnosticEvent,
    Inject,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    LoopAction,
    MaxTurnsExhausted,
    MessagePersistedEvent,
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
    PauseTurn,
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
from .tool_executor import execute_tool_call




# --- Config -----------------------------------------------------------------


@dataclass(slots=True)
class LoopConfig:
    """Loop tuning knobs.

    ``max_turns`` defaults to ``None`` — no turn cap. The agent then runs
    until it terminates on its own (``ModelEndTurn``), a tool/budget cap
    (``max_tool_calls``), an abort signal, or a provider error. A positive
    int reinstates a hard ceiling that ends the run with
    ``MaxTurnsExhausted`` once reached. Set a cap via the CLI
    (``--max-turns``), the ``loop_budget`` atom listed in a scenario
    manifest, or an explicit ``LoopConfig`` in embedded use.
    """

    max_turns: int | None = None
    max_tool_calls: int | None = None
    max_tool_calls_per_turn: int | None = None


# --- Helpers ----------------------------------------------------------------


T = TypeVar("T")

_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
_PROMPT_DEBUG_STRING_LIMIT = 20_000
_PROMPT_DEBUG_SEQUENCE_LIMIT = 200
_PROMPT_DEBUG_DEPTH_LIMIT = 12


def _last_of(returns: list[Any], typ: type[T]) -> T | None:
    """Return the last value matching ``typ`` (last-wins scan)."""
    chosen: T | None = None
    for value in returns:
        if isinstance(value, typ):
            chosen = value
    return chosen


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _ENV_TRUE_VALUES


def _debug_string(value: str) -> str:
    if len(value) <= _PROMPT_DEBUG_STRING_LIMIT:
        return value
    omitted = len(value) - _PROMPT_DEBUG_STRING_LIMIT
    return f"{value[:_PROMPT_DEBUG_STRING_LIMIT]}\n... <truncated {omitted} chars>"


def _debug_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth >= _PROMPT_DEBUG_DEPTH_LIMIT:
        return f"<truncated depth {depth}>"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _debug_string(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= _PROMPT_DEBUG_SEQUENCE_LIMIT:
                out["..."] = f"<truncated {len(value) - index} keys>"
                break
            out[str(key)] = _debug_jsonable(child, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        out_list = [
            _debug_jsonable(child, depth=depth + 1)
            for child in seq[:_PROMPT_DEBUG_SEQUENCE_LIMIT]
        ]
        if len(seq) > _PROMPT_DEBUG_SEQUENCE_LIMIT:
            out_list.append(f"<truncated {len(seq) - _PROMPT_DEBUG_SEQUENCE_LIMIT} items>")
        return out_list
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _debug_jsonable(getattr(value, field.name), depth=depth + 1)
            for field in fields(value)
        }
    return repr(value)


def _tool_prompt_view(tool: Tool) -> dict[str, Any]:
    return {
        "name": getattr(tool, "name", ""),
        "description": _debug_jsonable(getattr(tool, "description", "")),
        "parameters": _debug_jsonable(getattr(tool, "parameters", {})),
        "metadata": _debug_jsonable(getattr(tool, "metadata", {})),
    }


def _llm_prompt_dump_payload(
    *,
    turn_index: int,
    turn_id: int,
    messages: list[AgentMessage],
    model: Model,
    tools: list[Tool],
    system: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "turn_index": turn_index,
        "turn_id": turn_id,
        "model": _debug_jsonable(model),
        "system": _debug_jsonable(system or ""),
        "messages": _debug_jsonable(messages),
        "tools": [_tool_prompt_view(tool) for tool in tools],
    }


async def _emit_llm_prompt_dump(
    bus: EventBus,
    *,
    turn_index: int,
    turn_id: int,
    messages: list[AgentMessage],
    model: Model,
    tools: list[Tool],
    system: str | None,
    dry_run: bool,
) -> None:
    payload = _llm_prompt_dump_payload(
        turn_index=turn_index,
        turn_id=turn_id,
        messages=messages,
        model=model,
        tools=tools,
        system=system,
        dry_run=dry_run,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    logger.info("LLM prompt dump (dry_run={}):\n{}", dry_run, text)
    await bus.emit(
        DiagnosticEvent.CHANNEL,
        DiagnosticEvent(
            level="info",
            source="llm_prompt_dump",
            message=text,
        ),
    )


def _dry_run_assistant_message(
    *,
    turn_index: int,
    turn_id: int,
    model: Model,
    messages: list[AgentMessage],
    tools: list[Tool],
) -> AssistantMessage:
    text = (
        "# AgentM LLM Prompt Dry Run\n\n"
        "Provider call skipped because `AGENTM_LLM_PROMPT_DRY_RUN` is enabled. "
        "Read the `llm_prompt_dump` diagnostic in this session trace for the "
        "full preflight prompt payload.\n\n"
        f"- turn_index: {turn_index}\n"
        f"- turn_id: {turn_id}\n"
        f"- model: {getattr(model, 'id', '')}\n"
        f"- message_count: {len(messages)}\n"
        f"- tool_count: {len(tools)}"
    )
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=_now(),
        stop_reason="end_turn",
        termination=EndTurn(),
    )


def _last_key(returns: list[Any], key: str) -> Any | None:
    """Return the last non-None ``returns[i][key]`` from dict returns."""
    chosen: Any | None = None
    for value in returns:
        if isinstance(value, dict) and value.get(key) is not None:
            chosen = value[key]
    return chosen


def _last_messages(returns: list[Any]) -> list[AgentMessage] | None:
    r"""Return the last replacement message list (bare list or ``{"messages": [...]}``\ )."""
    chosen: list[AgentMessage] | None = None
    for value in returns:
        if isinstance(value, list):
            chosen = value
        elif isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list):
                chosen = messages
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


def default_loop_action(
    assistant_msg: AssistantMessage,
    paired_outcomes: list[tuple[str, ToolOutcome]],
) -> LoopAction:
    """Compute the kernel's default :class:`LoopAction` for a turn.

    ``paired_outcomes`` carries each :class:`ToolOutcome` together with the
    name of the originating tool call so :class:`ToolTerminated` can identify
    *which* terminal tool fired without a separate lookup.

    Order of precedence:
    1. Any tool returned :class:`ToolTerminate` → ``Stop(ToolTerminated(...))``
       (first wins so the cause maps to the *first* terminal tool call).
    2. No tool calls at all → dispatch on the provider's
       :class:`TerminationHint`:
       - :class:`MaxTokens` → ``Stop(ProviderTruncated(kind="max_tokens"))``
       - :class:`PauseTurn` → ``Step()`` (resend with partial reply in history)
       - :class:`ProviderError` → ``Stop(ProviderTruncated(kind="error"))``
       - :class:`ToolUseExpected` → ``Stop(ProviderProtocolViolation)``
       - :class:`EndTurn` / :class:`Aborted` / :class:`VendorSpecific` /
         missing hint → ``Stop(ModelEndTurn())``
    3. Tools ran successfully and none asked to terminate → ``Step()``.

    Provider adapters must always populate ``termination``; the raw
    ``stop_reason`` string is not part of the kernel contract.
    """

    for tool_name, out in paired_outcomes:
        if isinstance(out, ToolTerminate):
            return Stop(ToolTerminated(tool_name=tool_name, reason=out.reason))

    if not paired_outcomes:
        hint = assistant_msg.termination
        if hint is None:
            return Stop(ModelEndTurn())

        if isinstance(hint, MaxTokens):
            return Stop(ProviderTruncated(kind="max_tokens"))
        if isinstance(hint, PauseTurn):
            # Provider paused mid-turn; the assistant message is partial.
            # The kernel already appended it to the message list before
            # this decision, so stepping into another turn resends the
            # full conversation (now including the partial reply) and the
            # model resumes from where it stopped. ``max_turns`` still
            # caps any pathological loop.
            return Step()
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


def resolve_loop_action(default: LoopAction, returns: list[Any]) -> LoopAction:
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
        start_error = _last_of(start_returns, BaseException)
        if start_error is not None:
            raise start_error

        max_turns = self._config.max_turns
        max_tool_calls = self._config.max_tool_calls
        tool_calls_used = 0
        last_assistant: AssistantMessage | None = None
        last_turn_index = -1

        # ``max_turns is None`` ⇒ no turn cap: iterate forever and rely on an
        # in-loop ``return`` (ModelEndTurn / budget / signal / error) to end
        # the run. A finite cap uses ``range`` so falling off the end below
        # surfaces ``MaxTurnsExhausted``.
        turn_iter = itertools.count() if max_turns is None else range(max_turns)

        try:
            for turn_index in turn_iter:
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
                replacement = _last_messages(ctx_returns)
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
                prompt_dry_run = _env_enabled("AGENTM_LLM_PROMPT_DRY_RUN")
                if prompt_dry_run or _env_enabled("AGENTM_LLM_PROMPT_DUMP"):
                    await _emit_llm_prompt_dump(
                        self._bus,
                        turn_index=turn_index,
                        turn_id=turn_id,
                        messages=messages,
                        model=model,
                        tools=tools,
                        system=system,
                        dry_run=prompt_dry_run,
                    )

                # Drain the LLM stream, emitting llm_request_start/end so
                # observers (cost trackers, observability) see request
                # boundaries without wrapping ``stream_fn`` themselves.
                _skip_system = os.environ.get(
                    "AGENTM_TRACE_SYSTEM_PROMPT", ""
                ).strip().lower() in {"0", "false", "no", "off"}
                await self._bus.emit(
                    LlmRequestStartEvent.CHANNEL,
                    LlmRequestStartEvent(
                        turn_index=turn_index,
                        message_count=len(messages),
                        tool_count=len(tools),
                        system_chars=len(system or ""),
                        model_id=getattr(model, "id", None),
                        turn_id=turn_id,
                        system_text=None if _skip_system else (system or ""),
                    ),
                )
                stream_events: list[AssistantStreamEvent] = []
                stream_start_ns = time.perf_counter_ns()
                stream_error: str | None = None
                try:
                    if prompt_dry_run:
                        stream_events.append(
                            MessageEnd(
                                _dry_run_assistant_message(
                                    turn_index=turn_index,
                                    turn_id=turn_id,
                                    model=model,
                                    messages=messages,
                                    tools=tools,
                                )
                            )
                        )
                    else:
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
                    MessagePersistedEvent.CHANNEL,
                    MessagePersistedEvent(
                        message=assistant_msg,
                        source="assistant",
                        turn_index=turn_index,
                        turn_id=turn_id,
                    ),
                )
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
                cap = self._config.max_tool_calls_per_turn
                dropped_tool_calls: list[ToolCallBlock] = []
                if cap is not None and len(tool_calls) > cap:
                    dropped_tool_calls = tool_calls[cap:]
                    tool_calls = tool_calls[:cap]
                    logger.warning(f"turn {turn_index}: {len(tool_calls) + len(dropped_tool_calls)} tool calls truncated to {cap} (max_tool_calls_per_turn={cap})")
                paired_outcomes: list[tuple[str, ToolOutcome]] = []
                if tool_calls:
                    raw_outcomes = await self._execute_tool_calls(
                        messages=messages,
                        tool_calls=tool_calls,
                        tool_index=tool_index,
                        signal=signal,
                        turn_index=turn_index,
                        turn_id=turn_id,
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
                if dropped_tool_calls:
                    total = len(dropped_tool_calls) + len(tool_calls)
                    limit = cap or len(tool_calls)
                    drop_msg = (
                        f"Tool call dropped: you issued {total} parallel "
                        f"tool calls but the limit is {limit} per turn. "
                        f"Split your queries across multiple turns."
                    )
                    for tc in dropped_tool_calls:
                        paired_outcomes.append((
                            tc.name,
                            ToolContinue(
                                result=ToolResult(
                                    content=[TextContent(type="text", text=drop_msg)],
                                    is_error=True,
                                ),
                            ),
                        ))

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
                    for injected_msg in action.messages:
                        await self._bus.emit(
                            MessagePersistedEvent.CHANNEL,
                            MessagePersistedEvent(
                                message=injected_msg,
                                source="injected",
                                turn_index=turn_index,
                                turn_id=turn_id,
                            ),
                        )
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
        turn_index: int,
        turn_id: int,
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
            blocked = _last_key(call_returns, "block")
            outcome: ToolOutcome
            if blocked:
                reason = (
                    _last_key(call_returns, "reason")
                    or "blocked by extension"
                )
                block_kind = _last_key(call_returns, "kind") or "blocked"
                if block_kind not in ("blocked", "user_rejected"):
                    block_kind = "blocked"
                outcome = ToolContinue(
                    result=await self._make_error_result(
                        kind=block_kind,
                        tool_name=tc.name,
                        reason=str(reason),
                        exception=None,
                    )
                )
            else:
                rewrite = _last_key(call_returns, "rewrite")
                if isinstance(rewrite, dict):
                    tc_event.args.update(rewrite)
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
                        raw_out = await self._execute_tool_task(
                            tool, tc_event.args, signal=signal
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
            replaced = _last_of(res_returns, ToolResult)
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

        tool_result_msg = ToolResultMessage(
            role="tool_result",
            content=result_blocks,
            timestamp=_now(),
        )
        messages.append(tool_result_msg)
        await self._bus.emit(
            MessagePersistedEvent.CHANNEL,
            MessagePersistedEvent(
                message=tool_result_msg,
                source="tool_result",
                turn_index=turn_index,
                turn_id=turn_id,
            ),
        )
        return paired

    async def _execute_tool_task(
        self,
        tool: Tool,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None,
    ) -> ToolResult | ToolOutcome:
        """Run one tool through the substrate execution boundary."""

        return await execute_tool_call(tool, args, signal=signal)

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

        default = default_loop_action(assistant_msg, paired_outcomes)
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
        return resolve_loop_action(default, returns)

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


__all__ = ["AgentLoop", "LoopConfig", "default_loop_action", "resolve_loop_action"]
