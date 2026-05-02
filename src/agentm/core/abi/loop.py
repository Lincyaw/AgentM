"""Minimal agent loop tying messages, tools, stream, and event bus together.

Implements the seed ``AgentLoop`` referenced across §3 of
`.claude/designs/pluggable-architecture.md`. This is the only kernel module
allowed to import all four others.

Loop sketch (one ``run`` invocation):

    emit "agent_start"
    while turn_index < max_turns:
        if signal.is_set(): bail (stop_reason="aborted")
        emit "turn_start"
        emit "context"  (handlers may rewrite the messages list)
        stream LLM, assemble AssistantMessage from events
        emit "turn_end"
        append assistant message to messages
        if no tool_calls: emit "agent_end" and return
        for each tool_call:
            emit "tool_call" (handlers may mutate args, or block with an error)
            execute tool (or synthesize block error result)
            emit "tool_result" (handlers may replace the result)
            append a ToolResultMessage entry
    emit "agent_end" (stop_reason="max_turns")

Handler return-collection rule (``_collect_replacement``): scan returns in
registration order and pick the **last non-None** value matching ``key``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .events import (
    AgentEndEvent,
    AgentStartEvent,
    BeforeAgentEndEvent,
    BeforeSendToLlmEvent,
    ContextEvent,
    EventBus,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)
from .stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    StreamFn,
    TextDelta,
)
from .tool import Tool, ToolResult


# --- Config -----------------------------------------------------------------


@dataclass(slots=True)
class LoopConfig:
    """Loop tuning knobs. Defaults are deliberately conservative."""

    max_turns: int = 32


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
    """Return the last non-None ``ToolResult`` from handler returns.

    Handlers on ``tool_result`` may directly return a replacement
    :class:`ToolResult`; we accept that shape rather than requiring a dict
    wrapper, mirroring how the seed events are typed.
    """

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


def _collect_before_agent_end_decision(
    returns: list[Any],
) -> tuple[bool, list[AgentMessage]]:
    """Collect cancel + appended-message decisions from handler returns."""

    cancel = False
    appended: list[AgentMessage] = []
    for value in returns:
        if not isinstance(value, dict):
            continue
        if value.get("cancel") is True:
            cancel = True
        raw_append = value.get("append")
        if isinstance(raw_append, list):
            appended.extend(
                message
                for message in raw_append
                if isinstance(
                    message,
                    (AssistantMessage, ToolResultMessage, UserMessage),
                )
            )
    return cancel, appended


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

    async def run(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Drive the loop until ``end_turn``, ``max_turns``, or ``aborted``.

        Returns the full updated message list (input messages plus all
        appended assistant and tool-result messages).
        """

        messages = list(messages)  # local copy; we won't mutate caller's list
        start_returns = await self._bus.emit(
            "agent_start", AgentStartEvent(messages=messages)
        )
        start_error = _collect_error(start_returns)
        if start_error is not None:
            raise start_error

        tool_index = {t.name: t for t in tools}
        max_turns = self._config.max_turns

        try:
            for turn_index in range(max_turns):
                if signal is not None and signal.is_set():
                    await self._bus.emit(
                        "agent_end",
                        AgentEndEvent(messages=messages, stop_reason="aborted"),
                    )
                    return messages

                await self._bus.emit("turn_start", TurnStartEvent(turn_index=turn_index))

                # context event — handlers may rewrite message list
                ctx_event = ContextEvent(messages=messages)
                ctx_returns = await self._bus.emit("context", ctx_event)
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
                await self._bus.emit("before_send_to_llm", before_send_event)

                # Drain the LLM stream, emitting llm_request_start/end so
                # observers (cost trackers, observability) see request
                # boundaries without wrapping ``stream_fn`` themselves.
                await self._bus.emit(
                    "llm_request_start",
                    LlmRequestStartEvent(
                        turn_index=turn_index,
                        message_count=len(messages),
                        tool_count=len(tools),
                        system_chars=len(system or ""),
                        model_id=getattr(model, "id", None),
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
                            "stream_delta",
                            StreamDeltaEvent(turn_index=turn_index, delta=ev),
                        )
                except Exception as exc:
                    stream_error = repr(exc)
                    await self._bus.emit(
                        "llm_request_end",
                        LlmRequestEndEvent(
                            turn_index=turn_index,
                            chunk_count=len(stream_events),
                            duration_ns=time.perf_counter_ns() - stream_start_ns,
                            error=stream_error,
                        ),
                    )
                    raise
                await self._bus.emit(
                    "llm_request_end",
                    LlmRequestEndEvent(
                        turn_index=turn_index,
                        chunk_count=len(stream_events),
                        duration_ns=time.perf_counter_ns() - stream_start_ns,
                        error=None,
                    ),
                )

                assistant_msg = _assemble_assistant_message(
                    stream_events, fallback_timestamp=_now()
                )
                await self._bus.emit(
                    "turn_end",
                    TurnEndEvent(turn_index=turn_index, message=assistant_msg),
                )
                messages.append(assistant_msg)

                tool_calls = _extract_tool_calls(assistant_msg)
                if not tool_calls:
                    before_end_event = BeforeAgentEndEvent(
                        messages=messages,
                        stop_reason="end_turn",
                    )
                    before_end_returns = await self._bus.emit(
                        "before_agent_end", before_end_event
                    )
                    messages = list(before_end_event.messages)
                    cancel_end, appended = _collect_before_agent_end_decision(
                        before_end_returns
                    )
                    if appended:
                        messages.extend(appended)
                    if cancel_end:
                        continue
                    await self._bus.emit(
                        "agent_end",
                        AgentEndEvent(
                            messages=messages,
                            stop_reason=assistant_msg.stop_reason or "end_turn",
                        ),
                    )
                    return messages

                # Execute tool calls sequentially.
                result_blocks: list[ToolResultBlock] = []
                for tc in tool_calls:
                    if signal is not None and signal.is_set():
                        await self._bus.emit(
                            "agent_end",
                            AgentEndEvent(messages=messages, stop_reason="aborted"),
                        )
                        return messages

                    tc_event = ToolCallEvent(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        args=dict(tc.arguments),  # mutable copy for handlers
                    )
                    call_returns = await self._bus.emit("tool_call", tc_event)
                    blocked = _collect_replacement(call_returns, "block")
                    if blocked:
                        reason = (
                            _collect_replacement(call_returns, "reason")
                            or "blocked by extension"
                        )
                        result = ToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"Tool call blocked: {reason}",
                                )
                            ],
                            is_error=True,
                        )
                    else:
                        tool = tool_index.get(tc.name)
                        if tool is None:
                            result = ToolResult(
                                content=[
                                    TextContent(
                                        type="text",
                                        text=f"Unknown tool: {tc.name}",
                                    )
                                ],
                                is_error=True,
                            )
                        else:
                            try:
                                result = await tool.execute(
                                    tc_event.args, signal=signal
                                )
                            except asyncio.CancelledError:
                                # Cooperative cancellation: surface "aborted"
                                # and propagate so callers can clean up.
                                await self._bus.emit(
                                    "agent_end",
                                    AgentEndEvent(
                                        messages=messages, stop_reason="aborted"
                                    ),
                                )
                                raise
                            except Exception as exc:  # noqa: BLE001
                                # Uniform exception → error-result conversion.
                                result = ToolResult(
                                    content=[
                                        TextContent(
                                            type="text",
                                            text=f"Tool execution error: {exc}",
                                        )
                                    ],
                                    is_error=True,
                                )

                    res_event = ToolResultEvent(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result=result,
                    )
                    res_returns = await self._bus.emit("tool_result", res_event)
                    replaced = _collect_tool_result_replacement(res_returns)
                    final_result = replaced if replaced is not None else res_event.result

                    result_blocks.append(
                        ToolResultBlock(
                            type="tool_result",
                            tool_call_id=tc.id,
                            content=list(final_result.content),
                            is_error=final_result.is_error,
                        )
                    )

                messages.append(
                    ToolResultMessage(
                        role="tool_result",
                        content=result_blocks,
                        timestamp=_now(),
                    )
                )

            # Loop fell out without returning → exhausted max_turns.
            before_end_event = BeforeAgentEndEvent(
                messages=messages,
                stop_reason="max_turns",
            )
            before_end_returns = await self._bus.emit(
                "before_agent_end", before_end_event
            )
            messages = list(before_end_event.messages)
            _, appended = _collect_before_agent_end_decision(before_end_returns)
            if appended:
                messages.extend(appended)
            await self._bus.emit(
                "agent_end",
                AgentEndEvent(messages=messages, stop_reason="max_turns"),
            )
            return messages

        except asyncio.CancelledError:
            # Already emitted agent_end above where we caught it; just
            # propagate.
            raise


__all__ = ["AgentLoop", "LoopConfig"]
