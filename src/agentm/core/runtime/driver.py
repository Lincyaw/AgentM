"""Session driver — persistent loop that converts triggers into committed turns.

Design:
- Outer loop: one trigger → one Turn.  Exits on shutdown / queue-closed / max_turns.
- Inner loop (_react_loop): ReAct rounds within one turn.  Exits on Stop action.
- interrupt: aborts current turn only (SignalAborted), driver continues.
- shutdown: aborts current turn AND exits driver.
- ToolTerminate: sets shutdown, turn commits, driver exits.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from loguru import logger

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)
from agentm.core.abi.stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    StreamFn,
    TextDelta,
)
from agentm.core.abi.tool import (
    Tool,
    ToolContinue,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.tool_executor import execute_tool_call
from agentm.core.abi.bus import EventBus
from agentm.core.abi.context import ContextPolicy, build_context, render_trigger
from agentm.core.abi.events import (
    BeforeRunEvent,
    BeforeSendEvent,
    ContextEvent,
    DecideEvent,
    DiagnosticEvent,
    Inject,
    LoopAction,
    ModelEndTurn,
    ProviderTruncated,
    RunEndEvent,
    SignalAborted,
    Step,
    Stop,
    StreamDeltaEvent,
    TerminationCause,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolTerminated,
    TurnBeginEvent,
    TurnCommittedEvent,
    TurnObservation,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import (
    Outcome,
    ToolRecord,
    TurnMeta,
)
from agentm.core.abi.trigger import (
    BackgroundCompletion,
    SubagentResult,
    Trigger,
    TriggerRenderer,
)
from agentm.core.abi.lifecycle import AbandonEvent, LifecycleHookRegistry
from agentm.core.runtime.execution import Execution
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.trigger_queue import QueueClosed, TriggerQueue

ThinkingLevel = Literal["off", "low", "medium", "high"]


# --- Helpers ----------------------------------------------------------------

def _is_aborted(interrupt: asyncio.Event, shutdown: asyncio.Event) -> bool:
    return interrupt.is_set() or shutdown.is_set()


def _last_of(returns: list[Any], typ: type) -> Any | None:
    chosen = None
    for value in returns:
        if isinstance(value, typ):
            chosen = value
    return chosen


def _last_key(returns: list[Any], key: str) -> Any | None:
    chosen = None
    for value in returns:
        if isinstance(value, dict) and value.get(key) is not None:
            chosen = value[key]
    return chosen


def _last_messages(returns: list[Any]) -> list[AgentMessage] | None:
    chosen = None
    for value in returns:
        if isinstance(value, list):
            chosen = value
        elif isinstance(value, dict):
            msgs = value.get("messages")
            if isinstance(msgs, list):
                chosen = msgs
    return chosen


def _extract_tool_calls(msg: AssistantMessage) -> list[ToolCallBlock]:
    return [b for b in msg.content if isinstance(b, ToolCallBlock)]


def _assemble_assistant_message(
    events: list[AssistantStreamEvent],
) -> AssistantMessage:
    for ev in reversed(events):
        if isinstance(ev, MessageEnd):
            return ev.message
    text_buf: list[str] = []
    for ev in events:
        if isinstance(ev, TextDelta):
            text_buf.append(ev.text)
    content: list[Any] = []
    if text_buf:
        content.append(TextContent(type="text", text="".join(text_buf)))
    return AssistantMessage(
        role="assistant", content=content, timestamp=time.time(),
        stop_reason="incomplete_stream",
    )


def _default_action(
    response: AssistantMessage,
    tool_outcomes: list[tuple[str, ToolOutcome]],
) -> LoopAction:
    for name, out in tool_outcomes:
        if isinstance(out, ToolTerminate):
            return Stop(cause=ToolTerminated(tool_name=name, reason=out.reason))
    if not tool_outcomes:
        from agentm.core.abi.termination import MaxTokens, PauseTurn, ProviderError
        hint = response.termination
        if hint is None:
            return Stop(cause=ModelEndTurn())
        if isinstance(hint, MaxTokens):
            return Stop(cause=ProviderTruncated(kind="max_tokens"))
        if isinstance(hint, PauseTurn):
            return Step()
        if isinstance(hint, ProviderError):
            return Stop(cause=ProviderTruncated(kind="error"))
        return Stop(cause=ModelEndTurn())
    return Step()


def _resolve_action(default: LoopAction, returns: list[Any]) -> LoopAction:
    if isinstance(default, Stop) and not getattr(default.cause, "overridable", True):
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
        return Inject(messages=tuple(inject_msgs))
    if stop_override is not None:
        return stop_override
    if has_step:
        return Step()
    return default


def _trigger_carries_terminal(trigger: Trigger) -> TerminationCause | None:
    if isinstance(trigger, BackgroundCompletion) and trigger.terminal:
        return ToolTerminated(tool_name=f"background:{trigger.task_id}", reason="terminal")
    if isinstance(trigger, SubagentResult) and trigger.terminal:
        return ToolTerminated(tool_name=f"subagent:{trigger.child_session_id}", reason="terminal")
    return None


def _meta(inp: int, out: int, start_ns: int, model: Any = None,
          cache_read: int = 0, cache_write: int = 0) -> TurnMeta:
    return TurnMeta(
        total_input_tokens=inp, total_output_tokens=out,
        cache_read_tokens=cache_read, cache_write_tokens=cache_write,
        duration_ns=time.perf_counter_ns() - start_ns,
        model_id=getattr(model, "id", None),
    )


def _execution_to_messages(
    execution: Execution,
    trigger_messages: list[AgentMessage],
) -> list[AgentMessage]:
    messages: list[AgentMessage] = list(trigger_messages)
    inject_by_round: dict[int, list[AgentMessage]] = {}
    for round_idx, msgs in execution.injected:
        inject_by_round.setdefault(round_idx, []).extend(msgs)

    for i, rnd in enumerate(execution.rounds):
        messages.append(rnd.response)
        if rnd.tool_results:
            result_blocks = [
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id=tr.call.id,
                    content=list(tr.result.content),
                    is_error=tr.result.is_error,
                )
                for tr in rnd.tool_results
            ]
            messages.append(ToolResultMessage(
                role="tool_result", content=result_blocks, timestamp=0.0,
            ))
        if i in inject_by_round:
            messages.extend(inject_by_round[i])
    return messages


# --- Main driver ------------------------------------------------------------


async def drive(
    *,
    trajectory: Trajectory,
    triggers: TriggerQueue,
    bus: EventBus,
    stream_fn: StreamFn,
    model: Model,
    tools: list[Tool],
    store: TrajectoryStore | None = None,
    session_id: str = "",
    system: str | None = None,
    context_policies: list[ContextPolicy] | None = None,
    trigger_renderers: dict[str, TriggerRenderer] | None = None,
    interrupt: asyncio.Event | None = None,
    shutdown: asyncio.Event | None = None,
    max_turns: int | None = None,
    thinking: ThinkingLevel = "off",
    lifecycle: LifecycleHookRegistry | None = None,
) -> None:
    """Persistent driver loop.

    Processes triggers one at a time.  Each trigger becomes one Turn
    (potentially with multiple ReAct rounds).  Exits when:
    - shutdown is set
    - trigger queue is closed (QueueClosed)
    - max_turns committed turns reached
    - a session-terminal cause fires (ToolTerminated, BudgetExhausted)
    """

    _interrupt = interrupt or asyncio.Event()
    _shutdown = shutdown or asyncio.Event()
    policies = context_policies or []
    turns_run = 0

    while True:
        if _shutdown.is_set():
            await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
            return

        try:
            trigger = await triggers.wait()
        except QueueClosed:
            await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
            return

        if max_turns is not None and turns_run >= max_turns:
            await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
            return

        _interrupt.clear()

        execution = trajectory.begin(trigger)
        await bus.emit(TurnBeginEvent.CHANNEL, TurnBeginEvent(
            index=execution.index, trigger=trigger,
        ))

        try:
            outcome, meta = await _react_loop(
                execution=execution,
                trajectory=trajectory,
                trigger=trigger,
                bus=bus,
                stream_fn=stream_fn,
                model=model,
                tools=tools,
                system=system,
                policies=policies,
                trigger_renderers=trigger_renderers,
                interrupt=_interrupt,
                shutdown=_shutdown,
                thinking=thinking,
                store=store,
                session_id=session_id,
            )

            turn = trajectory.commit(outcome, meta)
            if store is not None:
                try:
                    store.append(session_id, turn)
                except Exception:
                    logger.exception("store.append failed; turn committed but not persisted")

            try:
                await bus.emit(TurnCommittedEvent.CHANNEL, TurnCommittedEvent(turn=turn))
            except asyncio.CancelledError:
                logger.debug("TurnCommittedEvent emit interrupted by cancellation")
            turns_run += 1

            if getattr(outcome.cause, "session_terminal", False):
                await bus.emit(RunEndEvent.CHANNEL, RunEndEvent(
                    outcome=turn.outcome, meta=turn.meta,
                ))
                return

        except asyncio.CancelledError:
            await _fire_abandon(lifecycle, session_id, execution)
            trajectory.abandon()
            raise
        except Exception:
            await _fire_abandon(lifecycle, session_id, execution)
            trajectory.abandon()
            logger.exception("driver: round raised; abandoning turn")
            try:
                await bus.emit(DiagnosticEvent.CHANNEL, DiagnosticEvent(
                    level="error",
                    source="driver",
                    message="turn abandoned: trigger dropped due to internal error",
                ))
            except Exception:
                logger.debug("diagnostic emit failed after turn abandon; non-fatal")


async def _react_loop(
    *,
    execution: Execution,
    trajectory: Trajectory,
    trigger: Trigger,
    bus: EventBus,
    stream_fn: StreamFn,
    model: Model,
    tools: list[Tool],
    system: str | None,
    policies: list[ContextPolicy],
    trigger_renderers: dict[str, TriggerRenderer] | None,
    interrupt: asyncio.Event,
    shutdown: asyncio.Event,
    thinking: ThinkingLevel,
    store: TrajectoryStore | None,
    session_id: str,
) -> tuple[Outcome, TurnMeta]:
    """ReAct loop within one turn.  Returns when a Stop action fires."""

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    start_ns = time.perf_counter_ns()
    trigger_messages = render_trigger(trigger, trigger_renderers)

    history_messages = await build_context(trajectory.turns, policies, trigger_renderers)
    messages = list(history_messages) + list(trigger_messages)

    before_returns = await bus.emit(BeforeRunEvent.CHANNEL, BeforeRunEvent(
        messages=tuple(messages), system=system,
    ))
    veto = _last_key(before_returns, "veto")
    if veto is not None:
        return Outcome(cause=veto), _meta(0, 0, start_ns)
    replacement_msgs = _last_key(before_returns, "messages")
    if isinstance(replacement_msgs, list):
        messages = replacement_msgs
    replacement_sys = _last_key(before_returns, "system")
    if isinstance(replacement_sys, str):
        system = replacement_sys

    round_index = 0
    while True:
        if _is_aborted(interrupt, shutdown):
            return Outcome(cause=SignalAborted()), _meta(
                total_input, total_output, start_ns,
                cache_read=total_cache_read, cache_write=total_cache_write,
            )

        if round_index > 0:
            history_messages = await build_context(trajectory.turns, policies, trigger_renderers)
            round_messages = _execution_to_messages(execution, trigger_messages)
            messages = list(history_messages) + round_messages

        ctx_returns = await bus.emit(
            ContextEvent.CHANNEL,
            ContextEvent(messages=tuple(messages), turn_index=execution.index),
        )
        ctx_replacement = _last_messages(ctx_returns)
        if ctx_replacement is not None:
            messages = ctx_replacement

        send_returns = await bus.emit(
            BeforeSendEvent.CHANNEL,
            BeforeSendEvent(
                messages=tuple(messages), system=system,
                tools=tuple(tools), model=model,
                turn_index=execution.index,
            ),
        )
        effective_system = system
        effective_model = model
        effective_tools = list(tools)
        for ret in send_returns:
            if isinstance(ret, dict):
                if "messages" in ret and isinstance(ret["messages"], list):
                    messages = ret["messages"]
                if "system" in ret and isinstance(ret["system"], str):
                    effective_system = ret["system"]
                if "model" in ret:
                    effective_model = ret["model"]
                if "tools" in ret and isinstance(ret["tools"], list):
                    effective_tools = ret["tools"]

        tool_index = {t.name: t for t in effective_tools}

        stream_events: list[AssistantStreamEvent] = []
        async for ev in stream_fn(
            messages=messages, model=effective_model,
            tools=effective_tools, system=effective_system,
            signal=shutdown, thinking=thinking,
        ):
            stream_events.append(ev)
            await bus.emit(StreamDeltaEvent.CHANNEL, StreamDeltaEvent(
                turn_index=execution.index, delta=ev,
            ))

        response = _assemble_assistant_message(stream_events)
        if response.usage is not None:
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens
            total_cache_read += response.usage.cache_read
            total_cache_write += response.usage.cache_write

        messages.append(response)

        tool_calls = _extract_tool_calls(response)
        tool_records: list[ToolRecord] = []
        paired_outcomes: list[tuple[str, ToolOutcome]] = []

        if tool_calls:
            result_blocks: list[ToolResultBlock] = []
            for tc in tool_calls:
                if _is_aborted(interrupt, shutdown):
                    execution.add_round(response, tool_records)
                    return Outcome(cause=SignalAborted()), _meta(
                        total_input, total_output, start_ns,
                        cache_read=total_cache_read, cache_write=total_cache_write,
                    )

                tc_returns = await bus.emit(ToolCallEvent.CHANNEL, ToolCallEvent(
                    tool_call_id=tc.id, tool_name=tc.name, args=dict(tc.arguments),
                ))
                blocked = _last_key(tc_returns, "block")
                outcome: ToolOutcome

                if blocked:
                    reason = _last_key(tc_returns, "reason") or "blocked"
                    outcome = ToolContinue(result=ToolResult(
                        content=[TextContent(type="text", text=f"blocked: {reason}")],
                        is_error=True,
                    ))
                else:
                    rewrite = _last_key(tc_returns, "rewrite")
                    args = dict(tc.arguments)
                    if isinstance(rewrite, dict):
                        args.update(rewrite)
                    tool = tool_index.get(tc.name)
                    if tool is None:
                        outcome = ToolContinue(result=ToolResult(
                            content=[TextContent(type="text", text=f"unknown tool: {tc.name}")],
                            is_error=True,
                        ))
                    else:
                        try:
                            raw = await execute_tool_call(tool, args, signal=shutdown)
                            outcome = raw if isinstance(raw, ToolOutcome) else ToolContinue(result=raw)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.debug("tool {} raised: {}", tc.name, exc)
                            err_returns = await bus.emit(
                                ToolErrorEvent.CHANNEL,
                                ToolErrorEvent(kind="execution_failed", tool_name=tc.name, reason=str(exc), exception=exc),
                            )
                            err_text = _last_key(err_returns, "text") or f"tool error: {tc.name}: {exc}"
                            outcome = ToolContinue(result=ToolResult(
                                content=[TextContent(type="text", text=err_text)],
                                is_error=True,
                            ))

                result = outcome.result if isinstance(outcome, (ToolContinue, ToolTerminate)) else ToolResult(content=[], is_error=True)
                res_returns = await bus.emit(ToolResultEvent.CHANNEL, ToolResultEvent(
                    tool_call_id=tc.id, tool_name=tc.name, result=result,
                ))
                replaced = _last_of(res_returns, ToolResult)
                final_result = replaced if replaced is not None else result

                result_block = ToolResultBlock(
                    type="tool_result", tool_call_id=tc.id,
                    content=list(final_result.content), is_error=final_result.is_error,
                )
                result_blocks.append(result_block)
                tool_records.append(ToolRecord(call=tc, result=result_block))
                paired_outcomes.append((tc.name, outcome))

            messages.append(ToolResultMessage(
                role="tool_result", content=result_blocks, timestamp=time.time(),
            ))

        execution.add_round(response, tool_records)

        if store is not None:
            try:
                from agentm.core.abi.codec import DEFAULT_CODEC
                round_data = DEFAULT_CODEC._serialize_round(execution.rounds[-1])
                round_data["round_index"] = round_index
                store.append_round(session_id, execution.id, round_data)
            except Exception:
                logger.debug("durable round persist failed; non-fatal")

        terminal_from_trigger = _trigger_carries_terminal(trigger)
        default = _default_action(response, paired_outcomes)
        if terminal_from_trigger is not None and isinstance(default, Step):
            default = Stop(cause=terminal_from_trigger)

        decide_returns = await bus.emit(DecideEvent.CHANNEL, DecideEvent(
            observation=TurnObservation(
                turn_index=execution.index,
                assistant_message=response,
                tool_outcomes=tuple(paired_outcomes),
                default_action=default,
            ),
        ))
        action = _resolve_action(default, decide_returns)

        if isinstance(action, Stop):
            cause = action.cause if action.cause is not None else ModelEndTurn()
            return Outcome(cause=cause), _meta(
                total_input, total_output, start_ns, effective_model,
                cache_read=total_cache_read, cache_write=total_cache_write,
            )

        if isinstance(action, Inject):
            execution.add_injected(list(action.messages))

        round_index += 1


async def _fire_abandon(
    lifecycle: LifecycleHookRegistry | None,
    session_id: str,
    execution: Execution,
) -> None:
    if lifecycle is None:
        return
    try:
        await lifecycle.fire_abandon(AbandonEvent(
            session_id=session_id,
            turn_index=execution.index,
            completed_rounds=tuple(execution.rounds),
        ))
    except Exception:
        logger.debug("lifecycle on_abandon failed; non-fatal")


__all__ = ["ThinkingLevel", "drive"]
