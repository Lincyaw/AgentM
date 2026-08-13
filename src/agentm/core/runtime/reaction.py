# code-health: ignore-file[AM025] -- loop normalizes hook returns and event DTO variants
"""One-turn ReAct execution independent of trajectory commit coordination."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace

from agentm.core.abi.bus import EventBus
from agentm.core.abi.cancel import (
    CancelSignal,
    CompositeCancelSignal,
    cancel_reason,
)
from agentm.core.abi.compaction import (
    ContextBudget,
    ContextProjection,
    ProjectionInput,
)
from agentm.core.abi.context import (
    ContextPolicy,
    ContextTransformCancelled,
    apply_context_policies,
    apply_trigger_metadata,
    build_context,
    render_trigger,
    route_messages,
)
from agentm.core.abi.events import (
    BeforeRunEvent,
    BeforeSendEvent,
    DecideEvent,
    Inject,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    LoopAction,
    Step,
    Stop,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    TurnObservation,
)
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    InterruptionMessagePolicy,
    JsonValue,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
    freeze_json,
    thaw_json,
)
from agentm.core.abi.permission import (
    PermissionAudience,
    PermissionPolicy,
    PermissionRequest,
    permission_denial_result,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextDelta,
)
from agentm.core.abi.termination import (
    Aborted,
    BudgetExhausted,
    ModelEndTurn,
    PromptRunContinued,
    ProviderRequestFailed,
    ProviderTruncated,
    SignalAborted,
    TerminationCause,
    ToolTerminated,
)
from agentm.core.abi.tool import (
    Tool,
    ToolContinue,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.tool_executor import tool_execution_requirements
from agentm.core.abi.tool_orchestration import (
    ToolOrchestrationRequest,
    ToolOrchestrationResult,
    ToolWorkItem,
)
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    Outcome,
    ToolRecord,
    TrajectoryNode,
    Turn,
    TurnMeta,
)
from agentm.core.abi.trigger import (
    BackgroundCompletion,
    ContinueTrigger,
    SubagentResult,
    Trigger,
    TriggerRenderer,
)
from agentm.core.runtime.execution import Execution
from agentm.core.runtime.reaction_types import (
    ReactionDependencies,
    ReactionRequest,
    ReactionResult,
)

_INTERRUPTED_TOOL_TEXT = "Tool execution interrupted"


def _signal_aborted(signal: CancelSignal | None) -> SignalAborted:
    return SignalAborted(reason=cancel_reason(signal) or "")


def _last_of[T](returns: Sequence[object], typ: type[T]) -> T | None:
    chosen: T | None = None
    for value in returns:
        if isinstance(value, typ):
            chosen = value
    return chosen


def _last_key(returns: Sequence[object], key: str) -> object | None:
    chosen: object | None = None
    for value in returns:
        if isinstance(value, Mapping) and value.get(key) is not None:
            chosen = value[key]
    return chosen


def _message_list(value: object) -> list[AgentMessage] | None:
    if not isinstance(value, list) or not all(
        isinstance(item, (UserMessage, AssistantMessage, ToolResultMessage))
        for item in value
    ):
        return None
    return [
        item
        for item in value
        if isinstance(item, (UserMessage, AssistantMessage, ToolResultMessage))
    ]


def _reduce_before_run(event: BeforeRunEvent, value: object) -> BeforeRunEvent:
    if not isinstance(value, Mapping):
        return event
    messages = event.messages
    system = event.system
    if "messages" in value:
        replacement_messages = _message_list(value["messages"])
        if replacement_messages is None:
            raise TypeError("BeforeRunEvent messages must be a message list")
        messages = tuple(replacement_messages)
    if "system" in value:
        replacement_system = value["system"]
        if replacement_system is not None and not isinstance(replacement_system, str):
            raise TypeError("BeforeRunEvent system must be str or None")
        system = replacement_system
    return replace(event, messages=messages, system=system)


def _reduce_before_send(event: BeforeSendEvent, value: object) -> BeforeSendEvent:
    if not isinstance(value, Mapping):
        return event
    messages = event.messages
    system = event.system
    model = event.model
    tools = event.tools
    if "messages" in value:
        returned_messages = _message_list(value["messages"])
        if returned_messages is None:
            raise TypeError("BeforeSendEvent messages must be a message list")
        # Append-only. A handler may add to the tail of what it was given and
        # nothing else: the prefix is the committed trajectory, and a handler
        # that rewrote it would put words in front of the model that the
        # record cannot account for. Its own additions stay identifiable as
        # the tail beyond the incoming length.
        incoming = tuple(event.messages)
        returned = tuple(returned_messages)
        if returned[: len(incoming)] != incoming:
            raise TypeError(
                "BeforeSendEvent messages is append-only: a handler may extend "
                "the list it was given but may not alter or drop its entries"
            )
        messages = returned
    if "system" in value:
        replacement_system = value["system"]
        if replacement_system is not None and not isinstance(replacement_system, str):
            raise TypeError("BeforeSendEvent system must be str or None")
        system = replacement_system
    if "model" in value:
        replacement_model = value["model"]
        if not isinstance(replacement_model, Model):
            raise TypeError("BeforeSendEvent model must be a Model")
        model = replacement_model
    if "tools" in value:
        replacement_tools = value["tools"]
        if not isinstance(replacement_tools, list) or not all(
            isinstance(item, Tool) for item in replacement_tools
        ):
            raise TypeError("BeforeSendEvent tools must be a Tool list")
        tools = tuple(replacement_tools)
    return replace(
        event,
        messages=messages,
        system=system,
        model=model,
        tools=tools,
    )


def _extract_tool_calls(msg: AssistantMessage) -> list[ToolCallBlock]:
    return [b for b in msg.content if isinstance(b, ToolCallBlock)]


def _interrupted_tool_record(
    call: ToolCallBlock,
    *,
    reason: str,
    policy: InterruptionMessagePolicy | None,
) -> ToolRecord:
    result = (
        policy.interrupted_tool_result(call.id, reason)
        if policy is not None
        else ToolResultBlock(
            type="tool_result",
            tool_call_id=call.id,
            content=[TextContent(type="text", text=_INTERRUPTED_TOOL_TEXT)],
            is_error=True,
        )
    )
    return ToolRecord(
        call=call,
        result=result,
    )


def _append_interrupted_tool_records(
    *,
    calls: list[ToolCallBlock],
    records: list[ToolRecord],
    result_blocks: list[ToolResultBlock] | None = None,
    reason: str,
    policy: InterruptionMessagePolicy | None,
) -> None:
    for call in calls:
        record = _interrupted_tool_record(call, reason=reason, policy=policy)
        records.append(record)
        if result_blocks is not None:
            result_blocks.append(record.result)


def _cancelled_tool_records(
    *,
    calls: Sequence[ToolCallBlock],
    outcomes: Mapping[int, ToolOutcome],
    reason: str,
    policy: InterruptionMessagePolicy | None,
) -> list[ToolRecord]:
    """Close every tool call, preserving results completed before cancel."""

    records: list[ToolRecord] = []
    for index, call in enumerate(calls):
        outcome = outcomes.get(index)
        if outcome is None:
            records.append(
                _interrupted_tool_record(
                    call,
                    reason=reason,
                    policy=policy,
                )
            )
            continue
        records.append(
            ToolRecord(
                call=call,
                result=_tool_result_block(call.id, _outcome_result(outcome)),
            )
        )
    return records


def record_interruption_message(
    execution: Execution,
    outcome: Outcome,
    policy: InterruptionMessagePolicy | None,
) -> None:
    """Append a policy message after the interrupted request, when required."""

    cause = outcome.cause
    if not isinstance(cause, SignalAborted) or policy is None:
        return
    reason = cause.reason or "unknown"
    for_tool_use = execution.response is not None and bool(
        _extract_tool_calls(execution.response)
    )
    message = policy.interruption_message(
        reason,
        for_tool_use=for_tool_use,
    )
    if message is not None:
        execution.add_injected([message])


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
    content = [TextContent(type="text", text="".join(text_buf))] if text_buf else []
    return AssistantMessage(
        role="assistant",
        content=content,
        timestamp=time.time(),
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


def _resolve_action(default: LoopAction, returns: Sequence[object]) -> LoopAction:
    if isinstance(default, Stop) and not default.cause.overridable:
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
        return ToolTerminated(
            tool_name=f"background:{trigger.task_id}", reason="terminal"
        )
    if isinstance(trigger, SubagentResult) and trigger.terminal:
        return ToolTerminated(
            tool_name=f"subagent:{trigger.child_session_id}", reason="terminal"
        )
    return None


def _outcome_result(outcome: ToolOutcome) -> ToolResult:
    if isinstance(outcome, (ToolContinue, ToolTerminate)):
        return outcome.result
    raise TypeError(f"unsupported tool outcome: {type(outcome).__name__}")


def _replace_outcome_result(
    outcome: ToolOutcome,
    result: ToolResult,
) -> ToolOutcome:
    if isinstance(outcome, ToolContinue):
        return ToolContinue(result=result)
    if isinstance(outcome, ToolTerminate):
        return ToolTerminate(result=result, reason=outcome.reason)
    raise TypeError(f"unsupported tool outcome: {type(outcome).__name__}")


def _normalize_tool_output(output: ToolResult | ToolOutcome) -> ToolOutcome:
    return output if isinstance(output, ToolOutcome) else ToolContinue(result=output)


def _tool_result_block(tool_call_id: str, result: ToolResult) -> ToolResultBlock:
    return ToolResultBlock(
        type="tool_result",
        tool_call_id=tool_call_id,
        content=list(result.content),
        is_error=result.is_error,
        extras=result.extras,
    )


def _tool_result_content_hash(result: ToolResult) -> str:
    digest = hashlib.sha256()
    for block in result.content:
        digest.update(block.type.encode("utf-8"))
        if isinstance(block, TextContent):
            digest.update(block.text.encode("utf-8"))
        else:
            digest.update(block.mime_type.encode("utf-8"))
            digest.update(block.data)
    return digest.hexdigest()[:16]


def _args_hash(args: Mapping[str, object]) -> str:
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _tool_result_exit_code(result: ToolResult) -> int | None:
    if isinstance(result.extras, Mapping):
        for key in ("exit_code", "return_code", "returncode"):
            value = _coerce_int(result.extras.get(key))
            if value is not None:
                return value
    return None


def _with_runtime_tool_metadata(
    result: ToolResult,
    *,
    args: Mapping[str, object],
    duration_ms: int | None,
) -> ToolResult:
    runtime_metadata: dict[str, object] = {
        "args_hash": _args_hash(args),
        "result_content_hash": _tool_result_content_hash(result),
    }
    if duration_ms is not None:
        runtime_metadata["duration_ms"] = duration_ms

    thawed = thaw_json(result.extras)
    if isinstance(thawed, Mapping):
        extras = dict(thawed)
    elif thawed is None:
        extras = {}
    else:
        extras = {"tool_extras": thawed}
    extras["agentm_runtime"] = runtime_metadata
    return ToolResult(
        content=result.content,
        is_error=result.is_error,
        extras=extras,
    )


def _permission_request(
    *,
    tc: ToolCallBlock,
    args: Mapping[str, object],
    session_id: str,
    execution: Execution,
    audience: PermissionAudience,
) -> PermissionRequest:
    return PermissionRequest(
        action="tool_call",
        session_id=session_id,
        turn_id=execution.id,
        turn_index=execution.index,
        tool_call_id=tc.id,
        tool_name=tc.name,
        args=args,
        audience=audience,
    )


async def _permission_outcome(
    *,
    policy: PermissionPolicy | None,
    request: PermissionRequest,
    signal: CancelSignal | None,
) -> ToolOutcome | None:
    if policy is None:
        return None
    decision = await policy.decide(request, signal=signal)
    if decision.allowed:
        return None
    return ToolContinue(result=permission_denial_result(request, decision))


async def _orchestration_failure_outcome(
    *,
    result: ToolOrchestrationResult,
    bus: EventBus,
) -> ToolOutcome:
    if result.status == "cancelled":
        reason = result.cancel_reason or "cancelled"
        return ToolContinue(
            result=ToolResult(
                content=[TextContent(type="text", text=f"tool cancelled: {reason}")],
                is_error=True,
            )
        )
    exc = result.error
    reason = str(exc) if exc is not None else result.status
    err_returns = await bus.emit_decision(
        ToolErrorEvent.CHANNEL,
        ToolErrorEvent(
            kind="execution_failed",
            tool_name=result.item.call.name,
            reason=reason,
            exception=exc,
        ),
    )
    returned_text = _last_key(err_returns, "text")
    err_text = (
        returned_text
        if isinstance(returned_text, str)
        else f"tool error: {result.item.call.name}: {reason}"
    )
    return ToolContinue(
        result=ToolResult(
            content=[TextContent(type="text", text=err_text)],
            is_error=True,
        )
    )


async def _finalize_tool_outcome(
    *,
    bus: EventBus,
    call: ToolCallBlock,
    outcome: ToolOutcome,
    args: Mapping[str, object] | None = None,
    duration_ms: int | None = None,
) -> tuple[ToolRecord, ToolOutcome]:
    frozen_event_args = freeze_json(args or call.arguments)
    if not isinstance(frozen_event_args, Mapping):
        raise TypeError("ToolResultEvent args must be an object")
    event_args: dict[str, JsonValue] = dict(frozen_event_args)
    result = _with_runtime_tool_metadata(
        _outcome_result(outcome),
        args=event_args,
        duration_ms=duration_ms,
    )
    returns = await bus.emit_decision(
        ToolResultEvent.CHANNEL,
        ToolResultEvent(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            args=event_args,
            duration_ms=duration_ms,
            exit_code=_tool_result_exit_code(result),
            result_content_hash=_tool_result_content_hash(result),
        ),
    )
    replacement = _last_of(returns, ToolResult)
    final_result = (
        _with_runtime_tool_metadata(
            replacement, args=event_args, duration_ms=duration_ms
        )
        if replacement is not None
        else result
    )
    final_outcome = _replace_outcome_result(outcome, final_result)
    return (
        ToolRecord(
            call=call,
            result=_tool_result_block(call.id, final_result),
        ),
        final_outcome,
    )


def _tool_schema_digest(tools: Sequence[Tool]) -> str | None:
    """Digest the tool surface the model was actually offered.

    The names and their schemas both matter: an atom that rewrites a
    description changes what the model was told a tool does, and a run whose
    tool surface differed is not the same run.
    """

    if not tools:
        return None
    payload = json.dumps(
        [[tool.name, tool.description, tool.parameters] for tool in tools],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _meta(
    inp: int,
    out: int,
    start_ns: int,
    model: Model | None = None,
    cache_read: int = 0,
    cache_write: int = 0,
    system_prompt: str | None = None,
    tool_schema_digest: str | None = None,
) -> TurnMeta:
    return TurnMeta(
        total_input_tokens=inp,
        total_output_tokens=out,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        duration_ns=time.perf_counter_ns() - start_ns,
        model_id=model.id if model is not None else None,
        model_context_window=model.context_window if model is not None else None,
        system_prompt=system_prompt,
        tool_schema_digest=tool_schema_digest,
    )


async def _save_execution_checkpoint(
    request: ReactionRequest,
    meta: TurnMeta,
    *,
    pending_response: AssistantMessage | None = None,
    pending_tool_results: Sequence[ToolRecord] = (),
) -> None:
    if request.checkpoint is None:
        return
    await request.checkpoint(
        request.execution.checkpoint(
            meta,
            trigger_metadata=request.trigger_metadata,
            pending_response=pending_response,
            pending_tool_results=pending_tool_results,
        )
    )


async def _record_turn_payload(
    request: ReactionRequest,
    response: AssistantMessage,
    tool_records: list[ToolRecord],
    meta: TurnMeta,
) -> None:
    request.execution.set_result(response, tool_records)
    await _save_execution_checkpoint(request, meta)


async def _history_messages(
    *,
    turns: Sequence[Turn],
    policies: Sequence[ContextPolicy],
    trigger_renderers: dict[str, TriggerRenderer] | None,
    projection: ContextProjection | None,
    budget: ContextBudget,
    trajectory_store: TrajectoryStore | None,
    session_id: str,
    root_session_id: str | None,
    parent_session_id: str | None,
    signal: CancelSignal | None,
) -> list[AgentMessage]:
    if projection is None:
        return await build_context(
            turns,
            policies,
            trigger_renderers,
            signal=signal,
        )
    projection_input = await _projection_input(
        turns=turns,
        trajectory_store=trajectory_store,
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
    )
    messages = list(projection.project(projection_input, budget))
    return await apply_context_policies(
        messages,
        turns,
        policies,
        signal=signal,
    )


async def _projection_input(
    *,
    turns: Sequence[Turn],
    trajectory_store: TrajectoryStore | None,
    session_id: str,
    root_session_id: str | None,
    parent_session_id: str | None,
) -> ProjectionInput:
    if trajectory_store is None:
        raise RuntimeError(
            f"session {session_id}: a ContextProjection requires a trajectory store"
        )
    head = await asyncio.to_thread(
        trajectory_store.get_head,
        session_id,
        head_id=DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
        is_sidechain=False,
    )
    if head is None:
        if turns:
            raise RuntimeError(
                f"session {session_id}: node-chain ContextProjection requires "
                "an active trajectory head"
            )
        return ProjectionInput(
            turns=turns,
            session_id=session_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
        )
    leaf_node_id = head.node_id or head.logical_parent_id
    nodes: list[TrajectoryNode] = []
    if leaf_node_id is not None:
        nodes = await asyncio.to_thread(
            trajectory_store.load_chain,
            session_id,
            leaf_node_id,
            include_logical_parent=True,
        )
    return ProjectionInput(
        turns=turns,
        nodes=tuple(nodes),
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
        branch_id=head.branch_id,
        head_id=head.head_id,
        leaf_node_id=leaf_node_id,
        logical_parent_id=head.logical_parent_id,
    )


def _context_budget(model: Model) -> ContextBudget:
    return ContextBudget(
        max_input_tokens=model.context_window,
    )


async def react(
    request: ReactionRequest,
) -> ReactionResult:
    """ReAct loop within one turn.  Returns when a Stop action fires."""

    dependencies = request.dependencies
    execution = request.execution
    trajectory = dependencies.trajectory
    trigger = request.trigger
    trigger_metadata = request.trigger_metadata
    bus = dependencies.bus
    stream_fn = dependencies.stream_fn
    model = dependencies.model
    tools = dependencies.tools
    system = dependencies.system
    policies = dependencies.context_policies
    context_projection = request.context_projection
    trigger_renderers = dependencies.trigger_renderers
    interrupt = dependencies.interrupt
    shutdown = dependencies.shutdown
    parent_cancel_signal = dependencies.cancel_signal
    thinking = dependencies.thinking
    tool_executor = dependencies.tool_executor
    tool_orchestrator = dependencies.tool_orchestrator
    permission_policy = dependencies.permission_policy
    trajectory_store = dependencies.store
    session_id = dependencies.session_id
    root_session_id = dependencies.root_session_id
    parent_session_id = dependencies.parent_session_id
    permission_audience = dependencies.permission_audience
    interruption_policy = request.interruption_policy
    tool_calls_remaining = request.tool_calls_remaining
    tool_allowlist = dependencies.tool_allowlist

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    tool_calls_used = 0
    continuation_system_prompt = system
    start_ns = time.perf_counter_ns()

    def result(outcome: Outcome, meta: TurnMeta) -> ReactionResult:
        return ReactionResult(
            outcome=outcome,
            meta=meta,
            tool_calls_used=tool_calls_used,
            continuation_system_prompt=continuation_system_prompt,
        )

    trigger_messages = apply_trigger_metadata(
        render_trigger(trigger, trigger_renderers),
        trigger_metadata,
    )
    context_budget = _context_budget(model)
    turn_signal = CompositeCancelSignal(
        shutdown,
        parent_cancel_signal,
        interrupt,
    )
    try:
        history_messages = await _history_messages(
            turns=trajectory.turns,
            policies=policies,
            trigger_renderers=trigger_renderers,
            projection=context_projection,
            budget=context_budget,
            trajectory_store=trajectory_store,
            session_id=session_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            signal=turn_signal,
        )
    except ContextTransformCancelled:
        return result(
            Outcome(cause=_signal_aborted(turn_signal)),
            _meta(
                0,
                0,
                start_ns,
            ),
        )
    messages = list(history_messages) + list(trigger_messages)

    before_returns: list[object] = []
    if not isinstance(trigger, ContinueTrigger):
        before_event, before_returns = await bus.emit_reduced(
            BeforeRunEvent.CHANNEL,
            BeforeRunEvent(
                messages=tuple(messages),
                system=system,
            ),
            _reduce_before_run,
        )
        messages = list(before_event.messages)
        system = before_event.system
    continuation_system_prompt = system
    resolved_system_prompt = system
    resolved_tool_digest: str | None = None

    veto = _last_key(before_returns, "veto")
    if veto is not None:
        if not isinstance(veto, TerminationCause):
            raise TypeError("BeforeRunEvent veto must be a TerminationCause")
        return result(
            Outcome(cause=veto),
            _meta(
                0,
                0,
                start_ns,
                system_prompt=resolved_system_prompt,
                tool_schema_digest=resolved_tool_digest,
            ),
        )
    if turn_signal.is_set():
        return result(
            Outcome(cause=_signal_aborted(turn_signal)),
            _meta(
                total_input,
                total_output,
                start_ns,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
                system_prompt=resolved_system_prompt,
                tool_schema_digest=resolved_tool_digest,
            ),
        )

    allowed_tool_names = set(tool_allowlist) if tool_allowlist is not None else None
    send_tools = list(tools)
    if allowed_tool_names is not None:
        send_tools = [tool for tool in send_tools if tool.name in allowed_tool_names]
    if tool_calls_remaining is not None and tool_calls_used >= tool_calls_remaining:
        send_tools = []

    send_event, _send_returns = await bus.emit_reduced(
        BeforeSendEvent.CHANNEL,
        BeforeSendEvent(
            messages=tuple(messages),
            system=system,
            tools=tuple(send_tools),
            model=model,
            turn_index=execution.index,
        ),
        _reduce_before_send,
    )
    sent_before_handlers = len(messages)
    messages = list(send_event.messages)
    effective_system = send_event.system
    effective_model = send_event.model
    if effective_model is None:
        raise TypeError("BeforeSendEvent model cannot be None after reduction")
    effective_tools = list(send_event.tools)

    if effective_system is not None:
        resolved_system_prompt = effective_system

    # The reduce is append-only, so whatever sits past the incoming length is
    # exactly what handlers added. The turn carries it because the provider
    # read it, and a prefix the record cannot account for is not evidence.
    execution.set_request_appended(messages[sent_before_handlers:])

    messages = route_messages(messages, session_id=session_id)

    # Runtime constraints remain authoritative even if a transform adds tools.
    if allowed_tool_names is not None:
        effective_tools = [
            tool for tool in effective_tools if tool.name in allowed_tool_names
        ]
    if tool_calls_remaining is not None and tool_calls_used >= tool_calls_remaining:
        effective_tools = []
    resolved_tool_digest = _tool_schema_digest(effective_tools)
    tool_index = {t.name: t for t in effective_tools}

    stream_events: list[AssistantStreamEvent] = []
    llm_start_ns = time.perf_counter_ns()
    system_text = effective_system or ""
    await bus.emit(
        LlmRequestStartEvent.CHANNEL,
        LlmRequestStartEvent(
            turn_index=execution.index,
            turn_id=execution.id,
            model_id=effective_model.id,
            message_count=len(messages),
            tool_count=len(effective_tools),
            system_chars=len(system_text),
            system_text=system_text or None,
        ),
    )
    try:
        async for ev in stream_fn(
            messages=messages,
            model=effective_model,
            tools=effective_tools,
            system=effective_system,
            signal=turn_signal,
            thinking=thinking,
        ):
            stream_events.append(ev)
            await bus.emit(
                StreamDeltaEvent.CHANNEL,
                StreamDeltaEvent(
                    turn_index=execution.index,
                    delta=ev,
                ),
            )
    except BaseException as exc:
        await bus.emit(
            LlmRequestEndEvent.CHANNEL,
            LlmRequestEndEvent(
                turn_index=execution.index,
                turn_id=execution.id,
                chunk_count=len(stream_events),
                duration_ns=time.perf_counter_ns() - llm_start_ns,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        if turn_signal.is_set():
            response = _assemble_assistant_message(stream_events)
            reason = cancel_reason(turn_signal) or "unknown"
            interrupted_records = [
                _interrupted_tool_record(
                    call,
                    reason=reason,
                    policy=interruption_policy,
                )
                for call in _extract_tool_calls(response)
            ]
            checkpoint_meta = _meta(
                total_input,
                total_output,
                start_ns,
                effective_model,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
            )
            await _record_turn_payload(
                request,
                response,
                interrupted_records,
                checkpoint_meta,
            )
            return result(
                Outcome(cause=_signal_aborted(turn_signal)),
                checkpoint_meta,
            )
        if isinstance(exc, asyncio.CancelledError) or not isinstance(
            exc,
            Exception,
        ):
            raise
        if stream_events:
            partial_response = _assemble_assistant_message(stream_events)
            checkpoint_meta = _meta(
                total_input,
                total_output,
                start_ns,
                effective_model,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
            )
            await _record_turn_payload(
                request,
                partial_response,
                [],
                checkpoint_meta,
            )
        return result(
            Outcome(
                cause=ProviderRequestFailed(
                    error_type=type(exc).__name__,
                    detail=str(exc),
                    partial_event_count=len(stream_events),
                )
            ),
            _meta(
                total_input,
                total_output,
                start_ns,
                effective_model,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
                system_prompt=resolved_system_prompt,
                tool_schema_digest=resolved_tool_digest,
            ),
        )
    await bus.emit(
        LlmRequestEndEvent.CHANNEL,
        LlmRequestEndEvent(
            turn_index=execution.index,
            turn_id=execution.id,
            chunk_count=len(stream_events),
            duration_ns=time.perf_counter_ns() - llm_start_ns,
        ),
    )

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
    checkpoint_meta = _meta(
        total_input,
        total_output,
        start_ns,
        effective_model,
        cache_read=total_cache_read,
        cache_write=total_cache_write,
    )
    await _save_execution_checkpoint(
        request,
        checkpoint_meta,
        pending_response=response,
    )

    if turn_signal.is_set() or isinstance(response.termination, Aborted):
        _append_interrupted_tool_records(
            calls=tool_calls,
            records=tool_records,
            reason=cancel_reason(turn_signal) or "unknown",
            policy=interruption_policy,
        )
        await _record_turn_payload(
            request,
            response,
            tool_records,
            checkpoint_meta,
        )
        return result(
            Outcome(cause=_signal_aborted(turn_signal)),
            checkpoint_meta,
        )

    if tool_calls:
        if tool_calls_remaining is not None and len(tool_calls) > max(
            0, tool_calls_remaining - tool_calls_used
        ):
            for call in tool_calls:
                tool_records.append(
                    ToolRecord(
                        call=call,
                        result=ToolResultBlock(
                            type="tool_result",
                            tool_call_id=call.id,
                            content=[
                                TextContent(
                                    type="text",
                                    text=(
                                        "Tool call skipped: max_tool_calls exhausted"
                                    ),
                                )
                            ],
                            is_error=True,
                        ),
                    )
                )
                await _save_execution_checkpoint(
                    request,
                    checkpoint_meta,
                    pending_response=response,
                    pending_tool_results=tool_records,
                )
            await _record_turn_payload(
                request,
                response,
                tool_records,
                checkpoint_meta,
            )
            return result(
                Outcome(cause=BudgetExhausted(detail="max_tool_calls exhausted")),
                checkpoint_meta,
            )

        outcomes_by_index: dict[int, ToolOutcome] = {}
        records_by_index: dict[int, ToolRecord] = {}
        work_items: list[ToolWorkItem] = []
        tool_args_by_index: dict[int, Mapping[str, object]] = {}

        async def materialize_tool_outcome(
            index: int,
            outcome: ToolOutcome,
            *,
            duration_ms: int | None = None,
        ) -> None:
            nonlocal tool_calls_used
            if index in outcomes_by_index:
                raise RuntimeError(
                    f"tool orchestrator produced duplicate result index {index}"
                )
            record, final_outcome = await _finalize_tool_outcome(
                bus=bus,
                call=tool_calls[index],
                outcome=outcome,
                args=tool_args_by_index.get(index),
                duration_ms=duration_ms,
            )
            outcomes_by_index[index] = final_outcome
            records_by_index[index] = record
            tool_calls_used += 1
            materialized_records = [
                records_by_index[record_index]
                for record_index in sorted(records_by_index)
            ]
            await _save_execution_checkpoint(
                request,
                checkpoint_meta,
                pending_response=response,
                pending_tool_results=materialized_records,
            )

        for index, tc in enumerate(tool_calls):
            tool_args_by_index[index] = dict(tc.arguments)
            if turn_signal.is_set():
                tool_records = _cancelled_tool_records(
                    calls=tool_calls,
                    outcomes=outcomes_by_index,
                    reason=cancel_reason(turn_signal) or "unknown",
                    policy=interruption_policy,
                )
                await _record_turn_payload(
                    request,
                    response,
                    tool_records,
                    checkpoint_meta,
                )
                return result(
                    Outcome(cause=_signal_aborted(turn_signal)),
                    checkpoint_meta,
                )

            tc_returns = await bus.emit_decision(
                ToolCallEvent.CHANNEL,
                ToolCallEvent(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    args=dict(tc.arguments),
                ),
            )
            for returned in tc_returns:
                if isinstance(returned, Mapping):
                    unknown = set(returned) - {"block", "reason"}
                    if unknown:
                        # An atom written against an older ABI may still try to
                        # rewrite arguments. Ignoring it would run the tool with
                        # the arguments the handler believed it had replaced,
                        # so refuse the turn instead of failing open.
                        raise TypeError(
                            "ToolCallEvent handlers return only 'block' and "
                            f"'reason'; got {sorted(unknown)}"
                        )
            blocked = _last_key(tc_returns, "block")

            if blocked:
                returned_reason = _last_key(tc_returns, "reason")
                if returned_reason is not None and not isinstance(
                    returned_reason,
                    str,
                ):
                    raise TypeError("ToolCallEvent reason must be a string")
                reason = returned_reason or "blocked"
                await materialize_tool_outcome(
                    index,
                    ToolContinue(
                        result=ToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"blocked: {reason}",
                                )
                            ],
                            is_error=True,
                        )
                    ),
                )
                continue

            args = dict(tc.arguments)
            if allowed_tool_names is not None and tc.name not in allowed_tool_names:
                await materialize_tool_outcome(
                    index,
                    ToolContinue(
                        result=ToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"blocked by tool_allowlist: {tc.name}",
                                )
                            ],
                            is_error=True,
                        )
                    ),
                )
                continue

            tool = tool_index.get(tc.name)
            if tool is None:
                await materialize_tool_outcome(
                    index,
                    ToolContinue(
                        result=ToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"unknown tool: {tc.name}",
                                )
                            ],
                            is_error=True,
                        )
                    ),
                )
                continue

            permission_outcome = await _permission_outcome(
                policy=permission_policy,
                request=_permission_request(
                    tc=tc,
                    args=args,
                    session_id=session_id,
                    execution=execution,
                    audience=permission_audience,
                ),
                signal=turn_signal,
            )
            if turn_signal.is_set():
                tool_records = _cancelled_tool_records(
                    calls=tool_calls,
                    outcomes=outcomes_by_index,
                    reason=cancel_reason(turn_signal) or "unknown",
                    policy=interruption_policy,
                )
                await _record_turn_payload(
                    request,
                    response,
                    tool_records,
                    checkpoint_meta,
                )
                return result(
                    Outcome(cause=_signal_aborted(turn_signal)),
                    checkpoint_meta,
                )
            if permission_outcome is not None:
                await materialize_tool_outcome(index, permission_outcome)
                continue

            work_items.append(
                ToolWorkItem(
                    index=index,
                    call=tc,
                    tool=tool,
                    args=args,
                    requirements=tool_execution_requirements(tool),
                )
            )

        if work_items:
            requested_items = {item.index: item for item in work_items}
            async for orch_result in tool_orchestrator.stream_batch(
                ToolOrchestrationRequest(
                    items=tuple(work_items),
                    session_id=session_id,
                    turn_id=execution.id,
                    turn_index=execution.index,
                ),
                signal=turn_signal,
                executor=tool_executor,
            ):
                expected_item = requested_items.get(orch_result.item.index)
                if expected_item is None or orch_result.item is not expected_item:
                    raise RuntimeError(
                        "tool orchestrator returned a result that does not "
                        "reference its original request item"
                    )
                if turn_signal.is_set() and orch_result.status in {
                    "cancelled",
                    "skipped",
                }:
                    continue
                if orch_result.status == "completed" and orch_result.output is not None:
                    outcome = _normalize_tool_output(orch_result.output)
                else:
                    outcome = await _orchestration_failure_outcome(
                        result=orch_result,
                        bus=bus,
                    )
                await materialize_tool_outcome(
                    orch_result.item.index,
                    outcome,
                    duration_ms=orch_result.duration_ms,
                )

        if turn_signal.is_set() and len(outcomes_by_index) < len(tool_calls):
            tool_records = _cancelled_tool_records(
                calls=tool_calls,
                outcomes=outcomes_by_index,
                reason=cancel_reason(turn_signal) or "unknown",
                policy=interruption_policy,
            )
            await _record_turn_payload(
                request,
                response,
                tool_records,
                checkpoint_meta,
            )
            return result(
                Outcome(cause=_signal_aborted(turn_signal)),
                checkpoint_meta,
            )

        missing_indexes = set(range(len(tool_calls))) - outcomes_by_index.keys()
        if missing_indexes:
            raise RuntimeError(
                "tool orchestrator omitted terminal results for indexes "
                f"{sorted(missing_indexes)}"
            )

        tool_records = [records_by_index[index] for index in range(len(tool_calls))]
        paired_outcomes = [
            (tool_calls[index].name, outcomes_by_index[index])
            for index in range(len(tool_calls))
        ]
        result_blocks = [record.result for record in tool_records]

        if result_blocks:
            messages.append(
                ToolResultMessage(
                    role="tool_result",
                    content=result_blocks,
                    timestamp=time.time(),
                )
            )

    if turn_signal.is_set():
        checkpoint_meta = _meta(
            total_input,
            total_output,
            start_ns,
            effective_model,
            cache_read=total_cache_read,
            cache_write=total_cache_write,
        )
        await _record_turn_payload(
            request,
            response,
            tool_records,
            checkpoint_meta,
        )
        return result(
            Outcome(cause=_signal_aborted(turn_signal)),
            checkpoint_meta,
        )

    await _record_turn_payload(
        request,
        response,
        tool_records,
        _meta(
            total_input,
            total_output,
            start_ns,
            effective_model,
            cache_read=total_cache_read,
            cache_write=total_cache_write,
        ),
    )

    terminal_from_trigger = _trigger_carries_terminal(trigger)
    default = _default_action(response, paired_outcomes)
    if terminal_from_trigger is not None and isinstance(default, Step):
        default = Stop(cause=terminal_from_trigger)

    decide_returns = await bus.emit_decision(
        DecideEvent.CHANNEL,
        DecideEvent(
            observation=TurnObservation(
                turn_index=execution.index,
                assistant_message=response,
                tool_outcomes=tuple(paired_outcomes),
                default_action=default,
            ),
        ),
    )
    action = _resolve_action(default, decide_returns)

    if isinstance(action, Stop):
        cause = action.cause if action.cause is not None else ModelEndTurn()
        return result(
            Outcome(cause=cause),
            _meta(
                total_input,
                total_output,
                start_ns,
                effective_model,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
                system_prompt=resolved_system_prompt,
                tool_schema_digest=resolved_tool_digest,
            ),
        )

    if isinstance(action, Inject):
        execution.add_injected(list(action.messages))
        await _save_execution_checkpoint(
            request,
            _meta(
                total_input,
                total_output,
                start_ns,
                effective_model,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
            ),
        )

    return result(
        Outcome(cause=PromptRunContinued()),
        _meta(
            total_input,
            total_output,
            start_ns,
            effective_model,
            cache_read=total_cache_read,
            cache_write=total_cache_write,
            system_prompt=resolved_system_prompt,
            tool_schema_digest=resolved_tool_digest,
        ),
    )


__all__ = [
    "ReactionDependencies",
    "ReactionRequest",
    "ReactionResult",
    "react",
    "record_interruption_message",
]
