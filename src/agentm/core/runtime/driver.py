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
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar, cast

from loguru import logger

from agentm.core.abi.cancel import (
    CancelReason,
    CancelSignal,
    EventCancelSource,
    ResettableCancelSource,
    cancel_reason,
)
from agentm.core.abi.compaction import (
    ContextBudget,
    ContextProjection,
    ProjectionInput,
)
from agentm.core.abi.lifecycle import EffectScope, EffectTxn
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    InterruptionMessagePolicy,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)
from agentm.core.abi.permission import (
    PermissionAudience,
    PermissionPolicy,
    PermissionRequest,
    permission_denial_result,
)
from agentm.core.abi.provider import (
    ProviderPromptCacheAdapter,
    ProviderPromptCacheRequest,
)
from agentm.core.abi.resource import (
    ResourceMutation,
    ResourceTxn,
    ResourceTxnContext,
    ResourceWriter,
    TransactionalResourceWriter,
)
from agentm.core.abi.roles import (
    CONTEXT_PROJECTION_SERVICE,
    INTERRUPTION_MESSAGE_POLICY_SERVICE,
    PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE,
    RESOURCE_TXN_SERVICE,
)
from agentm.core.abi.services import ServiceRegistry
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
from agentm.core.abi.tool_executor import (
    ToolExecutionRequirements,
    ToolExecutor,
    tool_execution_requirements,
)
from agentm.core.abi.tool_orchestration import (
    ToolOrchestrationRequest,
    ToolOrchestrationResult,
    ToolOrchestrator,
    ToolWorkItem,
)
from agentm.core.abi.bus import EventBus
from agentm.core.abi.context import (
    ContextPolicy,
    ContextTransformCancelled,
    apply_trigger_metadata,
    apply_context_policies,
    build_context,
    render_trigger,
    route_messages,
)
from agentm.core.abi.events import (
    BeforeRunEvent,
    BeforeSendEvent,
    ContextEvent,
    DecideEvent,
    DiagnosticEvent,
    Inject,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    LoopAction,
    RunEndEvent,
    Step,
    Stop,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    TurnBeginEvent,
    TurnCommittedEvent,
    TurnObservation,
)
from agentm.core.abi.store import (
    TrajectoryNodeQuery,
    TrajectoryNodeStore,
    TrajectoryStore,
)
from agentm.core.abi.termination import (
    Aborted,
    BudgetExhausted,
    MaxTurnsExhausted,
    ModelEndTurn,
    PauseTurn,
    ProviderRequestFailed,
    ProviderTruncated,
    SignalAborted,
    TerminationCause,
    ToolTerminated,
)
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    Outcome,
    PromptCacheState,
    ToolRecord,
    TrajectoryHead,
    TrajectoryHeadAdvance,
    TrajectoryNode,
    TrajectoryProjectionStatus,
    Turn,
    TurnMeta,
)
from agentm.core.abi.trigger import (
    BackgroundCompletion,
    SubagentResult,
    Trigger,
    TriggerMetadata,
    TriggerRenderer,
)
from agentm.core.runtime.execution import Execution
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.tool_orchestration import default_tool_orchestrator
from agentm.core.lib.trajectory_nodes import turn_to_nodes, turns_to_nodes
from agentm.core.runtime.trigger_queue import (
    QueueClosed,
    TriggerQueue,
    TriggerTerminated,
)

ThinkingLevel = Literal["off", "low", "medium", "high"]


# --- Helpers ----------------------------------------------------------------

_INTERRUPTED_TOOL_TEXT = "Tool execution interrupted"
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _NodeAppendPosition:
    start_seq: int
    parent_node_id: str | None
    logical_parent_id: str | None
    head_id: str
    branch_id: str


class _TurnCancelSignal:
    """Composes turn interrupt, session shutdown, and an optional parent signal."""

    def __init__(
        self,
        *,
        interrupt: ResettableCancelSource,
        shutdown: ResettableCancelSource,
        parent: CancelSignal | None = None,
    ) -> None:
        self._interrupt = interrupt
        self._shutdown = shutdown
        self._parent = parent

    def is_set(self) -> bool:
        return (
            self._interrupt.is_set()
            or self._shutdown.is_set()
            or (self._parent is not None and self._parent.is_set())
        )

    @property
    def reason(self) -> CancelReason | str | None:
        if self._shutdown.is_set():
            return cancel_reason(self._shutdown) or "shutdown"
        if self._parent is not None and self._parent.is_set():
            return cancel_reason(self._parent) or "unknown"
        if self._interrupt.is_set():
            return cancel_reason(self._interrupt) or "user_cancel"
        return None

    async def wait(self) -> None:
        if self.is_set():
            return
        waiters: list[asyncio.Task[Any]] = [
            asyncio.create_task(self._interrupt.wait()),
            asyncio.create_task(self._shutdown.wait()),
        ]
        if self._parent is not None:
            waiters.append(asyncio.create_task(self._parent.wait()))
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)


def _signal_aborted(signal: CancelSignal | None) -> SignalAborted:
    return SignalAborted(reason=cancel_reason(signal) or "")


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


def _record_interruption_message(
    execution: Execution,
    outcome: Outcome,
    policy: InterruptionMessagePolicy | None,
) -> None:
    """Append a policy message after the interrupted request, when required."""

    cause = outcome.cause
    if not isinstance(cause, SignalAborted) or policy is None:
        return
    reason = cause.reason or "unknown"
    for_tool_use = any(
        _extract_tool_calls(round_.response)
        for round_ in execution.rounds
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
        return ToolTerminated(tool_name=f"background:{trigger.task_id}", reason="terminal")
    if isinstance(trigger, SubagentResult) and trigger.terminal:
        return ToolTerminated(tool_name=f"subagent:{trigger.child_session_id}", reason="terminal")
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
        return ToolContinue(result=ToolResult(
            content=[TextContent(type="text", text=f"tool cancelled: {reason}")],
            is_error=True,
        ))
    exc = result.error
    reason = str(exc) if exc is not None else result.status
    err_returns = await bus.emit(
        ToolErrorEvent.CHANNEL,
        ToolErrorEvent(
            kind="execution_failed",
            tool_name=result.item.call.name,
            reason=reason,
            exception=exc,
        ),
    )
    err_text = _last_key(err_returns, "text") or (
        f"tool error: {result.item.call.name}: {reason}"
    )
    return ToolContinue(result=ToolResult(
        content=[TextContent(type="text", text=err_text)],
        is_error=True,
    ))


async def _emit_lifecycle_diagnostic(
    bus: EventBus,
    *,
    boundary: str,
    action: str,
    exc: BaseException,
) -> None:
    try:
        await bus.emit(DiagnosticEvent.CHANNEL, DiagnosticEvent(
            level="error",
            source="lifecycle",
            message=f"{boundary} {action} failed: {type(exc).__name__}: {exc}",
        ))
    except Exception:
        logger.debug("diagnostic emit failed after lifecycle {}; non-fatal", action)


async def _begin_effect_turn(
    effect_scope: EffectScope | None,
    *,
    bus: EventBus,
    session_id: str,
    turn_id: str,
    turn_index: int,
) -> EffectTxn | None:
    if effect_scope is None:
        return None
    try:
        return await effect_scope.begin_turn(
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
        )
    except Exception as exc:
        logger.exception("effect_scope.begin_turn failed")
        await _emit_lifecycle_diagnostic(
            bus,
            boundary="effect scope",
            action="begin_turn",
            exc=exc,
        )
        raise


async def _commit_effect_turn(
    effect_scope: EffectScope | None,
    txn: EffectTxn | None,
    turn: Turn,
    *,
    bus: EventBus,
) -> None:
    if effect_scope is None or txn is None:
        return
    try:
        await effect_scope.commit_turn(txn, turn)
    except Exception as exc:
        logger.exception("effect_scope.commit_turn failed")
        await _emit_lifecycle_diagnostic(
            bus,
            boundary="effect scope",
            action="commit_turn",
            exc=exc,
        )
        raise


async def _prepare_effect_turn(
    effect_scope: EffectScope | None,
    txn: EffectTxn | None,
    turn: Turn,
    *,
    bus: EventBus,
) -> None:
    if effect_scope is None or txn is None:
        return
    try:
        await effect_scope.prepare_turn(txn, turn)
    except Exception as exc:
        logger.exception("effect_scope.prepare_turn failed")
        await _emit_lifecycle_diagnostic(
            bus,
            boundary="effect scope",
            action="prepare_turn",
            exc=exc,
        )
        raise


async def _append_turn(
    store: TrajectoryStore,
    session_id: str,
    turn: Turn,
) -> bool:
    """Append durably, returning whether the caller was cancelled meanwhile."""
    task = asyncio.create_task(asyncio.to_thread(store.append, session_id, turn))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    return cancelled


async def _upsert_turn(
    store: TrajectoryStore,
    session_id: str,
    turn: Turn,
) -> bool:
    """Upsert durably, returning whether the caller was cancelled meanwhile."""
    task = asyncio.create_task(asyncio.to_thread(store.upsert_turn, session_id, turn))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    return cancelled


async def _await_known_outcome(awaitable: Awaitable[_T]) -> tuple[_T, bool]:
    """Wait for a state transition without letting caller cancellation race it."""

    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


async def _abandon_effect_turn(
    effect_scope: EffectScope | None,
    txn: EffectTxn | None,
    *,
    bus: EventBus,
) -> None:
    if effect_scope is None or txn is None:
        return
    try:
        await effect_scope.abandon_turn(txn)
    except Exception as exc:
        logger.exception("effect_scope.abandon_turn failed")
        await _emit_lifecycle_diagnostic(
            bus,
            boundary="effect scope",
            action="abandon_turn",
            exc=exc,
        )
        raise


async def _begin_resource_txn(
    writer: ResourceWriter | None,
    services: ServiceRegistry | None,
    *,
    session_id: str,
    turn_id: str,
    turn_index: int,
) -> ResourceTxn | None:
    if writer is None or not isinstance(writer, TransactionalResourceWriter):
        return None
    txn = await writer.begin_txn(
        ResourceTxnContext(
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
            rationale="agent turn resource mutations",
        )
    )
    if services is not None:
        services.register(
            RESOURCE_TXN_SERVICE,
            txn,
            ResourceTxn,
            scope="session",
        )
    return txn


async def _prepare_resource_txn(
    txn: ResourceTxn | None,
) -> tuple[ResourceMutation, ...]:
    if txn is None:
        return ()
    return tuple(await txn.prepare())


async def _apply_resource_txn(txn: ResourceTxn | None) -> None:
    if txn is not None:
        await txn.apply()


async def _commit_resource_txn(txn: ResourceTxn | None) -> None:
    if txn is not None:
        await txn.commit()


async def _abandon_resource_txn(
    txn: ResourceTxn | None,
    *,
    bus: EventBus,
) -> None:
    if txn is None:
        return
    try:
        await txn.abandon()
    except Exception as exc:
        logger.exception("resource_txn.abandon failed")
        await _emit_lifecycle_diagnostic(
            bus,
            boundary="resource transaction",
            action="abandon",
            exc=exc,
        )
        raise


async def _rollback_unpublished_turn(
    *,
    resource_txn: ResourceTxn | None,
    abandon_resource: bool,
    effect_scope: EffectScope | None,
    effect_txn: EffectTxn | None,
    bus: EventBus,
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []

    # EffectScope is the outer world transaction: it snapshots the state after
    # ResourceTxn.apply(). Restore that world first, then let the resource
    # participant validate/remove its own journal. Running both concurrently
    # races over the same files for local and sandbox implementations.
    if effect_scope is not None and effect_txn is not None:
        try:
            await _await_known_outcome(
                _abandon_effect_turn(effect_scope, effect_txn, bus=bus)
            )
        except BaseException as exc:
            errors.append(exc)
    if abandon_resource:
        try:
            await _await_known_outcome(_abandon_resource_txn(resource_txn, bus=bus))
        except BaseException as exc:
            errors.append(exc)
    return tuple(errors)


async def _node_append_position(
    store: TrajectoryNodeStore,
    session_id: str,
    *,
    committed_turns: Sequence[Turn],
    root_session_id: str | None,
    parent_session_id: str | None,
    trigger_renderers: dict[str, TriggerRenderer] | None,
) -> _NodeAppendPosition:
    """Load the explicit append head, repairing stale projection if needed."""

    latest, head = await asyncio.gather(
        asyncio.to_thread(
            store.query_nodes,
            TrajectoryNodeQuery(session_id=session_id, sort="desc", limit=1),
        ),
        asyncio.to_thread(
            store.get_head,
            session_id,
            head_id=DEFAULT_TRAJECTORY_HEAD_ID,
            branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
            is_sidechain=False,
        ),
    )
    start_seq = latest[0].seq + 1 if latest else 0
    if head is not None and head.status == "active":
        return _NodeAppendPosition(
            start_seq=start_seq,
            parent_node_id=head.node_id,
            logical_parent_id=head.logical_parent_id if head.node_id is None else None,
            head_id=head.head_id,
            branch_id=head.branch_id,
        )

    if not latest:
        return _NodeAppendPosition(
            start_seq=0,
            parent_node_id=None,
            logical_parent_id=None,
            head_id=DEFAULT_TRAJECTORY_HEAD_ID,
            branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
        )

    logger.warning(
        "trajectory projection head missing for session {}; rebuilding projection",
        session_id,
    )
    await _rebuild_node_projection(
        store=store,
        session_id=session_id,
        turns=committed_turns,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
        trigger_renderers=trigger_renderers,
    )
    latest, head = await asyncio.gather(
        asyncio.to_thread(
            store.query_nodes,
            TrajectoryNodeQuery(session_id=session_id, sort="desc", limit=1),
        ),
        asyncio.to_thread(
            store.get_head,
            session_id,
            head_id=DEFAULT_TRAJECTORY_HEAD_ID,
            branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
            is_sidechain=False,
        ),
    )
    if not latest:
        return _NodeAppendPosition(
            start_seq=0,
            parent_node_id=None,
            logical_parent_id=None,
            head_id=DEFAULT_TRAJECTORY_HEAD_ID,
            branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
        )
    if head is None or head.status != "active":
        raise RuntimeError(
            f"trajectory projection for session {session_id} has nodes but no "
            "active append head after rebuild"
        )
    return _NodeAppendPosition(
        start_seq=latest[0].seq + 1,
        parent_node_id=head.node_id,
        logical_parent_id=head.logical_parent_id if head.node_id is None else None,
        head_id=head.head_id,
        branch_id=head.branch_id,
    )


def _projection_status_for_nodes(
    session_id: str,
    nodes: Sequence[TrajectoryNode],
) -> TrajectoryProjectionStatus:
    last = nodes[-1] if nodes else None
    return TrajectoryProjectionStatus(
        session_id=session_id,
        state="current",
        high_water_turn_id=last.turn_id if last is not None else None,
        high_water_turn_index=last.turn_index if last is not None else None,
        node_count=len(nodes),
        updated_at=time.time(),
    )


def _projection_head_for_nodes(
    *,
    session_id: str,
    root_session_id: str | None,
    parent_session_id: str | None,
    nodes: Sequence[TrajectoryNode],
    head_id: str = DEFAULT_TRAJECTORY_HEAD_ID,
    branch_id: str = DEFAULT_TRAJECTORY_BRANCH_ID,
) -> TrajectoryHead:
    last = nodes[-1] if nodes else None
    return TrajectoryHead(
        session_id=session_id,
        head_id=head_id,
        branch_id=branch_id,
        node_id=last.id if last is not None else None,
        seq=last.seq if last is not None else None,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
        status="active",
        updated_at=time.time(),
    )


async def _rebuild_node_projection(
    *,
    store: TrajectoryNodeStore,
    session_id: str,
    turns: Sequence[Turn],
    root_session_id: str | None,
    parent_session_id: str | None,
    trigger_renderers: dict[str, TriggerRenderer] | None,
) -> None:
    nodes = turns_to_nodes(
        turns,
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
        branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
        head_id=DEFAULT_TRAJECTORY_HEAD_ID,
        renderers=trigger_renderers,
    )
    head = _projection_head_for_nodes(
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
        nodes=nodes,
    )
    await asyncio.to_thread(
        store.replace_session_projection,
        session_id,
        nodes,
        heads=(head,),
        status=_projection_status_for_nodes(session_id, nodes),
    )


async def _append_or_rebuild_nodes(
    *,
    store: TrajectoryNodeStore,
    session_id: str,
    nodes: Sequence[TrajectoryNode],
    advance_head: TrajectoryHeadAdvance | None,
    committed_turns: Sequence[Turn],
    root_session_id: str | None,
    parent_session_id: str | None,
    trigger_renderers: dict[str, TriggerRenderer] | None,
) -> None:
    """Update or rebuild the projection; surface a failed recovery attempt."""

    try:
        await asyncio.to_thread(
            store.append_nodes,
            session_id,
            nodes,
            advance_head=advance_head,
        )
    except Exception as append_error:
        logger.exception(
            "trajectory node projection append failed; rebuilding session {}",
            session_id,
        )
        try:
            await _rebuild_node_projection(
                store=store,
                session_id=session_id,
                turns=committed_turns,
                root_session_id=root_session_id,
                parent_session_id=parent_session_id,
                trigger_renderers=trigger_renderers,
            )
        except Exception as rebuild_error:
            raise ExceptionGroup(
                f"trajectory projection append and rebuild failed for {session_id}",
                [append_error, rebuild_error],
            ) from rebuild_error


def _clear_resource_txn(services: ServiceRegistry | None) -> None:
    if services is not None:
        services.unregister(RESOURCE_TXN_SERVICE)


def _meta(inp: int, out: int, start_ns: int, model: Model | None = None,
          cache_read: int = 0, cache_write: int = 0) -> TurnMeta:
    return TurnMeta(
        total_input_tokens=inp, total_output_tokens=out,
        cache_read_tokens=cache_read, cache_write_tokens=cache_write,
        duration_ns=time.perf_counter_ns() - start_ns,
        model_id=model.id if model is not None else None,
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
                    deterministic=tr.result.deterministic,
                    extras=tr.result.extras,
                )
                for tr in rnd.tool_results
            ]
            messages.append(ToolResultMessage(
                role="tool_result", content=result_blocks, timestamp=0.0,
            ))
        if i in inject_by_round:
            messages.extend(inject_by_round[i])
    return messages


async def _history_messages(
    *,
    turns: Sequence[Turn],
    policies: list[ContextPolicy],
    trigger_renderers: dict[str, TriggerRenderer] | None,
    projection: ContextProjection | None,
    budget: ContextBudget,
    trajectory_node_store: TrajectoryNodeStore | None,
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
        projection=projection,
        trajectory_node_store=trajectory_node_store,
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


def _message_cache_key(messages: Sequence[AgentMessage]) -> str | None:
    for message in reversed(messages):
        cache_key = message.meta.tags.get("cache_key")
        if isinstance(cache_key, str) and cache_key:
            return cache_key
    return None


async def _apply_provider_prompt_cache(
    *,
    messages: Sequence[AgentMessage],
    model: Model,
    adapter: ProviderPromptCacheAdapter | None,
    store: TrajectoryNodeStore | None,
    session_id: str,
) -> list[AgentMessage]:
    if adapter is None:
        return list(messages)
    cache_key = _message_cache_key(messages)
    if cache_key is None:
        return list(messages)

    state = (
        await asyncio.to_thread(store.load_prompt_cache_state, session_id, cache_key)
        if store is not None
        else None
    )
    if state is None:
        state = PromptCacheState(cache_key=cache_key, provider=model.provider)
    result = adapter.apply_prompt_cache(
        ProviderPromptCacheRequest(
            messages=messages,
            model=model,
            state=state,
            metadata={"session_id": session_id},
        )
    )
    if result.state.cache_key != cache_key:
        raise ValueError(
            "provider prompt-cache adapter cannot change the cache identity"
        )
    if store is not None:
        await asyncio.to_thread(
            store.save_prompt_cache_state,
            session_id,
            result.state,
        )
    return list(result.messages)


async def _projection_input(
    *,
    turns: Sequence[Turn],
    projection: ContextProjection,
    trajectory_node_store: TrajectoryNodeStore | None,
    session_id: str,
    root_session_id: str | None,
    parent_session_id: str | None,
) -> ProjectionInput:
    if projection.source == "turns":
        return ProjectionInput(
            turns=turns,
            session_id=session_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
        )
    if projection.source != "node_chain":
        raise ValueError(f"unsupported context projection source {projection.source!r}")
    if trajectory_node_store is None:
        raise RuntimeError(
            "node-chain ContextProjection requires a trajectory_node_store"
        )
    head = await asyncio.to_thread(
        trajectory_node_store.get_head,
        session_id,
        head_id=DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
        is_sidechain=False,
    )
    if head is None:
        if turns:
            raise RuntimeError(
                "node-chain ContextProjection requires an active trajectory head"
            )
        return ProjectionInput(
            turns=turns,
            source="node_chain",
            session_id=session_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
        )
    leaf_node_id = head.node_id or head.logical_parent_id
    nodes: list[TrajectoryNode] = []
    if leaf_node_id is not None:
        nodes = await asyncio.to_thread(
            trajectory_node_store.load_chain,
            session_id,
            leaf_node_id,
            include_logical_parent=True,
        )
    return ProjectionInput(
        turns=turns,
        nodes=tuple(nodes),
        source="node_chain",
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
        reserved_output_tokens=model.max_output_tokens,
    )


# --- Main driver ------------------------------------------------------------


@dataclass(slots=True)
class DriverConfig:
    """Stable dependencies and policy inputs for one session driver."""

    trajectory: Trajectory
    triggers: TriggerQueue
    bus: EventBus
    stream_fn: StreamFn
    model: Model
    tools: list[Tool] = field(default_factory=list)
    store: TrajectoryStore | None = None
    session_id: str = ""
    root_session_id: str | None = None
    parent_session_id: str | None = None
    permission_audience: PermissionAudience = "user"
    system: str | None = None
    context_policies: list[ContextPolicy] | None = None
    prompt_cache_adapter: ProviderPromptCacheAdapter | None = None
    trigger_renderers: dict[str, TriggerRenderer] | None = None
    interrupt: ResettableCancelSource | None = None
    shutdown: ResettableCancelSource | None = None
    cancel_signal: CancelSignal | None = None
    effect_scope: EffectScope | None = None
    resource_writer: ResourceWriter | None = None
    services: ServiceRegistry | None = None
    tool_executor: ToolExecutor | None = None
    tool_orchestrator: ToolOrchestrator | None = None
    permission_policy: PermissionPolicy | None = None
    trajectory_node_store: TrajectoryNodeStore | None = None
    max_turns: int | None = None
    thinking: ThinkingLevel = "off"
    max_tool_calls: int | None = None
    tool_allowlist: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _ReactionRequest:
    execution: Execution
    trigger: Trigger
    trigger_metadata: TriggerMetadata
    config: DriverConfig
    context_projection: ContextProjection | None
    prompt_cache_adapter: ProviderPromptCacheAdapter | None
    interruption_policy: InterruptionMessagePolicy | None
    tool_calls_remaining: int | None


async def drive(config: DriverConfig) -> None:
    """Persistent driver loop.

    Processes triggers one at a time.  Each trigger becomes one Turn
    (potentially with multiple ReAct rounds).  Exits when:
    - shutdown is set
    - trigger queue is closed (QueueClosed)
    - max_turns committed turns reached
    - a session-terminal cause fires (ToolTerminated, BudgetExhausted)
    """

    trajectory = config.trajectory
    triggers = config.triggers
    bus = config.bus
    _interrupt = config.interrupt or EventCancelSource()
    _shutdown = config.shutdown or EventCancelSource()
    policies = config.context_policies or []
    context_projection = (
        config.services.get(
            CONTEXT_PROJECTION_SERVICE,
            cast(type[ContextProjection], ContextProjection),
        )
        if config.services is not None
        else None
    )
    prompt_cache_adapter = config.prompt_cache_adapter
    if prompt_cache_adapter is None and config.services is not None:
        prompt_cache_adapter = config.services.get(
            PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE,
            cast(type[ProviderPromptCacheAdapter], ProviderPromptCacheAdapter),
        )
    interruption_policy = (
        config.services.get(
            INTERRUPTION_MESSAGE_POLICY_SERVICE,
            cast(type[InterruptionMessagePolicy], InterruptionMessagePolicy),
        )
        if config.services is not None
        else None
    )
    turns_run = 0
    tool_calls_run = 0

    while True:
        if _shutdown.is_set():
            triggers.terminate(TriggerTerminated("session shut down"))
            await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
            return

        if config.max_turns is not None and turns_run >= config.max_turns:
            cause = MaxTurnsExhausted()
            triggers.terminate(TriggerTerminated(cause))
            await bus.emit(
                RunEndEvent.CHANNEL,
                RunEndEvent(outcome=Outcome(cause=cause)),
            )
            return

        try:
            envelope = await triggers.wait_envelope()
            trigger = envelope.trigger
        except QueueClosed:
            triggers.terminate(TriggerTerminated("trigger queue closed"))
            await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
            return

        _interrupt.clear()

        effect_txn: EffectTxn | None = None
        resource_txn: ResourceTxn | None = None
        resource_txn_committed = False
        durable_turn_appended = False
        turn_published = False
        try:
            execution = trajectory.begin(trigger)
            await bus.emit(TurnBeginEvent.CHANNEL, TurnBeginEvent(
                index=execution.index,
                turn_index=execution.index,
                turn_id=execution.id,
                trigger=trigger,
            ))

            effect_txn, cancelled_during_effect_begin = await _await_known_outcome(
                _begin_effect_turn(
                    config.effect_scope,
                    bus=bus,
                    session_id=config.session_id,
                    turn_id=execution.id,
                    turn_index=execution.index,
                )
            )
            if cancelled_during_effect_begin:
                raise asyncio.CancelledError
            resource_txn, cancelled_during_resource_begin = await _await_known_outcome(
                _begin_resource_txn(
                    config.resource_writer,
                    config.services,
                    session_id=config.session_id,
                    turn_id=execution.id,
                    turn_index=execution.index,
                )
            )
            if cancelled_during_resource_begin:
                raise asyncio.CancelledError
            outcome, meta, tool_calls_used = await _react_loop(_ReactionRequest(
                execution=execution,
                trigger=trigger,
                trigger_metadata=envelope.metadata,
                config=replace(
                    config,
                    context_policies=policies,
                    interrupt=_interrupt,
                    shutdown=_shutdown,
                ),
                context_projection=context_projection,
                prompt_cache_adapter=prompt_cache_adapter,
                interruption_policy=interruption_policy,
                tool_calls_remaining=(
                    None
                    if config.max_tool_calls is None
                    else max(0, config.max_tool_calls - tool_calls_run)
                ),
            ))

            _record_interruption_message(
                execution,
                outcome,
                interruption_policy,
            )
            turn = replace(
                trajectory.prepare_commit(outcome, meta),
                trigger_metadata=envelope.metadata,
            )
            if isinstance(outcome.cause, ProviderRequestFailed):
                cleanup_errors = await _rollback_unpublished_turn(
                    resource_txn=resource_txn,
                    abandon_resource=True,
                    effect_scope=config.effect_scope,
                    effect_txn=effect_txn,
                    bus=bus,
                )
                resource_txn = None
                effect_txn = None
                _clear_resource_txn(config.services)
                if cleanup_errors:
                    raise BaseExceptionGroup(
                        "provider request and turn rollback failed",
                        [
                            RuntimeError(
                                f"{outcome.cause.error_type}: "
                                f"{outcome.cause.detail}"
                            ),
                            *cleanup_errors,
                        ],
                    )
            else:
                (
                    resource_mutations,
                    cancelled_during_resource_prepare,
                ) = await _await_known_outcome(_prepare_resource_txn(resource_txn))
                if cancelled_during_resource_prepare:
                    raise asyncio.CancelledError
                if resource_mutations:
                    turn = replace(
                        turn,
                        meta=replace(
                            turn.meta,
                            resource_mutations=resource_mutations,
                        ),
                    )
                _, cancelled_during_resource_apply = await _await_known_outcome(
                    _apply_resource_txn(resource_txn)
                )
                if cancelled_during_resource_apply:
                    raise asyncio.CancelledError
                _, cancelled_during_effect_prepare = await _await_known_outcome(
                    _prepare_effect_turn(
                        config.effect_scope,
                        effect_txn,
                        turn,
                        bus=bus,
                    )
                )
                if cancelled_during_effect_prepare:
                    raise asyncio.CancelledError
            node_append_position: _NodeAppendPosition | None = None
            if config.trajectory_node_store is not None:
                (
                    node_append_position,
                    cancelled_during_node_position,
                ) = await _await_known_outcome(
                    _node_append_position(
                        config.trajectory_node_store,
                        config.session_id,
                        committed_turns=trajectory.turns,
                        root_session_id=config.root_session_id,
                        parent_session_id=config.parent_session_id,
                        trigger_renderers=config.trigger_renderers,
                    )
                )
                if cancelled_during_node_position:
                    raise asyncio.CancelledError
            cancelled_during_append = False
            if config.store is not None:
                logger.debug("driver: appending turn {} to store {}", turn.index, type(config.store).__name__)
                cancelled_during_append = await _append_turn(
                    config.store,
                    config.session_id,
                    turn,
                )
                durable_turn_appended = True
            else:
                logger.warning("driver: store is None — turn {} NOT persisted", turn.index)
            try:
                (
                    _,
                    cancelled_during_resource_commit,
                ) = await _await_known_outcome(_commit_resource_txn(resource_txn))
                resource_txn_committed = resource_txn is not None
            except Exception:
                if durable_turn_appended:
                    trajectory.finalize_commit(turn)
                    turn_published = True
                raise
            trajectory.finalize_commit(turn)
            turn_published = True
            _clear_resource_txn(config.services)
            _, cancelled_during_effect_commit = await _await_known_outcome(
                _commit_effect_turn(config.effect_scope, effect_txn, turn, bus=bus)
            )

            if (
                config.trajectory_node_store is not None
                and node_append_position is not None
            ):
                nodes = turn_to_nodes(
                    turn,
                    session_id=config.session_id,
                    start_seq=node_append_position.start_seq,
                    root_session_id=config.root_session_id,
                    parent_session_id=config.parent_session_id,
                    branch_id=node_append_position.branch_id,
                    head_id=node_append_position.head_id,
                    parent_node_id=node_append_position.parent_node_id,
                    logical_parent_id=node_append_position.logical_parent_id,
                    renderers=config.trigger_renderers,
                )
                if nodes:
                    advance_head = TrajectoryHeadAdvance(
                        session_id=config.session_id,
                        node_id=nodes[-1].id,
                        seq=nodes[-1].seq,
                        previous_node_id=node_append_position.parent_node_id,
                        head_id=node_append_position.head_id,
                        branch_id=node_append_position.branch_id,
                        root_session_id=config.root_session_id,
                        parent_session_id=config.parent_session_id,
                        updated_at=time.time(),
                    )
                    _, cancelled_during_node_append = await _await_known_outcome(
                        _append_or_rebuild_nodes(
                            store=config.trajectory_node_store,
                            session_id=config.session_id,
                            nodes=nodes,
                            advance_head=advance_head,
                            committed_turns=trajectory.turns,
                            root_session_id=config.root_session_id,
                            parent_session_id=config.parent_session_id,
                            trigger_renderers=config.trigger_renderers,
                        )
                    )
                else:
                    cancelled_during_node_append = False
            else:
                cancelled_during_node_append = False
            if isinstance(outcome.cause, ProviderRequestFailed):
                triggers.fail(TriggerTerminated(outcome.cause))
            else:
                triggers.complete(turn)
            cancelled_during_event = False
            try:
                await bus.emit(TurnCommittedEvent.CHANNEL, TurnCommittedEvent(turn=turn))
            except asyncio.CancelledError:
                cancelled_during_event = True
            turns_run += 1
            tool_calls_run += tool_calls_used

            if any(
                (
                    cancelled_during_append,
                    cancelled_during_resource_commit,
                    cancelled_during_node_append,
                    cancelled_during_effect_commit,
                    cancelled_during_event,
                )
            ):
                triggers.terminate(TriggerTerminated("driver cancelled after commit"))
                raise asyncio.CancelledError

            if outcome.cause.session_terminal:
                triggers.terminate(TriggerTerminated(outcome.cause))
                await bus.emit(RunEndEvent.CHANNEL, RunEndEvent(
                    outcome=turn.outcome, meta=turn.meta,
                ))
                return

        except asyncio.CancelledError as exc:
            cancel_error: BaseException = exc
            if not turn_published and execution is not None and execution.rounds:
                try:
                    if execution.active:
                        turn = trajectory.prepare_commit(
                            Outcome(cause=Aborted(reason="cancelled")),
                            TurnMeta(),
                        )
                    if config.store is not None:
                        logger.debug(
                            "driver: cancel-commit turn {} ({} rounds)",
                            turn.index, len(turn.rounds),
                        )
                        await _upsert_turn(config.store, config.session_id, turn)
                    trajectory.finalize_commit(turn)
                    turn_published = True
                    triggers.complete(turn)
                except Exception as commit_exc:
                    logger.debug(
                        "driver: cancel-commit failed: {}", commit_exc,
                    )
            if not turn_published:
                cleanup_errors = await _rollback_unpublished_turn(
                    resource_txn=resource_txn,
                    abandon_resource=not resource_txn_committed,
                    effect_scope=config.effect_scope,
                    effect_txn=effect_txn,
                    bus=bus,
                )
                try:
                    trajectory.abandon()
                except BaseException as cleanup_exc:
                    logger.debug(
                        "trajectory abandon failed during cancellation: {}",
                        cleanup_exc,
                    )
                    cleanup_errors = (*cleanup_errors, cleanup_exc)
                if cleanup_errors:
                    cancel_error = BaseExceptionGroup(
                        "turn cancellation and rollback failed",
                        [exc, *cleanup_errors],
                    )
            _clear_resource_txn(config.services)
            triggers.terminate(cancel_error)
            if cancel_error is exc:
                raise
            raise cancel_error
        except Exception as exc:
            execution_error: BaseException = exc
            if not turn_published:
                cleanup_errors = await _rollback_unpublished_turn(
                    resource_txn=resource_txn,
                    abandon_resource=not resource_txn_committed,
                    effect_scope=config.effect_scope,
                    effect_txn=effect_txn,
                    bus=bus,
                )
                try:
                    trajectory.abandon()
                except BaseException as cleanup_exc:
                    logger.debug(
                        "trajectory abandon failed during rollback: {}",
                        cleanup_exc,
                    )
                    cleanup_errors = (*cleanup_errors, cleanup_exc)
                if cleanup_errors:
                    execution_error = BaseExceptionGroup(
                        "turn execution and rollback failed",
                        [exc, *cleanup_errors],
                    )
            _clear_resource_txn(config.services)
            triggers.terminate(execution_error)
            diagnostic_message = (
                "driver stopped after a committed turn"
                if turn_published
                else "turn abandoned: trigger dropped due to internal error"
            )
            logger.exception("driver: {}", diagnostic_message)
            try:
                await bus.emit(DiagnosticEvent.CHANNEL, DiagnosticEvent(
                    level="error",
                    source="driver",
                    message=diagnostic_message,
                ))
            except Exception:
                logger.debug("diagnostic emit failed after turn abandon; non-fatal")
            return


async def _react_loop(
    request: _ReactionRequest,
) -> tuple[Outcome, TurnMeta, int]:
    """ReAct loop within one turn.  Returns when a Stop action fires."""

    config = request.config
    execution = request.execution
    trajectory = config.trajectory
    trigger = request.trigger
    trigger_metadata = request.trigger_metadata
    bus = config.bus
    stream_fn = config.stream_fn
    model = config.model
    tools = config.tools
    system = config.system
    policies = config.context_policies or []
    context_projection = request.context_projection
    prompt_cache_adapter = request.prompt_cache_adapter
    trigger_renderers = config.trigger_renderers
    interrupt = config.interrupt
    shutdown = config.shutdown
    if interrupt is None or shutdown is None:
        raise RuntimeError("reaction requires resolved cancellation sources")
    parent_cancel_signal = config.cancel_signal
    thinking = config.thinking
    tool_executor = config.tool_executor
    tool_orchestrator = config.tool_orchestrator
    permission_policy = config.permission_policy
    trajectory_node_store = config.trajectory_node_store
    session_id = config.session_id
    root_session_id = config.root_session_id
    parent_session_id = config.parent_session_id
    permission_audience = config.permission_audience
    interruption_policy = request.interruption_policy
    tool_calls_remaining = request.tool_calls_remaining
    tool_allowlist = config.tool_allowlist

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    tool_calls_used = 0
    start_ns = time.perf_counter_ns()
    trigger_messages = apply_trigger_metadata(
        render_trigger(trigger, trigger_renderers),
        trigger_metadata,
    )
    context_budget = _context_budget(model)
    turn_signal = _TurnCancelSignal(
        interrupt=interrupt,
        shutdown=shutdown,
        parent=parent_cancel_signal,
    )
    try:
        history_messages = await _history_messages(
            turns=trajectory.turns,
            policies=policies,
            trigger_renderers=trigger_renderers,
            projection=context_projection,
            budget=context_budget,
            trajectory_node_store=trajectory_node_store,
            session_id=session_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            signal=turn_signal,
        )
    except ContextTransformCancelled:
        return Outcome(cause=_signal_aborted(turn_signal)), _meta(
            0,
            0,
            start_ns,
        ), tool_calls_used
    messages = list(history_messages) + list(trigger_messages)

    before_returns = await bus.emit(BeforeRunEvent.CHANNEL, BeforeRunEvent(
        messages=tuple(messages), system=system,
    ))
    veto = _last_key(before_returns, "veto")
    if veto is not None:
        return Outcome(cause=veto), _meta(0, 0, start_ns), tool_calls_used
    replacement_msgs = _last_key(before_returns, "messages")
    if isinstance(replacement_msgs, list):
        messages = replacement_msgs
    replacement_sys = _last_key(before_returns, "system")
    if isinstance(replacement_sys, str):
        system = replacement_sys

    round_index = 0
    while True:
        if turn_signal.is_set():
            return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                total_input, total_output, start_ns,
                cache_read=total_cache_read, cache_write=total_cache_write,
            ), tool_calls_used

        if round_index > 0:
            try:
                history_messages = await _history_messages(
                    turns=trajectory.turns,
                    policies=policies,
                    trigger_renderers=trigger_renderers,
                    projection=context_projection,
                    budget=context_budget,
                    trajectory_node_store=trajectory_node_store,
                    session_id=session_id,
                    root_session_id=root_session_id,
                    parent_session_id=parent_session_id,
                    signal=turn_signal,
                )
            except ContextTransformCancelled:
                return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                    total_input,
                    total_output,
                    start_ns,
                    cache_read=total_cache_read,
                    cache_write=total_cache_write,
                ), tool_calls_used
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

        messages = route_messages(messages, session_id=session_id)
        messages = await _apply_provider_prompt_cache(
            messages=messages,
            model=effective_model,
            adapter=prompt_cache_adapter,
            store=trajectory_node_store,
            session_id=session_id,
        )

        allowed_tool_names = set(tool_allowlist) if tool_allowlist is not None else None
        if allowed_tool_names is not None:
            effective_tools = [
                tool for tool in effective_tools if tool.name in allowed_tool_names
            ]
        if tool_calls_remaining is not None and tool_calls_used >= tool_calls_remaining:
            effective_tools = []
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
                messages=messages, model=effective_model,
                tools=effective_tools, system=effective_system,
                signal=turn_signal, thinking=thinking,
            ):
                stream_events.append(ev)
                await bus.emit(StreamDeltaEvent.CHANNEL, StreamDeltaEvent(
                    turn_index=execution.index, delta=ev,
                ))
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
                execution.add_round(response, interrupted_records)
                return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                    total_input, total_output, start_ns,
                    cache_read=total_cache_read, cache_write=total_cache_write,
                ), tool_calls_used
            if isinstance(exc, asyncio.CancelledError) or not isinstance(
                exc,
                Exception,
            ):
                raise
            if stream_events:
                execution.add_round(
                    _assemble_assistant_message(stream_events),
                    [],
                )
            return Outcome(
                cause=ProviderRequestFailed(
                    error_type=type(exc).__name__,
                    detail=str(exc),
                    partial_event_count=len(stream_events),
                )
            ), _meta(
                total_input,
                total_output,
                start_ns,
                effective_model,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
            ), tool_calls_used
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

        if turn_signal.is_set() or isinstance(response.termination, Aborted):
            _append_interrupted_tool_records(
                calls=tool_calls,
                records=tool_records,
                reason=cancel_reason(turn_signal) or "unknown",
                policy=interruption_policy,
            )
            execution.add_round(response, tool_records)
            return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                total_input, total_output, start_ns, effective_model,
                cache_read=total_cache_read, cache_write=total_cache_write,
            ), tool_calls_used

        if tool_calls:
            if (
                tool_calls_remaining is not None
                and len(tool_calls)
                > max(0, tool_calls_remaining - tool_calls_used)
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
                                            "Tool call skipped: max_tool_calls "
                                            "exhausted"
                                        ),
                                    )
                                ],
                                is_error=True,
                            ),
                        )
                    )
                execution.add_round(response, tool_records)
                return Outcome(cause=BudgetExhausted(
                    detail="max_tool_calls exhausted"
                )), _meta(
                    total_input,
                    total_output,
                    start_ns,
                    effective_model,
                    cache_read=total_cache_read,
                    cache_write=total_cache_write,
                ), tool_calls_used

            result_blocks: list[ToolResultBlock] = []
            immediate_outcomes: dict[int, ToolOutcome] = {}
            work_items: list[ToolWorkItem] = []
            for index, tc in enumerate(tool_calls):
                if turn_signal.is_set():
                    tool_records = _cancelled_tool_records(
                        calls=tool_calls,
                        outcomes=immediate_outcomes,
                        reason=cancel_reason(turn_signal) or "unknown",
                        policy=interruption_policy,
                    )
                    tool_calls_used += len(immediate_outcomes)
                    execution.add_round(response, tool_records)
                    return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                        total_input, total_output, start_ns,
                        cache_read=total_cache_read, cache_write=total_cache_write,
                    ), tool_calls_used

                tc_returns = await bus.emit(ToolCallEvent.CHANNEL, ToolCallEvent(
                    tool_call_id=tc.id, tool_name=tc.name, args=dict(tc.arguments),
                ))
                blocked = _last_key(tc_returns, "block")

                if blocked:
                    reason = _last_key(tc_returns, "reason") or "blocked"
                    immediate_outcomes[index] = ToolContinue(result=ToolResult(
                        content=[TextContent(type="text", text=f"blocked: {reason}")],
                        is_error=True,
                    ))
                    continue

                rewrite = _last_key(tc_returns, "rewrite")
                args = dict(tc.arguments)
                if isinstance(rewrite, dict):
                    args.update(rewrite)
                if allowed_tool_names is not None and tc.name not in allowed_tool_names:
                    immediate_outcomes[index] = ToolContinue(result=ToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"blocked by tool_allowlist: {tc.name}",
                        )],
                        is_error=True,
                    ))
                    continue

                tool = tool_index.get(tc.name)
                if tool is None:
                    immediate_outcomes[index] = ToolContinue(result=ToolResult(
                        content=[TextContent(type="text", text=f"unknown tool: {tc.name}")],
                        is_error=True,
                    ))
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
                        outcomes=immediate_outcomes,
                        reason=cancel_reason(turn_signal) or "unknown",
                        policy=interruption_policy,
                    )
                    tool_calls_used += len(immediate_outcomes)
                    execution.add_round(response, tool_records)
                    return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                        total_input, total_output, start_ns,
                        cache_read=total_cache_read,
                        cache_write=total_cache_write,
                    ), tool_calls_used
                if permission_outcome is not None:
                    immediate_outcomes[index] = permission_outcome
                    continue

                work_items.append(
                    ToolWorkItem(
                        index=index,
                        call=tc,
                        tool=tool,
                        args=args,
                        requirements=tool_execution_requirements(tool)
                        or ToolExecutionRequirements(),
                    )
                )

            outcomes_by_index: dict[int, ToolOutcome] = dict(immediate_outcomes)
            if work_items:
                orchestrator = tool_orchestrator or default_tool_orchestrator()
                orchestration_results = await orchestrator.execute_batch(
                    ToolOrchestrationRequest(
                        items=tuple(work_items),
                        session_id=session_id,
                        turn_id=execution.id,
                        turn_index=execution.index,
                    ),
                    signal=turn_signal,
                    executor=tool_executor,
                )
                for orch_result in orchestration_results:
                    if (
                        turn_signal.is_set()
                        and orch_result.status in {"cancelled", "skipped"}
                    ):
                        continue
                    if orch_result.status == "completed" and orch_result.output is not None:
                        outcomes_by_index[orch_result.item.index] = _normalize_tool_output(
                            orch_result.output
                        )
                    else:
                        outcomes_by_index[orch_result.item.index] = (
                            await _orchestration_failure_outcome(
                                result=orch_result,
                                bus=bus,
                            )
                        )

            if turn_signal.is_set() and len(outcomes_by_index) < len(tool_calls):
                tool_records = _cancelled_tool_records(
                    calls=tool_calls,
                    outcomes=outcomes_by_index,
                    reason=cancel_reason(turn_signal) or "unknown",
                    policy=interruption_policy,
                )
                tool_calls_used += len(outcomes_by_index)
                execution.add_round(response, tool_records)
                return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                    total_input, total_output, start_ns,
                    cache_read=total_cache_read,
                    cache_write=total_cache_write,
                ), tool_calls_used

            for index, tc in enumerate(tool_calls):
                outcome = outcomes_by_index.get(index)
                if outcome is None:
                    continue
                result = _outcome_result(outcome)
                res_returns = await bus.emit(ToolResultEvent.CHANNEL, ToolResultEvent(
                    tool_call_id=tc.id, tool_name=tc.name, result=result,
                ))
                replaced = _last_of(res_returns, ToolResult)
                final_result = replaced if replaced is not None else result
                final_outcome = _replace_outcome_result(outcome, final_result)

                result_block = _tool_result_block(tc.id, final_result)
                result_blocks.append(result_block)
                tool_records.append(ToolRecord(call=tc, result=result_block))
                paired_outcomes.append((tc.name, final_outcome))
                tool_calls_used += 1

            if result_blocks:
                messages.append(ToolResultMessage(
                    role="tool_result", content=result_blocks, timestamp=time.time(),
                ))

        if turn_signal.is_set():
            execution.add_round(response, tool_records)
            return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                total_input, total_output, start_ns,
                cache_read=total_cache_read,
                cache_write=total_cache_write,
            ), tool_calls_used

        execution.add_round(response, tool_records)

        if config.store is not None:
            try:
                in_progress_meta = _meta(
                    total_input, total_output, start_ns, effective_model,
                    cache_read=total_cache_read, cache_write=total_cache_write,
                )
                snapshot = trajectory.snapshot_in_progress(
                    Outcome(cause=SignalAborted(reason="in_progress")),
                    in_progress_meta,
                )
                if snapshot is not None:
                    await asyncio.to_thread(
                        config.store.upsert_turn, session_id, snapshot,
                    )
            except Exception as _persist_exc:
                logger.debug("incremental turn persist failed (non-fatal): {}", _persist_exc)

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
            ), tool_calls_used

        if isinstance(action, Inject):
            execution.add_injected(list(action.messages))

        round_index += 1


__all__ = ["ThinkingLevel", "drive"]
