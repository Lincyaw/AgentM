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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

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
    project_context,
    supports_node_chain_projection,
)
from agentm.core.abi.lifecycle import EffectScope, EffectTxn
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)
from agentm.core.abi.permission import (
    PermissionPolicy,
    PermissionRequest,
    permission_denial_result,
)
from agentm.core.abi.resource import (
    ResourceMutation,
    ResourceTxn,
    ResourceTxnContext,
    ResourceWriter,
    TransactionalResourceWriter,
)
from agentm.core.abi.roles import CONTEXT_PROJECTION_SERVICE, RESOURCE_TXN_SERVICE
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
from agentm.core.abi.context import ContextPolicy, build_context, render_trigger
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
    ProviderTruncated,
    SignalAborted,
    TerminationCause,
    ToolTerminated,
)
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    Outcome,
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
    TriggerRenderer,
)
from agentm.core.runtime.execution import Execution
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.tool_orchestration import default_tool_orchestrator
from agentm.core.runtime.trajectory_nodes import turn_to_nodes, turns_to_nodes
from agentm.core.runtime.trigger_queue import (
    QueueClosed,
    TriggerQueue,
    TriggerTerminated,
)

ThinkingLevel = Literal["off", "low", "medium", "high"]


# --- Helpers ----------------------------------------------------------------

_INTERRUPTED_BY_USER = "Interrupted by user"


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


def _interrupted_tool_record(call: ToolCallBlock) -> ToolRecord:
    return ToolRecord(
        call=call,
        result=ToolResultBlock(
            type="tool_result",
            tool_call_id=call.id,
            content=[TextContent(type="text", text=_INTERRUPTED_BY_USER)],
            is_error=True,
        ),
    )


def _append_interrupted_tool_records(
    *,
    calls: list[ToolCallBlock],
    records: list[ToolRecord],
    result_blocks: list[ToolResultBlock] | None = None,
) -> None:
    for call in calls:
        record = _interrupted_tool_record(call)
        records.append(record)
        if result_blocks is not None:
            result_blocks.append(record.result)


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


def _outcome_result(outcome: ToolOutcome) -> ToolResult:
    if isinstance(outcome, (ToolContinue, ToolTerminate)):
        return outcome.result
    return ToolResult(content=[], is_error=True)


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
) -> PermissionRequest:
    return PermissionRequest(
        action="tool_call",
        session_id=session_id,
        turn_id=execution.id,
        turn_index=execution.index,
        tool_call_id=tc.id,
        tool_name=tc.name,
        args=args,
        audience="subagent" if session_id and ":" in session_id else "user",
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
    action: str,
    exc: BaseException,
) -> None:
    try:
        await bus.emit(DiagnosticEvent.CHANNEL, DiagnosticEvent(
            level="error",
            source="lifecycle",
            message=f"effect scope {action} failed: {type(exc).__name__}: {exc}",
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
        await _emit_lifecycle_diagnostic(bus, action="begin_turn", exc=exc)
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
        await _emit_lifecycle_diagnostic(bus, action="commit_turn", exc=exc)
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
            # Cancelling to_thread only cancels its awaiter.  The append is the
            # commit boundary, so repeated cancellation must not obscure its result.
            cancelled = True
    task.result()
    return cancelled


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
        await _emit_lifecycle_diagnostic(bus, action="abandon_turn", exc=exc)


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


async def _commit_resource_txn(
    txn: ResourceTxn | None,
) -> tuple[ResourceMutation, ...]:
    if txn is None:
        return ()
    return tuple(await txn.commit())


async def _abandon_resource_txn(txn: ResourceTxn | None) -> None:
    if txn is not None:
        await txn.abandon()


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
    """Update the non-authoritative node projection without failing turn commit."""

    try:
        await asyncio.to_thread(
            store.append_nodes,
            session_id,
            nodes,
            advance_head=advance_head,
        )
    except Exception:
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
        except Exception:
            logger.exception(
                "trajectory node projection rebuild failed for session {}",
                session_id,
            )


def _clear_resource_txn(services: ServiceRegistry | None) -> None:
    if services is not None:
        services.unregister(RESOURCE_TXN_SERVICE)


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
) -> list[AgentMessage]:
    if projection is None:
        return await build_context(turns, policies, trigger_renderers)
    projection_input = await _projection_input(
        turns=turns,
        projection=projection,
        trajectory_node_store=trajectory_node_store,
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
    )
    messages = list(project_context(projection, projection_input, budget))
    for policy in policies:
        messages = await policy.transform(messages, turns)
    return messages


async def _projection_input(
    *,
    turns: Sequence[Turn],
    projection: ContextProjection,
    trajectory_node_store: TrajectoryNodeStore | None,
    session_id: str,
    root_session_id: str | None,
    parent_session_id: str | None,
) -> ProjectionInput:
    if not supports_node_chain_projection(projection):
        return ProjectionInput(
            turns=turns,
            session_id=session_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
        )
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
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
    system: str | None = None,
    context_policies: list[ContextPolicy] | None = None,
    trigger_renderers: dict[str, TriggerRenderer] | None = None,
    interrupt: ResettableCancelSource | None = None,
    shutdown: ResettableCancelSource | None = None,
    cancel_signal: CancelSignal | None = None,
    effect_scope: EffectScope | None = None,
    resource_writer: ResourceWriter | None = None,
    services: ServiceRegistry | None = None,
    tool_executor: ToolExecutor | None = None,
    tool_orchestrator: ToolOrchestrator | None = None,
    permission_policy: PermissionPolicy | None = None,
    trajectory_node_store: TrajectoryNodeStore | None = None,
    max_turns: int | None = None,
    thinking: ThinkingLevel = "off",
    max_tool_calls: int | None = None,
    tool_allowlist: tuple[str, ...] | None = None,
) -> None:
    """Persistent driver loop.

    Processes triggers one at a time.  Each trigger becomes one Turn
    (potentially with multiple ReAct rounds).  Exits when:
    - shutdown is set
    - trigger queue is closed (QueueClosed)
    - max_turns committed turns reached
    - a session-terminal cause fires (ToolTerminated, BudgetExhausted)
    """

    _interrupt = interrupt or EventCancelSource()
    _shutdown = shutdown or EventCancelSource()
    policies = context_policies or []
    context_projection = (
        services.get(
            CONTEXT_PROJECTION_SERVICE,
            cast(type[ContextProjection], ContextProjection),
        )
        if services is not None
        else None
    )
    turns_run = 0
    tool_calls_run = 0

    while True:
        if _shutdown.is_set():
            triggers.terminate(TriggerTerminated("session shut down"))
            await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
            return

        if max_turns is not None and turns_run >= max_turns:
            cause = MaxTurnsExhausted()
            triggers.terminate(TriggerTerminated(cause))
            await bus.emit(
                RunEndEvent.CHANNEL,
                RunEndEvent(outcome=Outcome(cause=cause)),
            )
            return

        try:
            trigger = await triggers.wait()
        except QueueClosed:
            triggers.terminate(TriggerTerminated("trigger queue closed"))
            await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
            return

        _interrupt.clear()

        execution = trajectory.begin(trigger)
        await bus.emit(TurnBeginEvent.CHANNEL, TurnBeginEvent(
            index=execution.index,
            turn_index=execution.index,
            turn_id=execution.id,
            trigger=trigger,
        ))

        effect_txn: EffectTxn | None = None
        resource_txn: ResourceTxn | None = None
        resource_txn_committed = False
        turn_published = False
        try:
            effect_txn = await _begin_effect_turn(
                effect_scope,
                bus=bus,
                session_id=session_id,
                turn_id=execution.id,
                turn_index=execution.index,
            )
            resource_txn = await _begin_resource_txn(
                resource_writer,
                services,
                session_id=session_id,
                turn_id=execution.id,
                turn_index=execution.index,
            )
            outcome, meta, tool_calls_used = await _react_loop(
                execution=execution,
                trajectory=trajectory,
                trigger=trigger,
                bus=bus,
                stream_fn=stream_fn,
                model=model,
                tools=tools,
                system=system,
                policies=policies,
                context_projection=context_projection,
                trigger_renderers=trigger_renderers,
                interrupt=_interrupt,
                shutdown=_shutdown,
                parent_cancel_signal=cancel_signal,
                thinking=thinking,
                tool_executor=tool_executor,
                tool_orchestrator=tool_orchestrator,
                permission_policy=permission_policy,
                trajectory_node_store=trajectory_node_store,
                store=store,
                session_id=session_id,
                root_session_id=root_session_id,
                parent_session_id=parent_session_id,
                tool_calls_remaining=(
                    None
                    if max_tool_calls is None
                    else max(0, max_tool_calls - tool_calls_run)
                ),
                tool_allowlist=tool_allowlist,
            )

            turn = trajectory.prepare_commit(outcome, meta)
            resource_mutations = await _commit_resource_txn(resource_txn)
            resource_txn_committed = resource_txn is not None
            if resource_mutations:
                turn = replace(
                    turn,
                    meta=replace(
                        turn.meta,
                        resource_mutations=resource_mutations,
                    ),
                )
            node_append_position: _NodeAppendPosition | None = None
            if trajectory_node_store is not None:
                node_append_position = await _node_append_position(
                    trajectory_node_store,
                    session_id,
                    committed_turns=trajectory.turns,
                    root_session_id=root_session_id,
                    parent_session_id=parent_session_id,
                    trigger_renderers=trigger_renderers,
                )
            cancelled_during_append = False
            if store is not None:
                cancelled_during_append = await _append_turn(store, session_id, turn)
            if trajectory_node_store is not None and node_append_position is not None:
                nodes = turn_to_nodes(
                    turn,
                    session_id=session_id,
                    start_seq=node_append_position.start_seq,
                    root_session_id=root_session_id,
                    parent_session_id=parent_session_id,
                    branch_id=node_append_position.branch_id,
                    head_id=node_append_position.head_id,
                    parent_node_id=node_append_position.parent_node_id,
                    logical_parent_id=node_append_position.logical_parent_id,
                    renderers=trigger_renderers,
                )
                if nodes:
                    advance_head = TrajectoryHeadAdvance(
                        session_id=session_id,
                        node_id=nodes[-1].id,
                        seq=nodes[-1].seq,
                        previous_node_id=node_append_position.parent_node_id,
                        head_id=node_append_position.head_id,
                        branch_id=node_append_position.branch_id,
                        root_session_id=root_session_id,
                        parent_session_id=parent_session_id,
                        updated_at=time.time(),
                    )
                    await _append_or_rebuild_nodes(
                        store=trajectory_node_store,
                        session_id=session_id,
                        nodes=nodes,
                        advance_head=advance_head,
                        committed_turns=(*trajectory.turns, turn),
                        root_session_id=root_session_id,
                        parent_session_id=parent_session_id,
                        trigger_renderers=trigger_renderers,
                    )
            trajectory.finalize_commit(turn)
            turn_published = True
            _clear_resource_txn(services)

            await _commit_effect_turn(effect_scope, effect_txn, turn, bus=bus)
            triggers.complete(turn)
            try:
                await bus.emit(TurnCommittedEvent.CHANNEL, TurnCommittedEvent(turn=turn))
            except asyncio.CancelledError:
                logger.debug("TurnCommittedEvent emit interrupted by cancellation")
            turns_run += 1
            tool_calls_run += tool_calls_used

            if cancelled_during_append:
                triggers.terminate(TriggerTerminated("driver cancelled after commit"))
                raise asyncio.CancelledError

            if getattr(outcome.cause, "session_terminal", False):
                triggers.terminate(TriggerTerminated(outcome.cause))
                await bus.emit(RunEndEvent.CHANNEL, RunEndEvent(
                    outcome=turn.outcome, meta=turn.meta,
                ))
                return

        except asyncio.CancelledError as exc:
            if not turn_published:
                if not resource_txn_committed:
                    await _abandon_resource_txn(resource_txn)
                await _abandon_effect_turn(effect_scope, effect_txn, bus=bus)
                trajectory.abandon()
            _clear_resource_txn(services)
            triggers.terminate(exc)
            raise
        except Exception as exc:
            if not turn_published:
                if not resource_txn_committed:
                    await _abandon_resource_txn(resource_txn)
                await _abandon_effect_turn(effect_scope, effect_txn, bus=bus)
                trajectory.abandon()
            _clear_resource_txn(services)
            triggers.terminate(exc)
            logger.exception("driver: round raised; abandoning turn")
            try:
                await bus.emit(DiagnosticEvent.CHANNEL, DiagnosticEvent(
                    level="error",
                    source="driver",
                    message="turn abandoned: trigger dropped due to internal error",
                ))
            except Exception:
                logger.debug("diagnostic emit failed after turn abandon; non-fatal")
            return


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
    context_projection: ContextProjection | None,
    trigger_renderers: dict[str, TriggerRenderer] | None,
    interrupt: ResettableCancelSource,
    shutdown: ResettableCancelSource,
    parent_cancel_signal: CancelSignal | None,
    thinking: ThinkingLevel,
    tool_executor: ToolExecutor | None,
    tool_orchestrator: ToolOrchestrator | None,
    permission_policy: PermissionPolicy | None,
    trajectory_node_store: TrajectoryNodeStore | None,
    store: TrajectoryStore | None,
    session_id: str,
    root_session_id: str | None,
    parent_session_id: str | None,
    tool_calls_remaining: int | None,
    tool_allowlist: tuple[str, ...] | None,
) -> tuple[Outcome, TurnMeta, int]:
    """ReAct loop within one turn.  Returns when a Stop action fires."""

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    tool_calls_used = 0
    start_ns = time.perf_counter_ns()
    trigger_messages = render_trigger(trigger, trigger_renderers)
    context_budget = _context_budget(model)

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
    )
    messages = list(history_messages) + list(trigger_messages)
    turn_signal = _TurnCancelSignal(
        interrupt=interrupt,
        shutdown=shutdown,
        parent=parent_cancel_signal,
    )

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
            )
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
                model_id=getattr(effective_model, "id", "unknown"),
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
                interrupted_records = [
                    _interrupted_tool_record(call)
                    for call in _extract_tool_calls(response)
                ]
                execution.add_round(response, interrupted_records)
                return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                    total_input, total_output, start_ns,
                    cache_read=total_cache_read, cache_write=total_cache_write,
                ), tool_calls_used
            raise
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
            _append_interrupted_tool_records(calls=tool_calls, records=tool_records)
            execution.add_round(response, tool_records)
            return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                total_input, total_output, start_ns, effective_model,
                cache_read=total_cache_read, cache_write=total_cache_write,
            ), tool_calls_used

        if tool_calls:
            result_blocks: list[ToolResultBlock] = []
            immediate_outcomes: dict[int, ToolOutcome] = {}
            work_items: list[ToolWorkItem] = []
            for index, tc in enumerate(tool_calls):
                if turn_signal.is_set():
                    _append_interrupted_tool_records(
                        calls=tool_calls[index:],
                        records=tool_records,
                        result_blocks=result_blocks,
                    )
                    execution.add_round(response, tool_records)
                    return Outcome(cause=_signal_aborted(turn_signal)), _meta(
                        total_input, total_output, start_ns,
                        cache_read=total_cache_read, cache_write=total_cache_write,
                    ), tool_calls_used

                if (
                    tool_calls_remaining is not None
                    and (
                        tool_calls_used
                        + len(immediate_outcomes)
                        + len(work_items)
                    ) >= tool_calls_remaining
                ):
                    execution.add_round(response, tool_records)
                    return Outcome(cause=BudgetExhausted(
                        detail="max_tool_calls exhausted"
                    )), _meta(
                        total_input, total_output, start_ns, effective_model,
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
                    ),
                    signal=turn_signal,
                )
                if turn_signal.is_set():
                    _append_interrupted_tool_records(
                        calls=tool_calls[index:],
                        records=tool_records,
                        result_blocks=result_blocks,
                    )
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
                missing = [
                    call
                    for index, call in enumerate(tool_calls)
                    if index not in outcomes_by_index
                ]
                _append_interrupted_tool_records(
                    calls=missing,
                    records=tool_records,
                    result_blocks=result_blocks,
                )
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

                result_block = _tool_result_block(tc.id, final_result)
                result_blocks.append(result_block)
                tool_records.append(ToolRecord(call=tc, result=result_block))
                paired_outcomes.append((tc.name, outcome))
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
