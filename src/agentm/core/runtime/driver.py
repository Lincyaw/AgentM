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
from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, TypeVar, cast

from loguru import logger

from agentm.core.abi.cancel import (
    CancelSignal,
    EventCancelSource,
    ResettableCancelSource,
)
from agentm.core.abi.compaction import (
    ContextProjection,
)
from agentm.core.abi.lifecycle import EffectScope, EffectTxn
from agentm.core.abi.messages import (
    InterruptionMessagePolicy,
)
from agentm.core.abi.permission import (
    PermissionAudience,
    PermissionPolicy,
)
from agentm.core.abi.provider import (
    ProviderPromptCacheAdapter,
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
    Model,
    StreamFn,
)
from agentm.core.abi.tool import (
    Tool,
)
from agentm.core.abi.tool_executor import (
    ToolExecutor,
)
from agentm.core.abi.tool_orchestration import (
    ToolOrchestrator,
)
from agentm.core.abi.bus import EventBus
from agentm.core.abi.context import (
    ContextPolicy,
)
from agentm.core.abi.events import (
    DiagnosticEvent,
    RunEndEvent,
    TurnBeginEvent,
    TurnCommittedEvent,
)
from agentm.core.abi.store import (
    TrajectoryCommit,
    TrajectoryNodeQuery,
    TrajectoryStore,
)
from agentm.core.abi.termination import (
    MaxTurnsExhausted,
    ProviderRequestFailed,
)
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    Outcome,
    TrajectoryHeadAdvance,
    TrajectoryNode,
    Turn,
    TurnCheckpoint,
    TurnMeta,
)
from agentm.core.abi.trigger import (
    TriggerRenderer,
)
from agentm.core.runtime.reaction import (
    ReactionRequest,
    react,
    record_interruption_message,
)
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.lib.trajectory_nodes import turn_to_nodes
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


async def _emit_lifecycle_diagnostic(
    bus: EventBus,
    *,
    boundary: str,
    action: str,
    exc: BaseException,
) -> None:
    try:
        await bus.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="error",
                source="lifecycle",
                message=f"{boundary} {action} failed: {type(exc).__name__}: {exc}",
            ),
        )
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


async def _commit_trajectory_turn(
    store: TrajectoryStore,
    session_id: str,
    commit: TrajectoryCommit,
) -> bool:
    """Commit durably, returning whether the caller was cancelled meanwhile."""
    task = asyncio.create_task(asyncio.to_thread(store.commit_turn, session_id, commit))
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


async def _save_checkpoint(
    store: TrajectoryStore,
    session_id: str,
    checkpoint: TurnCheckpoint,
) -> None:
    """Persist one checkpoint before honoring caller cancellation."""
    task = asyncio.create_task(
        asyncio.to_thread(store.save_checkpoint, session_id, checkpoint)
    )
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


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
    store: TrajectoryStore,
    session_id: str,
) -> _NodeAppendPosition:
    """Load the explicit append head from the authoritative store."""

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
    if head is None or head.status != "active":
        raise RuntimeError(
            f"trajectory store for session {session_id} has no active append head"
        )
    return _NodeAppendPosition(
        start_seq=start_seq,
        parent_node_id=head.node_id,
        logical_parent_id=head.logical_parent_id if head.node_id is None else None,
        head_id=head.head_id,
        branch_id=head.branch_id,
    )


def _clear_resource_txn(services: ServiceRegistry | None) -> None:
    if services is not None:
        services.unregister(RESOURCE_TXN_SERVICE)


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
    max_turns: int | None = None
    thinking: ThinkingLevel = "off"
    max_tool_calls: int | None = None
    tool_allowlist: tuple[str, ...] | None = None


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
        durable_turn_committed = False
        turn_published = False
        try:
            execution = trajectory.begin(trigger)
            await bus.emit(
                TurnBeginEvent.CHANNEL,
                TurnBeginEvent(
                    index=execution.index,
                    turn_index=execution.index,
                    turn_id=execution.id,
                    trigger=trigger,
                ),
            )
            checkpoint_writer = (
                partial(
                    _save_checkpoint,
                    config.store,
                    config.session_id,
                )
                if config.store is not None
                else None
            )
            if checkpoint_writer is not None:
                await checkpoint_writer(
                    execution.checkpoint(
                        TurnMeta(),
                        trigger_metadata=envelope.metadata,
                    )
                )

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
            outcome, meta, tool_calls_used = await react(
                ReactionRequest(
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
                    checkpoint=checkpoint_writer,
                    tool_calls_remaining=(
                        None
                        if config.max_tool_calls is None
                        else max(0, config.max_tool_calls - tool_calls_run)
                    ),
                )
            )

            record_interruption_message(
                execution,
                outcome,
                interruption_policy,
            )
            if checkpoint_writer is not None:
                await checkpoint_writer(
                    execution.checkpoint(
                        meta,
                        trigger_metadata=envelope.metadata,
                    )
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
                                f"{outcome.cause.error_type}: {outcome.cause.detail}"
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
            if config.store is not None:
                (
                    node_append_position,
                    cancelled_during_node_position,
                ) = await _await_known_outcome(
                    _node_append_position(
                        config.store,
                        config.session_id,
                    )
                )
                if cancelled_during_node_position:
                    raise asyncio.CancelledError
            nodes: list[TrajectoryNode] = []
            advance_head: TrajectoryHeadAdvance | None = None
            if node_append_position is not None:
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
            cancelled_during_trajectory_commit = False
            if config.store is not None:
                cancelled_during_trajectory_commit = await _commit_trajectory_turn(
                    config.store,
                    config.session_id,
                    TrajectoryCommit(
                        turn=turn,
                        nodes=tuple(nodes),
                        advance_head=advance_head,
                    ),
                )
                durable_turn_committed = True
            try:
                (
                    _,
                    cancelled_during_resource_commit,
                ) = await _await_known_outcome(_commit_resource_txn(resource_txn))
                resource_txn_committed = resource_txn is not None
            except Exception:
                if durable_turn_committed:
                    trajectory.finalize_commit(turn)
                    turn_published = True
                raise
            trajectory.finalize_commit(turn)
            turn_published = True
            _clear_resource_txn(config.services)
            _, cancelled_during_effect_commit = await _await_known_outcome(
                _commit_effect_turn(config.effect_scope, effect_txn, turn, bus=bus)
            )

            if isinstance(outcome.cause, ProviderRequestFailed):
                triggers.fail(TriggerTerminated(outcome.cause))
            else:
                triggers.complete(turn)
            cancelled_during_event = False
            try:
                await bus.emit(
                    TurnCommittedEvent.CHANNEL, TurnCommittedEvent(turn=turn)
                )
            except asyncio.CancelledError:
                cancelled_during_event = True
            turns_run += 1
            tool_calls_run += tool_calls_used

            if any(
                (
                    cancelled_during_trajectory_commit,
                    cancelled_during_resource_commit,
                    cancelled_during_effect_commit,
                    cancelled_during_event,
                )
            ):
                triggers.terminate(TriggerTerminated("driver cancelled after commit"))
                raise asyncio.CancelledError

            if outcome.cause.session_terminal:
                triggers.terminate(TriggerTerminated(outcome.cause))
                await bus.emit(
                    RunEndEvent.CHANNEL,
                    RunEndEvent(
                        outcome=turn.outcome,
                        meta=turn.meta,
                    ),
                )
                return

        except asyncio.CancelledError as exc:
            cancel_error: BaseException = exc
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
                await bus.emit(
                    DiagnosticEvent.CHANNEL,
                    DiagnosticEvent(
                        level="error",
                        source="driver",
                        message=diagnostic_message,
                    ),
                )
            except Exception:
                logger.debug("diagnostic emit failed after turn abandon; non-fatal")
            return


__all__ = ["ThinkingLevel", "drive"]
