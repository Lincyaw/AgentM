# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
"""Session driver — persistent loop that converts triggers into committed turns.

Design:
- One accepted trigger owns a prompt run and one completion receipt.
- Every provider response and its tool results commit as one Turn and transaction.
- A continuation Turn stays attached to the accepted trigger until Stop.
- interrupt: aborts only the active Turn; already committed Turns stay intact.
- shutdown: aborts current turn AND exits driver.
- ToolTerminate: sets shutdown, turn commits, driver exits.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import cast
from uuid import uuid4

from loguru import logger

from agentm.core.abi.cancel import (
    CancelSignal,
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
    ThinkingLevel,
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
    TrajectoryDiagnostic,
    TrajectoryNodeQuery,
    TrajectoryStore,
)
from agentm.core.abi.termination import (
    PromptRunContinued,
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
    ContinueTrigger,
    TriggerEnvelope,
    TriggerMetadata,
    TriggerRenderer,
)
from agentm.core.runtime.reaction import (
    ReactionRequest,
    react,
    record_interruption_message,
)
from agentm.core.lib.async_cancel import await_known_outcome, settle_known_outcome
from agentm.core.lib.redact import redact_text_secrets
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.lib.trajectory_nodes import turn_to_nodes
from agentm.core.runtime.trigger_queue import (
    QueueClosed,
    TriggerQueue,
    TriggerTerminated,
)

# --- Helpers ----------------------------------------------------------------

_INTERRUPTED_TOOL_TEXT = "Tool execution interrupted"


@dataclass(frozen=True, slots=True)
class _NodeAppendPosition:
    start_seq: int
    parent_node_id: str | None
    logical_parent_id: str | None
    head_id: str
    branch_id: str


@dataclass(frozen=True, slots=True)
class _RollbackResult:
    errors: tuple[BaseException, ...]
    caller_cancelled: bool


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
    _, cancelled = await settle_known_outcome(
        asyncio.to_thread(store.commit_turn, session_id, commit)
    )
    return cancelled


async def _save_checkpoint(
    store: TrajectoryStore,
    session_id: str,
    checkpoint: TurnCheckpoint,
) -> None:
    """Persist one checkpoint before honoring caller cancellation."""
    await await_known_outcome(
        asyncio.to_thread(store.save_checkpoint, session_id, checkpoint)
    )


def _safe_error_detail(exc: BaseException, *, limit: int = 1_000) -> str:
    detail = " ".join(str(exc).split())
    redacted = redact_text_secrets(detail)
    if len(redacted) <= limit:
        return redacted
    return redacted[:limit] + f"... [truncated {len(redacted) - limit} chars]"


async def _record_driver_failure(
    *,
    store: TrajectoryStore | None,
    bus: EventBus,
    session_id: str,
    phase: str,
    exc: BaseException,
    message: str,
    turn_id: str | None,
    turn_index: int | None,
    checkpoint_id: str | None,
) -> None:
    detail = _safe_error_detail(exc)
    diagnostic = TrajectoryDiagnostic(
        id=uuid4().hex,
        session_id=session_id,
        timestamp=time.time(),
        level="error",
        source="driver",
        phase=phase,
        message=message,
        error_type=type(exc).__name__,
        error_detail=detail,
        turn_id=turn_id,
        turn_index=turn_index,
        checkpoint_id=checkpoint_id,
    )
    if store is not None:
        try:
            await await_known_outcome(
                asyncio.to_thread(store.append_diagnostic, diagnostic)
            )
        except Exception as diagnostic_error:
            logger.warning(
                "driver diagnostic persistence failed: {}",
                diagnostic_error,
            )
    try:
        await bus.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level=diagnostic.level,
                source=diagnostic.source,
                message=diagnostic.message,
                phase=diagnostic.phase,
                error_type=diagnostic.error_type,
                error_detail=diagnostic.error_detail,
                turn_id=diagnostic.turn_id,
                turn_index=diagnostic.turn_index,
                checkpoint_id=diagnostic.checkpoint_id,
            ),
        )
    except Exception:
        logger.debug("diagnostic emit failed after turn abandon; non-fatal")


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
    services: ServiceRegistry,
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
) -> _RollbackResult:
    errors: list[BaseException] = []
    caller_cancelled = False

    # EffectScope is the outer world transaction: it snapshots the state after
    # ResourceTxn.apply(). Restore that world first, then let the resource
    # participant validate/remove its own journal. Running both concurrently
    # races over the same files for local and sandbox implementations.
    if effect_scope is not None and effect_txn is not None:
        try:
            _, cancelled = await settle_known_outcome(
                _abandon_effect_turn(effect_scope, effect_txn, bus=bus)
            )
            caller_cancelled = caller_cancelled or cancelled
        except BaseException as exc:
            errors.append(exc)
    if abandon_resource:
        try:
            _, cancelled = await settle_known_outcome(
                _abandon_resource_txn(resource_txn, bus=bus)
            )
            caller_cancelled = caller_cancelled or cancelled
        except BaseException as exc:
            errors.append(exc)
    return _RollbackResult(
        errors=tuple(errors),
        caller_cancelled=caller_cancelled,
    )


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


def _clear_resource_txn(services: ServiceRegistry) -> None:
    services.unregister(RESOURCE_TXN_SERVICE)


def _system_prompt_ref(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Main driver ------------------------------------------------------------


@dataclass(slots=True)
class DriverConfig:
    """Stable dependencies and policy inputs for one session driver."""

    trajectory: Trajectory
    triggers: TriggerQueue
    bus: EventBus
    stream_fn: StreamFn
    model: Model
    session_id: str
    root_session_id: str
    services: ServiceRegistry
    interrupt: ResettableCancelSource
    shutdown: ResettableCancelSource
    tool_orchestrator: ToolOrchestrator
    tools: list[Tool] = field(default_factory=list)
    store: TrajectoryStore | None = None
    parent_session_id: str | None = None
    permission_audience: PermissionAudience = "user"
    system: str | None = None
    context_policies: list[ContextPolicy] | None = None
    prompt_cache_adapter: ProviderPromptCacheAdapter | None = None
    trigger_renderers: dict[str, TriggerRenderer] | None = None
    cancel_signal: CancelSignal | None = None
    effect_scope: EffectScope | None = None
    resource_writer: ResourceWriter | None = None
    tool_executor: ToolExecutor | None = None
    permission_policy: PermissionPolicy | None = None
    max_turns: int | None = None
    thinking: ThinkingLevel = "off"
    max_tool_calls: int | None = None
    tool_allowlist: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _PromptRun:
    """One accepted external trigger and its internal continuation state."""

    run_id: str
    run_step: int
    envelope: TriggerEnvelope
    system_prompt: str | None

    def continue_with(self, *, system_prompt: str | None) -> _PromptRun:
        return replace(
            self,
            run_step=self.run_step + 1,
            envelope=TriggerEnvelope(
                trigger=ContinueTrigger(),
                metadata=TriggerMetadata(
                    priority="now",
                    origin="runtime",
                    mode="continue",
                ),
            ),
            system_prompt=system_prompt,
        )


async def drive(config: DriverConfig) -> None:
    """Persistent driver loop.

    Processes one external trigger as a PromptRun. Each provider response and
    its complete tool-result set becomes a separate Turn. Exits when:
    - shutdown is set
    - trigger queue is closed (QueueClosed)
    - max_turns committed turns reached
    - a session-terminal cause fires (ToolTerminated, BudgetExhausted)
    """

    trajectory = config.trajectory
    triggers = config.triggers
    bus = config.bus
    _interrupt = config.interrupt
    _shutdown = config.shutdown
    policies = config.context_policies or []
    context_projection = config.services.get(
        CONTEXT_PROJECTION_SERVICE,
        cast(type[ContextProjection], ContextProjection),
    )
    prompt_cache_adapter = config.prompt_cache_adapter
    if prompt_cache_adapter is None:
        prompt_cache_adapter = config.services.get(
            PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE,
            cast(type[ProviderPromptCacheAdapter], ProviderPromptCacheAdapter),
        )
    interruption_policy = config.services.get(
        INTERRUPTION_MESSAGE_POLICY_SERVICE,
        cast(type[InterruptionMessagePolicy], InterruptionMessagePolicy),
    )
    turns_run = 0
    tool_calls_run = 0
    _last_system_prompt_ref: str | None = None
    for _existing_turn in reversed(trajectory.turns):
        _existing_sp = _existing_turn.meta.system_prompt
        if _existing_sp is not None:
            _last_system_prompt_ref = _system_prompt_ref(_existing_sp)
            break

    prompt_run: _PromptRun | None = None
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

        if prompt_run is None:
            try:
                external_envelope = await triggers.wait_envelope()
            except QueueClosed:
                triggers.terminate(TriggerTerminated("trigger queue closed"))
                await bus.emit(RunEndEvent.CHANNEL, RunEndEvent())
                return
            _interrupt.clear()
            prompt_run = _PromptRun(
                run_id=uuid4().hex,
                run_step=0,
                envelope=external_envelope,
                system_prompt=config.system,
            )
        envelope = prompt_run.envelope
        trigger = envelope.trigger

        effect_txn: EffectTxn | None = None
        resource_txn: ResourceTxn | None = None
        resource_txn_committed = False
        durable_turn_committed = False
        turn_published = False
        cancelled_during_rollback = False
        phase = "turn_begin"
        active_turn_id: str | None = None
        active_turn_index: int | None = None
        active_checkpoint_id: str | None = None
        try:
            execution = trajectory.begin(
                trigger,
                run_id=prompt_run.run_id,
                run_step=prompt_run.run_step,
            )
            active_turn_id = execution.id
            active_turn_index = execution.index
            await bus.emit(
                TurnBeginEvent.CHANNEL,
                TurnBeginEvent(
                    index=execution.index,
                    turn_index=execution.index,
                    turn_id=execution.id,
                    run_id=execution.run_id,
                    run_step=execution.run_step,
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
                phase = "checkpoint_save"
                await checkpoint_writer(
                    execution.checkpoint(
                        TurnMeta(),
                        trigger_metadata=envelope.metadata,
                    )
                )
                active_checkpoint_id = execution.id

            phase = "effect_begin"
            effect_txn, cancelled_during_effect_begin = await settle_known_outcome(
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
            phase = "resource_begin"
            resource_txn, cancelled_during_resource_begin = await settle_known_outcome(
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
            phase = "reaction"
            reaction_result = await react(
                ReactionRequest(
                    execution=execution,
                    trigger=trigger,
                    trigger_metadata=envelope.metadata,
                    config=replace(
                        config,
                        context_policies=policies,
                        interrupt=_interrupt,
                        shutdown=_shutdown,
                        system=prompt_run.system_prompt,
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
            outcome = reaction_result.outcome
            meta = reaction_result.meta
            tool_calls_used = reaction_result.tool_calls_used

            phase = "checkpoint_save"
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
                phase = "rollback"
                rollback = await _rollback_unpublished_turn(
                    resource_txn=resource_txn,
                    abandon_resource=True,
                    effect_scope=config.effect_scope,
                    effect_txn=effect_txn,
                    bus=bus,
                )
                cancelled_during_rollback = rollback.caller_cancelled
                resource_txn = None
                effect_txn = None
                _clear_resource_txn(config.services)
                if rollback.errors:
                    raise BaseExceptionGroup(
                        "provider request and turn rollback failed",
                        [
                            RuntimeError(
                                f"{outcome.cause.error_type}: {outcome.cause.detail}"
                            ),
                            *rollback.errors,
                        ],
                    )
            else:
                phase = "resource_prepare"
                (
                    resource_mutations,
                    cancelled_during_resource_prepare,
                ) = await settle_known_outcome(_prepare_resource_txn(resource_txn))
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
                phase = "resource_apply"
                _, cancelled_during_resource_apply = await settle_known_outcome(
                    _apply_resource_txn(resource_txn)
                )
                if cancelled_during_resource_apply:
                    raise asyncio.CancelledError
                phase = "effect_prepare"
                _, cancelled_during_effect_prepare = await settle_known_outcome(
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
                phase = "trajectory_prepare"
                (
                    node_append_position,
                    cancelled_during_node_position,
                ) = await settle_known_outcome(
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
                sys_text = turn.meta.system_prompt
                sys_ref = _system_prompt_ref(sys_text) if sys_text else None
                sys_node: TrajectoryNode | None = None
                if sys_ref is not None and sys_ref != _last_system_prompt_ref:
                    sys_node = TrajectoryNode(
                        id=(
                            f"session:{config.session_id}:turn:{turn.id}:system_prompt"
                        ),
                        session_id=config.session_id,
                        seq=node_append_position.start_seq,
                        kind="system_prompt",
                        role="control",
                        root_session_id=config.root_session_id,
                        parent_session_id=config.parent_session_id,
                        branch_id=node_append_position.branch_id,
                        head_id=node_append_position.head_id,
                        parent_id=node_append_position.parent_node_id,
                        logical_parent_id=node_append_position.logical_parent_id,
                        content_ref=sys_ref,
                        payload={"text": sys_text},
                        timestamp=turn.timestamp,
                    )
                    _last_system_prompt_ref = sys_ref
                parent_node_id: str | None
                logical_parent_id: str | None
                if sys_node is not None:
                    start_seq = node_append_position.start_seq + 1
                    parent_node_id = sys_node.id
                    logical_parent_id = None
                else:
                    start_seq = node_append_position.start_seq
                    parent_node_id = node_append_position.parent_node_id
                    logical_parent_id = node_append_position.logical_parent_id
                nodes = turn_to_nodes(
                    turn,
                    session_id=config.session_id,
                    start_seq=start_seq,
                    root_session_id=config.root_session_id,
                    parent_session_id=config.parent_session_id,
                    branch_id=node_append_position.branch_id,
                    head_id=node_append_position.head_id,
                    parent_node_id=parent_node_id,
                    logical_parent_id=logical_parent_id,
                    renderers=config.trigger_renderers,
                )
                if sys_node is not None:
                    nodes.insert(0, sys_node)
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
                phase = "trajectory_commit"
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
                phase = "resource_commit"
                (
                    _,
                    cancelled_during_resource_commit,
                ) = await settle_known_outcome(_commit_resource_txn(resource_txn))
                resource_txn_committed = resource_txn is not None
            except Exception:
                if durable_turn_committed:
                    trajectory.finalize_commit(turn)
                    turn_published = True
                raise
            trajectory.finalize_commit(turn)
            turn_published = True
            _clear_resource_txn(config.services)
            phase = "effect_commit"
            _, cancelled_during_effect_commit = await settle_known_outcome(
                _commit_effect_turn(config.effect_scope, effect_txn, turn, bus=bus)
            )

            if isinstance(outcome.cause, ProviderRequestFailed):
                triggers.fail(TriggerTerminated(outcome.cause))
                prompt_run = None
            elif isinstance(outcome.cause, PromptRunContinued):
                prompt_run = prompt_run.continue_with(
                    system_prompt=reaction_result.continuation_system_prompt,
                )
            else:
                triggers.complete(turn)
                prompt_run = None
            cancelled_during_event = False
            try:
                phase = "event_publish"
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
                    cancelled_during_rollback,
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
                rollback = await _rollback_unpublished_turn(
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
                    rollback = replace(
                        rollback,
                        errors=(*rollback.errors, cleanup_exc),
                    )
                if rollback.errors:
                    cancel_error = BaseExceptionGroup(
                        "turn cancellation and rollback failed",
                        [exc, *rollback.errors],
                    )
            _clear_resource_txn(config.services)
            triggers.terminate(cancel_error)
            if cancel_error is exc:
                raise
            raise cancel_error
        except Exception as exc:
            execution_error: BaseException = exc
            cancelled_while_rolling_back = False
            if not turn_published:
                rollback = await _rollback_unpublished_turn(
                    resource_txn=resource_txn,
                    abandon_resource=not resource_txn_committed,
                    effect_scope=config.effect_scope,
                    effect_txn=effect_txn,
                    bus=bus,
                )
                cancelled_while_rolling_back = rollback.caller_cancelled
                try:
                    trajectory.abandon()
                except BaseException as cleanup_exc:
                    logger.debug(
                        "trajectory abandon failed during rollback: {}",
                        cleanup_exc,
                    )
                    rollback = replace(
                        rollback,
                        errors=(*rollback.errors, cleanup_exc),
                    )
                if rollback.errors:
                    execution_error = BaseExceptionGroup(
                        "turn execution and rollback failed",
                        [exc, *rollback.errors],
                    )
            _clear_resource_txn(config.services)
            triggers.terminate(execution_error)
            diagnostic_message = (
                "driver stopped after a committed turn"
                if turn_published
                else "turn abandoned: trigger dropped due to internal error"
            )
            logger.exception("driver: {}", diagnostic_message)
            await _record_driver_failure(
                store=config.store,
                bus=bus,
                session_id=config.session_id,
                phase=phase,
                exc=exc,
                message=diagnostic_message,
                turn_id=active_turn_id,
                turn_index=active_turn_index,
                checkpoint_id=active_checkpoint_id,
            )
            if cancelled_while_rolling_back:
                raise asyncio.CancelledError
            return


__all__ = ["drive"]
