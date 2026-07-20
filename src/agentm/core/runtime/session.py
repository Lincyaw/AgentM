"""SDK session facade for child creation, fork, and durable resume."""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self, cast

from agentm.core.abi.cancel import (
    CancelSignal,
    CompositeCancelSignal,
)
from agentm.core.abi.codec import CodecRegistry, RawTrigger
from agentm.core.abi.events import (
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.lifecycle import (
    EnvironmentFork,
    EnvironmentRestoreError,
    EnvironmentRestoreStatus,
)
from agentm.core.abi.resource import (
    EnvironmentForkableResourceWriter,
    ResourceReader,
    ResourceRecoveryContext,
    ResourceStore,
    TransactionalResourceWriter,
)
from agentm.core.abi.roles import RESOLVED_SESSION_SPEC_SERVICE
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import (
    AgentSessionConfig,
    ChildCancellationMode,
    ResolvedSessionSpec,
)
from agentm.core.abi.store import (
    TrajectoryNodeQuery,
    TrajectoryStore,
)
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.tool import Tool
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    TrajectoryForkPoint,
    TrajectoryHead,
    TrajectoryNode,
    Turn,
    TurnRef,
)
from agentm.core.runtime.session_core import (
    SessionRuntimeConfig,
    _SessionComposition,
)
from agentm.core.runtime.session_meta import (
    context_from_session_meta,
    provider_identity_from_session_meta,
    validate_resume_identity,
    validate_resume_metadata,
)
from agentm.core.runtime.trajectory import Trajectory


def _rehydrate_turn_triggers(
    turns: Sequence[Turn],
    codec: CodecRegistry,
) -> list[Turn]:
    """Restore custom triggers after resume-time extensions register codecs."""

    restored: list[Turn] = []
    for turn in turns:
        trigger = turn.trigger
        if isinstance(trigger, RawTrigger):
            hydrated = codec.deserialize_trigger(dict(trigger.data))
            if isinstance(hydrated, RawTrigger):
                raise ValueError(
                    f"no TriggerCodec registered for persisted source "
                    f"{trigger.source!r}"
                )
            turn = replace(turn, trigger=hydrated)
        restored.append(turn)
    return restored


@dataclass(frozen=True, slots=True)
class _ResolvedForkAnchor:
    turn_ref: TurnRef
    node_id: str | None = None


def _fork_head(
    *,
    source: Session,
    target_session_id: str,
    target_parent_session_id: str,
    logical_parent_id: str,
) -> TrajectoryHead:
    return TrajectoryHead(
        session_id=target_session_id,
        head_id=DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
        root_session_id=source.ctx.root_session_id,
        parent_session_id=target_parent_session_id,
        logical_parent_id=logical_parent_id,
        status="active",
        updated_at=time.time(),
    )


async def _active_chain(
    *,
    source: Session,
    head_id: str = DEFAULT_TRAJECTORY_HEAD_ID,
    branch_id: str = DEFAULT_TRAJECTORY_BRANCH_ID,
) -> list[TrajectoryNode]:
    store = source.store
    if store is None:
        raise ValueError("node/head fork points require a trajectory store")
    head = await asyncio.to_thread(
        store.get_head,
        source.id,
        head_id=head_id,
        branch_id=branch_id,
    )
    if head is None:
        raise KeyError(head_id)
    leaf_node_id = head.node_id or head.logical_parent_id
    if leaf_node_id is None:
        return []
    return await asyncio.to_thread(
        store.load_chain,
        source.id,
        leaf_node_id,
        include_logical_parent=True,
    )


async def _resolve_fork_anchor(
    *,
    source: Session,
    at: TurnRef | TrajectoryForkPoint,
) -> _ResolvedForkAnchor:
    if isinstance(at, TrajectoryForkPoint) and at.session_id != source.id:
        raise ValueError(
            "trajectory fork point session_id must match the source session"
        )
    turn_ref = at.turn_ref if isinstance(at, TrajectoryForkPoint) else at
    if turn_ref is not None:
        prefix = source.trajectory.prefix(turn_ref)
        turn = prefix.turns[-1]
        if source.store is None:
            return _ResolvedForkAnchor(turn_ref=turn_ref)
        chain = await _active_chain(
            source=source,
            branch_id=(
                at.branch_id
                if isinstance(at, TrajectoryForkPoint)
                else DEFAULT_TRAJECTORY_BRANCH_ID
            ),
        )
        matching_nodes = [
            node
            for node in chain
            if node.turn_id == turn.id and node.turn_index == turn.index
        ]
        if not matching_nodes:
            raise RuntimeError(
                f"trajectory store has no committed boundary for fork turn "
                f"{turn.id}"
            )
        return _ResolvedForkAnchor(
            turn_ref=turn_ref,
            node_id=matching_nodes[-1].id,
        )

    if not isinstance(at, TrajectoryForkPoint):
        raise TypeError("fork point must be a TurnRef or TrajectoryForkPoint")
    head_id = at.head_id or DEFAULT_TRAJECTORY_HEAD_ID
    chain = await _active_chain(
        source=source,
        head_id=head_id,
        branch_id=at.branch_id,
    )
    node_id = at.node_id
    if node_id is None:
        if not chain:
            raise ValueError("cannot fork from an empty trajectory head")
        node_id = chain[-1].id
    selected = next((node for node in chain if node.id == node_id), None)
    if selected is None:
        raise ValueError(
            "trajectory fork node must be reachable from the selected "
            "source head"
        )
    if selected.turn_index is None or selected.turn_id is None:
        raise ValueError(
            "trajectory fork node has no committed turn boundary"
        )
    prefix = source.trajectory.prefix(selected.turn_index)
    turn = prefix.turns[-1]
    if turn.id != selected.turn_id:
        raise ValueError(
            "trajectory fork node does not match source committed history"
        )
    if selected.kind == "message":
        if source.store is None:
            raise ValueError("message node fork points require a trajectory store")
        turn_messages = await asyncio.to_thread(
            source.store.query_nodes,
            TrajectoryNodeQuery(
                session_id=selected.session_id,
                turn_index=selected.turn_index,
                kinds=("message",),
                sort="desc",
                limit=1,
            ),
        )
        if not turn_messages or turn_messages[0].id != selected.id:
            raise ValueError(
                "message node fork points must target the final message of a "
                "committed turn; context projection cannot restore mid-turn "
                "external effects"
            )
    return _ResolvedForkAnchor(
        turn_ref=selected.turn_index,
        node_id=selected.id,
    )


class Session(_SessionComposition):
    """Top-level SDK session with child, fork, and resume operations."""

    async def spawn(
        self,
        *,
        purpose: str = "subagent",
        tools: list[Tool] | None = None,
        system: str | None = None,
        model: Model | None = None,
        stream_fn: StreamFn | None = None,
        scenario: str | None = None,
        cwd: str | None = None,
        max_turns: int | None = None,
        extra_services: ServiceRegistry | None = None,
        cancel_signal: CancelSignal | None = None,
        parent_cancellation: ChildCancellationMode = "inherit",
    ) -> Self:
        """Spawn a child by reinstalling the parent's atom composition."""

        child_ctx = self.ctx.child(
            session_id=uuid.uuid4().hex[:16],
            purpose=purpose,
            cwd=cwd,
            scenario=scenario,
        )
        child_services = ServiceRegistry()
        child_services.inherit_from(self.services)
        if extra_services is not None:
            child_services.update_from(extra_services)

        if parent_cancellation not in {"inherit", "independent"}:
            raise ValueError("parent_cancellation must be 'inherit' or 'independent'")
        inherited_cancel = (
            CompositeCancelSignal(
                self._interrupt,
                self._shutdown,
                self._parent_cancel_signal,
            )
            if parent_cancellation == "inherit"
            else None
        )
        child_cancel_signal = (
            CompositeCancelSignal(inherited_cancel, cancel_signal)
            if inherited_cancel is not None and cancel_signal is not None
            else cancel_signal
            if cancel_signal is not None
            else inherited_cancel
        )

        direct_provider_override = stream_fn is not None or model is not None
        if scenario is not None and direct_provider_override:
            raise ValueError(
                "spawn cannot combine a scenario change with direct stream/model "
                "overrides; use spawn_child_session with an explicit provider"
            )
        extension_specs = (
            None
            if scenario is not None
            else self._composition_extensions(
                include_provider_atoms=not direct_provider_override,
            )
        )
        from agentm.core.runtime.session_factory import (
            SessionBuildConfig,
            create_session,
        )

        source_tool_allowlist = self._tool_allowlist()
        child = await create_session(
            SessionBuildConfig(
                scenario=child_ctx.scenario,
                extensions=extension_specs,
                session_context=child_ctx,
                services=child_services,
                store=self.store,
                graph=self.graph,
                stream_fn=stream_fn or self._stream_fn,
                model=model or self._model,
                tools=(
                    list(tools) if tools is not None else list(self._external_tools())
                ),
                system=system if system is not None else self.system,
                context_policies=[
                    copy.copy(policy) for policy in self._external_context_policies()
                ],
                trigger_renderers=self._external_trigger_renderers(),
                codec=self._composition_codec(),
                max_turns=self._max_turns if max_turns is None else max_turns,
                max_tool_calls=self._max_tool_calls,
                tool_allowlist=(
                    list(source_tool_allowlist)
                    if source_tool_allowlist is not None
                    else None
                ),
                thinking=self._thinking,
                cancel_signal=child_cancel_signal,
            ),
            session_type=type(self),
        )

        await self._register_child(child, purpose=purpose)
        return cast(Self, child)

    async def spawn_child_session(
        self,
        config: AgentSessionConfig,
    ) -> Self:
        """Spawn a fully-constructed child via the session factory."""

        from agentm.core.runtime.session_factory import create_child_session

        child = await create_child_session(parent=self, config=config)
        await self._register_child(child, purpose=config.purpose)
        return cast(Self, child)

    async def _register_child(self, child: Session, *, purpose: str) -> None:
        async def _on_child_shutdown(_: SessionShutdownEvent) -> None:
            await self.bus.emit(
                ChildSessionEndEvent.CHANNEL,
                ChildSessionEndEvent(
                    child_session_id=child.id,
                    parent_session_id=self.id,
                    final_message_count=len(child.get_messages()),
                    error=child._driver_error,
                ),
            )

        child.bus.on(SessionShutdownEvent.CHANNEL, _on_child_shutdown)

        if child.store is not None and not await asyncio.to_thread(
            child.store.session_exists,
            child.id,
        ):
            raise RuntimeError(
                "child session factory returned before persisting session "
                f"metadata for {child.id}"
            )

        if child.graph is not None:
            child.graph.register(
                child.id,
                parent_id=self.id,
                purpose=purpose,
                edge_kind="spawned",
            )

        await self.bus.emit(
            ChildSessionStartEvent.CHANNEL,
            ChildSessionStartEvent(
                child_session_id=child.id,
                parent_session_id=self.id,
                purpose=purpose,
            ),
        )

    @classmethod
    async def fork(
        cls,
        source: Session,
        at: TurnRef | TrajectoryForkPoint,
        *,
        purpose: str = "fork",
    ) -> Self:
        anchor = await _resolve_fork_anchor(source=source, at=at)
        turn_ref = anchor.turn_ref
        prefix = source.trajectory.prefix(turn_ref)
        child_ctx = source.ctx.child(
            session_id=uuid.uuid4().hex[:16],
            purpose=purpose,
        )
        provider_identity = source.provider_session_identity()
        initial_head = (
            _fork_head(
                source=source,
                target_session_id=child_ctx.session_id,
                target_parent_session_id=source.id,
                logical_parent_id=anchor.node_id,
            )
            if anchor.node_id is not None
            else None
        )

        source_resource_writer = source.get_resource_writer()
        if (
            source.get_effect_scope() is not None
            and source_resource_writer is not None
            and not isinstance(
                source_resource_writer,
                EnvironmentForkableResourceWriter,
            )
        ):
            raise TypeError(
                "forking an isolated environment requires its ResourceWriter "
                "to implement EnvironmentForkableResourceWriter"
            )

        environment_fork: EnvironmentFork | None = None
        effect_scope = source.get_effect_scope()
        if effect_scope is not None:
            environment_fork = await effect_scope.fork_at(
                turn_ref,
                source_session_id=source.id,
                child_session_id=child_ctx.session_id,
            )
            if not isinstance(environment_fork, EnvironmentFork):
                raise TypeError("EffectScope.fork_at() must return an EnvironmentFork")
            child_ctx = replace(child_ctx, cwd=environment_fork.cwd)

        child_services = ServiceRegistry()
        child_services.inherit_from(source.services)
        resolved_spec = source._resolved_session_spec()
        if resolved_spec is not None:
            child_services.register(
                RESOLVED_SESSION_SPEC_SERVICE,
                resolved_spec,
                ResolvedSessionSpec,
                scope="session",
            )

        child_resource_writer = source_resource_writer
        child_resource_reader = source.get_resource_reader()
        child_resource_store = source.get_resource_store()
        if environment_fork is not None and source_resource_writer is not None:
            forkable_writer = cast(
                EnvironmentForkableResourceWriter,
                source_resource_writer,
            )
            child_resource_writer = await forkable_writer.fork_for_environment(
                workspace_root=environment_fork.cwd,
                child_session_id=child_ctx.session_id,
            )
            if child_resource_store is source_resource_writer:
                if not isinstance(child_resource_writer, ResourceStore):
                    raise TypeError(
                        "forked ResourceWriter must preserve ResourceStore when "
                        "the source service implemented both contracts"
                    )
                child_resource_store = child_resource_writer
            if child_resource_reader is source_resource_writer:
                if not isinstance(child_resource_writer, ResourceReader):
                    raise TypeError(
                        "forked ResourceWriter must preserve ResourceReader when "
                        "the source service implemented both contracts"
                    )
                child_resource_reader = child_resource_writer

        from agentm.core.runtime.session_factory import (
            SessionBuildConfig,
            create_session,
        )

        source_tool_allowlist = source._tool_allowlist()
        forked = await create_session(
            SessionBuildConfig(
                extensions=source._composition_extensions(include_provider_atoms=True),
                stream_fn=source._stream_fn,
                model=source._model,
                system=source.system,
                cwd=child_ctx.cwd,
                purpose=purpose,
                store=source.store,
                graph=source.graph,
                session_context=child_ctx,
                initial_turns=list(prefix.turns),
                initial_head=initial_head,
                fork_point=turn_ref,
                tools=list(source._external_tools()),
                context_policies=[
                    copy.copy(policy) for policy in source._external_context_policies()
                ],
                trigger_renderers=source._external_trigger_renderers(),
                codec=source._composition_codec(),
                provider_identity=provider_identity,
                resource_reader=child_resource_reader,
                resource_store=child_resource_store,
                resource_writer=child_resource_writer,
                tool_executor=source.get_tool_executor(),
                tool_orchestrator=source.get_tool_orchestrator(),
                permission_policy=source.get_permission_policy(),
                effect_scope=(
                    environment_fork.effect_scope
                    if environment_fork is not None
                    else None
                ),
                environment_operations=(
                    environment_fork.operations
                    if environment_fork is not None
                    else None
                ),
                environment_restore_failure_handler=(
                    source._environment_restore_failure_handler()
                ),
                services=child_services,
                resolved_spec=resolved_spec,
                max_turns=source._max_turns,
                max_tool_calls=source._max_tool_calls,
                tool_allowlist=(
                    list(source_tool_allowlist)
                    if source_tool_allowlist is not None
                    else None
                ),
                thinking=source._thinking,
            ),
            session_type=cls,
        )

        if forked.graph is not None:
            forked.graph.register(
                forked.id,
                parent_id=source.id,
                fork_point=turn_ref,
                purpose=purpose,
                edge_kind="forked",
            )

        return cast(Self, forked)

    @classmethod
    async def resume(
        cls,
        session_id: str,
        store: TrajectoryStore,
        config: AgentSessionConfig,
    ) -> Self:
        meta, turns = await asyncio.to_thread(store.load, session_id)
        validate_resume_metadata(meta, has_committed_turns=bool(turns))
        ctx = context_from_session_meta(session_id, meta)
        provider_identity = provider_identity_from_session_meta(meta)
        from agentm.core.runtime.session_factory import create_from_config

        resume_scenario = config.scenario
        if resume_scenario is None and config.extensions is None:
            resume_scenario = ctx.scenario
        resume_config = replace(
            config,
            cwd=ctx.cwd,
            scenario=resume_scenario,
            purpose=ctx.purpose,
            trajectory_store=store,
            session_id=ctx.session_id,
            root_session_id=ctx.root_session_id,
            parent_session_id=ctx.parent_session_id,
            initial_turns=list(turns),
        )
        session = await create_from_config(
            resume_config,
            restored_context=ctx,
            restored_provider_identity=provider_identity,
            session_type=cls,
        )
        restored_turns = _rehydrate_turn_triggers(turns, session.codec)
        if restored_turns != turns:
            session.trajectory = Trajectory(restored_turns)
            turns = restored_turns
        validate_resume_identity(
            meta,
            resolved_spec=session._resolved_session_spec(),
            active_set=session._active_set_fingerprint(),
        )
        resource_writer = session.get_resource_writer()
        if isinstance(resource_writer, TransactionalResourceWriter):
            transaction_ids = tuple(
                dict.fromkeys(
                    mutation.transaction_id
                    for turn in turns
                    for mutation in turn.meta.resource_mutations
                    if mutation.transaction_id is not None
                )
            )
            await resource_writer.recover(
                ResourceRecoveryContext(
                    session_id=session.id,
                    committed_transaction_ids=transaction_ids,
                )
            )
        effect_scope = session.get_effect_scope()
        if effect_scope is not None:
            try:
                await effect_scope.restore(
                    session_id=session.id,
                    turns=tuple(turns),
                )
                session._record_environment_restore_status(
                    EnvironmentRestoreStatus(
                        session_id=session.id,
                        restored=True,
                        state="restored",
                    )
                )
            except Exception as exc:
                handler = session._environment_restore_failure_handler()
                restore_status = EnvironmentRestoreStatus(
                    session_id=session.id,
                    restored=False,
                    state="degraded_readonly",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if handler is None:
                    raise EnvironmentRestoreError(
                        f"environment restore failed for session {session.id}"
                    ) from exc
                try:
                    await handler.activate_degraded_readonly(restore_status)
                except Exception as handler_exc:
                    raise ExceptionGroup(
                        f"environment restore and degraded-mode activation "
                        f"failed for session {session.id}",
                        [exc, handler_exc],
                    ) from handler_exc
                session._record_environment_restore_status(restore_status)

        return cast(Self, session)


__all__ = ["Session", "SessionRuntimeConfig"]
