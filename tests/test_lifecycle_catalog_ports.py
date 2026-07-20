"""Focused coverage for lifecycle and catalog SDK ports."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
import fcntl
from pathlib import Path
import shlex
import threading
from typing import Any, Literal

import pytest

from agentm.core.abi.cancel import CancelSignal, EventCancelSource
from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    CatalogActiveSetInput,
)
from agentm.core.abi.compaction import (
    ContextBudget,
    ContextProjection,
    ProjectionInput,
    ProjectionReport,
)
from agentm.core.abi.events import ChildSessionEndEvent, ChildSessionStartEvent
from agentm.core.abi.lifecycle import EffectTxn, EnvironmentFork
from agentm.core.abi.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCallBlock,
    UserMessage,
)
from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.provider import ProviderConfig
from agentm.core.abi.resource import (
    PathClass,
    ResourceMutation,
    ResourceRecoveryContext,
    ResourceRef,
    ResourceTxnContext,
    WriteResult,
)
from agentm.core.abi.roles import (
    ACTIVE_SET_FINGERPRINT_SERVICE,
    BASH_OPERATIONS_SERVICE,
    CONTEXT_PROJECTION_SERVICE,
    ENVIRONMENT_OPERATIONS_SERVICE,
    RESOLVED_SESSION_SPEC_SERVICE,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import AgentSessionConfig, ResolvedSessionSpec
from agentm.core.abi.store import TrajectoryCommit
from agentm.core.abi.stream import MessageEnd, Model, TextDelta
from agentm.core.abi.tool import ToolOutcome, ToolResult, ToolTerminate
from agentm.core.abi.tool_executor import (
    EnvironmentExecutableTool,
    ToolExecutionCapabilities,
    ToolExecutionRequest,
    ToolExecutionRequirements,
)
from agentm.core.abi.tool_orchestration import (
    ToolOrchestrationRequest,
    ToolWorkItem,
)
from agentm.core.abi.trajectory import (
    Turn,
    TurnRef,
)
from agentm.core.runtime.session import Session, SessionRuntimeConfig
from agentm.core.runtime.session_factory import (
    SessionBuildConfig,
    create_from_config,
    create_session,
)
from agentm.storage.trajectory.memory import InMemoryTrajectoryStore
from agentm.core.runtime.tool_orchestration import DefaultToolOrchestrator
from agentm.config import DefaultSessionSpecResolver
from agentm.environments import LocalBashOperations, LocalEnvironmentOperations
from agentm.execution import ProcessToolExecutor, SandboxToolExecutor
from agentm.storage.resources import LocalResourceStore


def _model() -> Model:
    return Model(
        id="mock-model",
        provider="mock",
        context_window=128000,
        max_output_tokens=4096,
    )


class _StaticStream:
    def __init__(self, text: str = "ok") -> None:
        self._text = text

    async def __call__(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any,
        system: Any = None,
        signal: Any = None,
        thinking: Any = "off",
    ) -> AsyncIterator[TextDelta | MessageEnd]:
        del messages, model, tools, system, signal, thinking
        response = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=self._text)],
            timestamp=0.0,
        )
        yield TextDelta(text=self._text)
        yield MessageEnd(message=response)


class _QueuedStream:
    def __init__(self, *responses: AssistantMessage) -> None:
        self._responses = list(responses)

    async def __call__(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any,
        system: Any = None,
        signal: Any = None,
        thinking: Any = "off",
    ) -> AsyncIterator[MessageEnd]:
        del messages, model, tools, system, signal, thinking
        if not self._responses:
            raise RuntimeError("no queued response")
        yield MessageEnd(message=self._responses.pop(0))


class _RecordingStream:
    def __init__(self, text: str = "ok") -> None:
        self._text = text
        self.calls: list[list[Any]] = []

    async def __call__(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any,
        system: Any = None,
        signal: Any = None,
        thinking: Any = "off",
    ) -> AsyncIterator[MessageEnd]:
        del model, tools, system, signal, thinking
        self.calls.append(list(messages))
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=self._text)],
                timestamp=0.0,
            )
        )


class _FailingStream:
    async def __call__(
        self,
        *,
        messages: Any,
        model: Any,
        tools: Any,
        system: Any = None,
        signal: Any = None,
        thinking: Any = "off",
    ) -> AsyncIterator[MessageEnd]:
        del messages, model, tools, system, signal, thinking
        raise RuntimeError("stream failed")
        yield  # pragma: no cover


class _RecordingProjection:
    def __init__(self) -> None:
        self.calls: list[tuple[ProjectionInput, ContextBudget]] = []

    @property
    def source(self) -> Literal["turns"]:
        return "turns"

    def project(
        self,
        projection_input: ProjectionInput,
        budget: ContextBudget,
    ) -> Sequence[UserMessage]:
        self.calls.append((projection_input, budget))
        return [
            UserMessage(
                role="user",
                content=[TextContent(type="text", text="projected history")],
                timestamp=0.0,
            )
        ]

    def explain(self) -> ProjectionReport:
        return ProjectionReport()


async def _wait_turn(session: Session, expected: int = 1) -> None:
    for _ in range(200):
        if len(session.trajectory) >= expected:
            return
        if session._driver_error:
            raise AssertionError(session._driver_error)
        await asyncio.sleep(0.01)
    raise TimeoutError("session did not commit a turn")


class _RecordingEnvironmentForkLease:
    def __init__(self) -> None:
        self.committed = False
        self.abandoned = False

    async def commit(self) -> None:
        self.committed = True

    async def abandon(self) -> None:
        self.abandoned = True


class _DelayedChildVisibilityStore(InMemoryTrajectoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.parent_session_id: str | None = None
        self.visibility_started = threading.Event()
        self.visibility_release = threading.Event()
        self._child_visibility_checks = 0

    def session_exists(self, session_id: str) -> bool:
        if (
            self.parent_session_id is not None
            and session_id != self.parent_session_id
        ):
            self._child_visibility_checks += 1
            if self._child_visibility_checks == 2:
                self.visibility_started.set()
                if not self.visibility_release.wait(timeout=2.0):
                    raise TimeoutError("test did not release child visibility check")
        return super().session_exists(session_id)


class _RecordingEffectScope:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | int]] = []
        self.children: list[_RecordingEffectScope] = []
        self.fork_leases: list[_RecordingEnvironmentForkLease] = []
        self.fork_started = asyncio.Event()
        self.fork_release = asyncio.Event()
        self.block_fork = False

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        turn_index: int,
    ) -> EffectTxn:
        self.events.append(("begin", session_id, turn_index))
        return EffectTxn(
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
            token=turn_id,
        )

    async def commit_turn(self, txn: EffectTxn, turn: Turn) -> None:
        self.events.append(("commit", txn.session_id, turn.index))

    async def prepare_turn(self, txn: EffectTxn, turn: Turn) -> None:
        self.events.append(("prepare", txn.session_id, turn.index))

    async def abandon_turn(self, txn: EffectTxn) -> None:
        self.events.append(("abandon", txn.session_id, txn.turn_index))

    async def fork_at(
        self,
        ref: TurnRef,
        *,
        source_session_id: str,
        child_session_id: str,
    ) -> EnvironmentFork:
        if self.block_fork:
            self.fork_started.set()
            await self.fork_release.wait()
        child = _RecordingEffectScope()
        lease = _RecordingEnvironmentForkLease()
        self.children.append(child)
        self.fork_leases.append(lease)
        self.events.append(("fork", source_session_id, child_session_id))
        del ref
        return EnvironmentFork(effect_scope=child, cwd="", lease=lease)

    async def restore(self, *, session_id: str, turns: Sequence[Turn]) -> None:
        self.events.append(("restore", session_id, len(turns)))


class _RecordingCatalog:
    def __init__(self) -> None:
        self.active_sets: list[CatalogActiveSetInput] = []

    async def record_active_set(
        self,
        active_set: CatalogActiveSetInput,
    ) -> ActiveSetFingerprint:
        captured = tuple(active_set.atoms)
        self.active_sets.append(active_set)
        return ActiveSetFingerprint(
            algorithm="test",
            digest=f"test:{len(captured)}",
            atoms=captured,
        )

    async def get_active_set(self, session_id: str) -> ActiveSetFingerprint | None:
        for active_set in self.active_sets:
            if active_set.session_id == session_id:
                return ActiveSetFingerprint(
                    algorithm="test",
                    digest=f"test:{len(active_set.atoms)}",
                    atoms=active_set.atoms,
                )
        return None


class _RequirementsTool:
    name = "needs_executor"
    description = "tool with typed execution requirements"
    parameters: dict[str, object] = {}
    execution_requirements = ToolExecutionRequirements(
        isolation="process",
        killable=True,
        filesystem="read",
    )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: Any = None,
    ) -> ToolResult | ToolOutcome:
        del args, signal
        raise AssertionError("custom tool executor should run this tool")


class _ProcessEntrypointTool:
    name = "process_tool"
    description = "importable process tool"
    parameters: dict[str, object] = {}

    def __init__(self, entrypoint: str) -> None:
        self.metadata = {"process_entrypoint": entrypoint}

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        del args, signal
        raise AssertionError("process executor must not call Tool.execute")


class _EnvironmentAwareTool(EnvironmentExecutableTool):
    name = "environment_tool"
    description = "typed environment tool"
    parameters: dict[str, object] = {}

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        del args, signal
        raise AssertionError("sandbox executor must use execute_in_environment")

    async def execute_in_environment(
        self,
        args: Mapping[str, object],
        *,
        environment: EnvironmentOperations,
        cwd: str | None = None,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        del args, signal
        self.calls.append((environment.ref.id, cwd))
        return ToolResult(content=[TextContent(type="text", text="sandboxed")])


class _RecordingToolExecutor:
    def __init__(self) -> None:
        self.requests: list[ToolExecutionRequest] = []

    def capabilities(self) -> ToolExecutionCapabilities:
        return ToolExecutionCapabilities(
            isolation=("none", "process"),
            filesystem=("none", "read"),
            killable=True,
        )

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: Any = None,
    ) -> ToolResult | ToolOutcome:
        del signal
        self.requests.append(request)
        return ToolResult(content=[TextContent(type="text", text="executed")])


class _Resolver:
    def resolve(self, request: AgentSessionConfig) -> ResolvedSessionSpec:
        return ResolvedSessionSpec(
            scenario="resolved",
            extensions=(),
            atom_config={"atom": {"value": 1}},
            provider=request.provider,
            provenance={"source": "test"},
        )


class _ProviderResolver:
    def __init__(self, selected: str) -> None:
        self.selected = selected

    def resolve_provider(
        self,
        providers: Mapping[str, ProviderConfig],
    ) -> str | None:
        if self.selected in providers:
            return self.selected
        return next(iter(providers), None)


class _NoopBatch:
    async def __aenter__(self) -> "_NoopBatch":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info

    async def write(self, path: str, content: bytes) -> None:
        del path, content

    async def replace(self, path: str, old: bytes, new: bytes) -> None:
        del path, old, new

    async def delete(self, path: str) -> None:
        del path


class _RecordingResourceTxn:
    def __init__(self, context: ResourceTxnContext) -> None:
        self.context = context
        self.writes: list[tuple[ResourceRef, bytes, str]] = []
        self.replacements: list[tuple[ResourceRef, bytes, bytes, str]] = []
        self.deletes: list[tuple[ResourceRef, str]] = []
        self.mutations: list[ResourceMutation] = []
        self.applied = False
        self.committed = False
        self.abandoned = False

    async def read(self, ref: ResourceRef) -> bytes | None:
        for pending_ref, content, _rationale in reversed(self.writes):
            if pending_ref == ref:
                return content
        return None

    async def create(
        self,
        ref: ResourceRef,
        content: bytes,
        *,
        rationale: str,
        author: str = "agent",
    ) -> ResourceMutation:
        del author
        self.writes.append((ref, content, rationale))
        mutation = ResourceMutation(ref=ref, op="create", after_version="txn-write")
        self.mutations.append(mutation)
        return mutation

    async def replace(
        self,
        ref: ResourceRef,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: str = "agent",
    ) -> ResourceMutation:
        del author
        self.replacements.append((ref, old, new, rationale))
        mutation = ResourceMutation(
            ref=ref,
            op="replace",
            before_version="txn-before",
            after_version="txn-after",
        )
        self.mutations.append(mutation)
        return mutation

    async def delete(
        self,
        ref: ResourceRef,
        *,
        rationale: str,
        author: str = "agent",
    ) -> ResourceMutation:
        del author
        self.deletes.append((ref, rationale))
        mutation = ResourceMutation(ref=ref, op="delete", before_version="txn-before")
        self.mutations.append(mutation)
        return mutation

    async def prepare(self) -> list[ResourceMutation]:
        return list(self.mutations)

    async def apply(self) -> None:
        self.applied = True

    async def commit(self) -> None:
        self.committed = True

    async def abandon(self) -> None:
        self.abandoned = True


class _TransactionalWriter:
    def __init__(self) -> None:
        self.txns: list[_RecordingResourceTxn] = []
        self.storage: dict[str, bytes] = {}

    async def begin_txn(self, context: ResourceTxnContext) -> _RecordingResourceTxn:
        txn = _RecordingResourceTxn(context)
        self.txns.append(txn)
        return txn

    async def recover(self, context: ResourceRecoveryContext) -> None:
        del context

    async def read(self, path: str) -> bytes:
        return self.storage[path]

    async def exists(self, path: str) -> bool:
        return path in self.storage

    async def list_dir(self, path: str) -> list[str]:
        del path
        return sorted(self.storage)

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: str = "agent",
    ) -> WriteResult:
        del rationale, author
        self.storage[path] = content
        return WriteResult(path=path, path_class="managed")

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: str = "agent",
    ) -> WriteResult:
        del rationale, author
        if self.storage.get(path) != old:
            return WriteResult(path=path, path_class="managed", error="stale")
        self.storage[path] = new
        return WriteResult(path=path, path_class="managed")

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: str = "agent",
    ) -> WriteResult:
        del rationale, author
        self.storage.pop(path, None)
        return WriteResult(path=path, path_class="managed")

    def classify(self, path: str) -> PathClass:
        del path
        return "managed"

    def batch(
        self,
        *,
        rationale: str,
        author: str = "agent",
    ) -> _NoopBatch:
        del rationale, author
        return _NoopBatch()


class _FailingAppendStore(InMemoryTrajectoryStore):
    def commit_turn(
        self,
        session_id: str,
        commit: TrajectoryCommit,
    ) -> None:
        del session_id, commit
        raise RuntimeError("authoritative append failed")


class _OrderedEffectScope(_RecordingEffectScope):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    async def abandon_turn(self, txn: EffectTxn) -> None:
        self._order.append("effect")
        await super().abandon_turn(txn)


class _OrderedResourceTxn(_RecordingResourceTxn):
    def __init__(self, context: ResourceTxnContext, order: list[str]) -> None:
        super().__init__(context)
        self._order = order

    async def abandon(self) -> None:
        if self._order != ["effect"]:
            raise RuntimeError("resource rollback raced effect rollback")
        self._order.append("resource")
        await super().abandon()


class _OrderedTransactionalWriter(_TransactionalWriter):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    async def begin_txn(self, context: ResourceTxnContext) -> _OrderedResourceTxn:
        txn = _OrderedResourceTxn(context, self._order)
        self.txns.append(txn)
        return txn


@pytest.mark.asyncio
async def test_effect_scope_wraps_committed_turns() -> None:
    scope = _RecordingEffectScope()
    session = Session(SessionRuntimeConfig(
        stream_fn=_StaticStream(),
        model=_model(),
        system="test",
    ))
    session.register_effect_scope(scope)

    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert scope.events == [
        ("begin", session.id, 0),
        ("prepare", session.id, 0),
        ("commit", session.id, 0),
    ]


@pytest.mark.asyncio
async def test_shutdown_has_one_cancellation_safe_cleanup_boundary(
    tmp_path: Path,
) -> None:
    close_started = threading.Event()
    close_release = threading.Event()
    close_finished = threading.Event()

    def close_environment() -> None:
        close_started.set()
        if not close_release.wait(timeout=2.0):
            raise TimeoutError("test did not release environment close")
        close_finished.set()

    session = Session(
        SessionRuntimeConfig(
            stream_fn=_StaticStream(),
            model=_model(),
        )
    )
    session.register_operations(
        environment=LocalEnvironmentOperations(
            cwd=tmp_path,
            close_callback=close_environment,
        )
    )

    cancelled_shutdown = asyncio.create_task(session.shutdown())
    assert await asyncio.to_thread(close_started.wait, 1.0)
    cancelled_shutdown.cancel()
    concurrent_shutdown = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0.05)
    assert not cancelled_shutdown.done()
    assert not concurrent_shutdown.done()

    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_shutdown
    await concurrent_shutdown
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_cancelled_spawn_publishes_paired_child_lifecycle() -> None:
    store = _DelayedChildVisibilityStore()
    parent = await create_session(
        SessionBuildConfig(
            extensions=[],
            stream_fn=_StaticStream(),
            model=_model(),
            store=store,
        )
    )
    store.parent_session_id = parent.id
    started: list[str] = []
    ended: list[str] = []
    parent.on(
        ChildSessionStartEvent.CHANNEL,
        lambda event: started.append(event.child_session_id),
    )
    parent.on(
        ChildSessionEndEvent.CHANNEL,
        lambda event: ended.append(event.child_session_id),
    )

    spawn = asyncio.create_task(parent.spawn(purpose="cancelled-child"))
    assert await asyncio.to_thread(store.visibility_started.wait, 1.0)
    spawn.cancel()
    store.visibility_release.set()
    with pytest.raises(asyncio.CancelledError):
        await spawn

    assert len(started) == 1
    assert ended == started
    assert store.session_children(parent.id) == started
    await parent.shutdown()


@pytest.mark.asyncio
async def test_local_resource_mutations_settle_before_cancellation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    resource_root = tmp_path / "resources"
    workspace.mkdir()
    store = LocalResourceStore(
        workspace_root=workspace,
        root=resource_root,
    )

    async def hold_store_lock() -> tuple[threading.Event, asyncio.Task[None]]:
        ready = threading.Event()
        release = threading.Event()

        def hold() -> None:
            resource_root.mkdir(parents=True, exist_ok=True)
            with (resource_root / "resource.lock").open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                ready.set()
                if not release.wait(timeout=2.0):
                    raise TimeoutError("test did not release resource lock")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        task = asyncio.create_task(asyncio.to_thread(hold))
        assert await asyncio.to_thread(ready.wait, 1.0)
        return release, task

    write_release, write_lock = await hold_store_lock()
    write = asyncio.create_task(
        store.write(
            "value.txt",
            b"value",
            rationale="cancellation contract",
        )
    )
    await asyncio.sleep(0.05)
    write.cancel()
    await asyncio.sleep(0.05)
    assert not write.done()
    write_release.set()
    await write_lock
    with pytest.raises(asyncio.CancelledError):
        await write
    assert (workspace / "value.txt").read_bytes() == b"value"

    txn = await store.begin_txn(
        ResourceTxnContext(
            session_id="session",
            turn_id="turn",
            turn_index=0,
        )
    )
    await txn.create(
        ResourceRef(namespace="workspace", path="staged.txt"),
        b"staged",
        rationale="cancellation contract",
    )
    prepare_release, prepare_lock = await hold_store_lock()
    prepare = asyncio.create_task(txn.prepare())
    await asyncio.sleep(0.05)
    prepare.cancel()
    await asyncio.sleep(0.05)
    assert not prepare.done()
    prepare_release.set()
    await prepare_lock
    with pytest.raises(asyncio.CancelledError):
        await prepare
    await txn.abandon()
    transaction_root = resource_root / "resource_transactions"
    assert not transaction_root.exists() or not tuple(transaction_root.iterdir())


@pytest.mark.asyncio
async def test_effect_scope_fork_and_resume() -> None:
    store = InMemoryTrajectoryStore()
    scope = _RecordingEffectScope()
    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=_StaticStream(),
        model=_model(),
        store=store,
        effect_scope=scope,
    ))

    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    forked = await Session.fork(session, at=0, purpose="branch")
    assert scope.events[-1] == ("fork", session.id, forked.id)
    assert forked.get_effect_scope() is scope.children[0]
    assert scope.fork_leases[0].committed
    assert not scope.fork_leases[0].abandoned
    await forked.shutdown()

    scope.block_fork = True
    cancelled_fork = asyncio.create_task(
        Session.fork(session, at=0, purpose="cancelled-branch")
    )
    await asyncio.wait_for(scope.fork_started.wait(), timeout=1.0)
    cancelled_fork.cancel()
    scope.fork_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancelled_fork, timeout=1.0)
    assert not scope.fork_leases[-1].committed
    assert scope.fork_leases[-1].abandoned

    resume_scope = _RecordingEffectScope()
    resumed = await Session.resume(
        session.id,
        store,
        AgentSessionConfig(
            extensions=[],
            stream_fn=_StaticStream("resumed"),
            model=_model(),
            effect_scope=resume_scope,
        ),
    )

    assert resumed.id == session.id
    assert resume_scope.events == [("restore", session.id, 1)]
    await resumed.shutdown()


@pytest.mark.asyncio
async def test_unpublished_turn_rolls_back_effect_before_resource() -> None:
    order: list[str] = []
    scope = _OrderedEffectScope(order)
    writer = _OrderedTransactionalWriter(order)
    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=_StaticStream(),
        model=_model(),
        store=_FailingAppendStore(),
        resource_writer=writer,
        effect_scope=scope,
    ))

    session.start()
    receipt = await session.prompt("go")
    with pytest.raises(RuntimeError, match="authoritative append failed"):
        await receipt.wait()
    await session.shutdown()

    assert order == ["effect", "resource"]
    assert writer.txns[0].applied
    assert writer.txns[0].abandoned
    assert len(session.trajectory) == 0


@pytest.mark.asyncio
async def test_atom_catalog_records_active_set_service() -> None:
    catalog = _RecordingCatalog()
    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=_StaticStream(),
        model=_model(),
        atom_catalog=catalog,
    ))

    fingerprint = session.services.get(ACTIVE_SET_FINGERPRINT_SERVICE)

    assert len(catalog.active_sets) == 1
    active_set = catalog.active_sets[0]
    assert active_set.session_id == session.id
    assert active_set.root_session_id == session.id
    assert active_set.parent_session_id is None
    assert active_set.scenario is None
    assert active_set.provider == "direct"
    assert active_set.atoms == ()
    assert active_set.created_at > 0
    assert isinstance(fingerprint, ActiveSetFingerprint)
    assert fingerprint.digest == "test:0"


def test_multiple_providers_require_explicit_selection_policy() -> None:
    first = ProviderConfig(
        stream_fn=_StaticStream("first"),
        model=_model(),
        name="first",
    )
    second = ProviderConfig(
        stream_fn=_StaticStream("second"),
        model=Model(
            id="second-model",
            provider="second",
            context_window=128_000,
            max_output_tokens=4_096,
        ),
        name="second",
    )
    unresolved = Session()
    unresolved.register_provider("first", first)

    with pytest.raises(RuntimeError, match="ProviderResolver"):
        unresolved.register_provider("second", second)

    assert unresolved.provider_names() == ["first"]
    assert unresolved.get_provider() is first

    resolved = Session(SessionRuntimeConfig(
        provider_resolver=_ProviderResolver("second")
    ))
    resolved.register_provider("first", first)
    resolved.register_provider("second", second)

    assert resolved.get_provider() is second


def test_session_spec_provider_precedence_is_source_accurate(
    tmp_path: Path,
) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text(
        "\n".join(
            (
                "[providers.openai]",
                'model = "user-model"',
                'base_url = "https://user.example/v1"',
                "verify_ssl = true",
                "",
            )
        )
    )
    project_config = tmp_path / "project.toml"
    project_config.write_text(
        "\n".join(
            (
                'default_provider = "openai"',
                "",
                "[providers.openai]",
                'model = "project-model"',
                'base_url = "https://project.example/v1"',
                'api_key_env = "PROJECT_OPENAI_KEY"',
                "prompt_cache_enabled = false",
                "",
            )
        )
    )
    resolved = DefaultSessionSpecResolver(
        project_config=project_config,
        user_config=user_config,
        env={
            "AGENTM_MODEL": "env-model",
            "PROJECT_OPENAI_KEY": "secret",
        },
    ).resolve(AgentSessionConfig())

    assert resolved.provider is not None
    assert resolved.provider.source.kind == "module"
    assert (
        resolved.provider.source.location
        == "agentm.extensions.builtin.llm_openai"
    )
    assert dict(resolved.provider.config) == {
        "api_key": "secret",
        "base_url": "https://project.example/v1",
        "model": "env-model",
        "name": "openai",
        "prompt_cache_enabled": False,
        "verify_ssl": True,
    }
    assert resolved.provider_identity is not None
    assert resolved.provider_identity.name == "openai"
    assert resolved.provider_identity.model_id == "env-model"
    provenance = {
        item.path: item.source
        for item in resolved.value_provenance
    }
    assert provenance["provider.model"] == "env"
    assert provenance["provider.base_url"] == "project_config"
    assert provenance["provider.api_key"] == "env"


@pytest.mark.asyncio
async def test_atom_catalog_freezes_atom_identity_versions() -> None:
    catalog = _RecordingCatalog()
    store = InMemoryTrajectoryStore()
    session = await create_session(SessionBuildConfig(
        extensions=[
            (
                "agentm.extensions.builtin.system_prompt",
                {"prompt": "catalog test"},
            )
        ],
        stream_fn=_StaticStream(),
        model=_model(),
        store=store,
        atom_catalog=catalog,
    ))

    assert len(catalog.active_sets) == 1
    atoms = catalog.active_sets[0].atoms
    assert len(atoms) == 1
    atom = atoms[0]
    assert atom.name == "system_prompt"
    assert atom.module_path == "agentm.extensions.builtin.system_prompt"
    assert atom.version is not None
    assert atom.version.resource_id == "atom:system_prompt"
    assert atom.version.media_type == "application/vnd.agentm.atom-identity+json"
    assert atom.version.metadata["module_path"] == atom.module_path
    assert atom.version.metadata["config_digest"] == atom.config_fingerprint

    version_store = session.get_versioned_resource_store()
    assert version_store is not None
    payload = await version_store.read(atom.version)
    assert b"agentm.extensions.builtin.system_prompt" in payload
    assert b"catalog test" in payload

    meta, _turns = store.load(session.id)
    assert meta.config["active_set_algorithm"] == "test"
    assert meta.config["active_set_digest"] == "test:1"
    assert meta.config["active_set_atom_count"] == 1


@pytest.mark.asyncio
async def test_tool_executor_receives_typed_requirements() -> None:
    executor = _RecordingToolExecutor()
    tool_call = AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id="call-1",
                name="needs_executor",
                arguments={"x": 1},
            )
        ],
        timestamp=0.0,
    )
    final = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        timestamp=0.0,
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=_QueuedStream(tool_call, final),
        model=_model(),
        tools=[_RequirementsTool()],
    ))
    session.register_tool_executor(executor)

    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.tool.name == "needs_executor"
    assert request.args == {"x": 1}
    assert request.requirements == _RequirementsTool.execution_requirements


@pytest.mark.asyncio
async def test_default_tool_executor_rejects_unsupported_requirements() -> None:
    tool_call = AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id="call-1",
                name="needs_executor",
                arguments={"x": 1},
            )
        ],
        timestamp=0.0,
    )
    final = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        timestamp=0.0,
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=_QueuedStream(tool_call, final),
        model=_model(),
        tools=[_RequirementsTool()],
    ))

    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    result = session.trajectory.turns[0].rounds[0].tool_results[0].result
    assert result.is_error
    assert "tool executor does not satisfy requirements" in result.content[0].text
    assert "isolation=process" in result.content[0].text
    assert "killable=true" in result.content[0].text


@pytest.mark.asyncio
async def test_process_executor_uses_strict_json_result_wire() -> None:
    executor = ProcessToolExecutor()
    requirements = ToolExecutionRequirements(
        isolation="process",
        killable=True,
    )
    result = await executor.execute(
        ToolExecutionRequest(
            tool=_ProcessEntrypointTool("tests.fixtures.process_tools:echo"),
            args={"text": "from child", "count": 3},
            requirements=requirements,
        )
    )

    assert isinstance(result, ToolResult)
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "from child"
    assert isinstance(result.content[1], ImageContent)
    assert result.content[1].data == b"\x00\x01"
    assert result.extras == {"nested": (3, True, None)}

    terminating = await executor.execute(
        ToolExecutionRequest(
            tool=_ProcessEntrypointTool("tests.fixtures.process_tools:terminate"),
            args={"text": "complete"},
            requirements=requirements,
        )
    )
    assert isinstance(terminating, ToolTerminate)
    assert terminating.reason == "test:complete"
    assert terminating.result.content[0].text == "complete"


@pytest.mark.asyncio
async def test_tool_orchestrator_propagates_caller_cancellation() -> None:
    started = asyncio.Event()
    finalized: list[str] = []
    started_count = 0

    class _WaitingTool:
        description = "wait"
        parameters: dict[str, Any] = {"type": "object"}

        def __init__(self, name: str) -> None:
            self.name = name

        async def execute(
            self,
            args: dict[str, Any],
            *,
            signal: CancelSignal | None = None,
        ) -> ToolResult:
            nonlocal started_count
            del args, signal
            started_count += 1
            if started_count == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finalized.append(self.name)
            raise AssertionError("unreachable")

    requirements = ToolExecutionRequirements(
        concurrency="parallel_safe",
        interrupt="cancel",
    )
    items = tuple(
        ToolWorkItem(
            index=index,
            call=ToolCallBlock(
                type="tool_call",
                id=f"call-{index}",
                name=name,
                arguments={},
            ),
            tool=_WaitingTool(name),
            args={},
            requirements=requirements,
        )
        for index, name in enumerate(("first", "second"))
    )
    async def consume_results() -> None:
        async for _result in DefaultToolOrchestrator().stream_batch(
            ToolOrchestrationRequest(
                items=items,
                session_id="session",
                turn_id="turn",
                turn_index=0,
            )
        ):
            pass

    task = asyncio.create_task(consume_results())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    assert sorted(finalized) == ["first", "second"]


@pytest.mark.asyncio
async def test_process_executor_reaps_on_signal_and_caller_cancellation(
    tmp_path: Path,
) -> None:
    executor = ProcessToolExecutor()
    requirements = ToolExecutionRequirements(
        isolation="process",
        killable=True,
        interrupt="cancel",
    )

    async def start(marker: Path) -> asyncio.Task[ToolResult | ToolOutcome]:
        task = asyncio.create_task(
            executor.execute(
                ToolExecutionRequest(
                    tool=_ProcessEntrypointTool(
                        "tests.fixtures.process_tools:wait_forever"
                    ),
                    args={"started_path": str(marker)},
                    requirements=requirements,
                ),
                signal=signal,
            )
        )
        for _ in range(200):
            if marker.exists():
                return task
            await asyncio.sleep(0.01)
        task.cancel()
        raise TimeoutError("process tool did not start")

    signal = EventCancelSource()
    signalled = await start(tmp_path / "signal-started")
    signal.set("task_stop")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(signalled, timeout=2.0)

    signal = EventCancelSource()
    caller_cancelled = await start(tmp_path / "caller-started")
    caller_cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller_cancelled, timeout=2.0)


@pytest.mark.asyncio
async def test_local_bash_reaps_on_signal_and_caller_cancellation(
    tmp_path: Path,
) -> None:
    operations = LocalBashOperations()

    async def start(
        marker: Path,
        signal: EventCancelSource,
    ) -> asyncio.Task[object]:
        command = (
            f"printf started > {shlex.quote(str(marker))}; "
            "sleep 3600"
        )
        task: asyncio.Task[object] = asyncio.create_task(
            operations.exec(
                command,
                cwd=str(tmp_path),
                signal=signal,
            )
        )
        for _ in range(200):
            if marker.exists():
                return task
            await asyncio.sleep(0.01)
        task.cancel()
        raise TimeoutError("local bash command did not start")

    signal = EventCancelSource()
    signalled = await start(tmp_path / "bash-signal-started", signal)
    signal.set("user_cancel")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(signalled, timeout=2.0)

    signal = EventCancelSource()
    caller_cancelled = await start(tmp_path / "bash-caller-started", signal)
    caller_cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller_cancelled, timeout=2.0)


@pytest.mark.asyncio
async def test_sandbox_executor_requires_typed_environment_adapter(
    tmp_path: Path,
) -> None:
    environment = LocalEnvironmentOperations(cwd=tmp_path)
    executor = SandboxToolExecutor(environment)
    requirements = ToolExecutionRequirements(
        isolation="environment",
        environment_id=environment.ref.id,
    )
    tool = _EnvironmentAwareTool()
    result = await executor.execute(
        ToolExecutionRequest(
            tool=tool,
            args={},
            requirements=requirements,
            environment=environment.ref,
            cwd=str(tmp_path),
        )
    )

    assert isinstance(result, ToolResult)
    assert tool.calls == [(environment.ref.id, str(tmp_path))]

    with pytest.raises(RuntimeError, match="EnvironmentExecutableTool"):
        await executor.execute(
            ToolExecutionRequest(
                tool=_ProcessEntrypointTool(
                    "tests.fixtures.process_tools:echo"
                ),
                args={},
                requirements=requirements,
                environment=environment.ref,
            )
        )


@pytest.mark.asyncio
async def test_session_spec_resolver_records_resolved_spec() -> None:
    store = InMemoryTrajectoryStore()
    session = await create_from_config(
        AgentSessionConfig(
            spec_resolver=_Resolver(),
            stream_fn=_StaticStream(),
            model=_model(),
            trajectory_store=store,
        )
    )

    resolved = session.services.get(RESOLVED_SESSION_SPEC_SERVICE)
    meta, _turns = store.load(session.id)

    assert session.ctx.scenario == "resolved"
    assert isinstance(resolved, ResolvedSessionSpec)
    assert resolved.provenance == {"source": "test"}
    assert meta.config["resolved_spec_digest"].startswith("sha256:")
    assert meta.config["resolved_spec_provenance_json"] == '{"source":"test"}'


@pytest.mark.asyncio
async def test_operations_atom_registers_environment_backend_and_bash_alias() -> None:
    session = await create_session(SessionBuildConfig(
        extensions=[("agentm.extensions.builtin.operations", {})],
        stream_fn=_StaticStream(),
        model=_model(),
        cwd="/tmp",
    ))

    environment = session.services.get(
        ENVIRONMENT_OPERATIONS_SERVICE,
        EnvironmentOperations,
    )
    bash = session.services.get(BASH_OPERATIONS_SERVICE)

    assert isinstance(environment, EnvironmentOperations)
    assert environment.bash is bash
    assert environment.ref.kind == "local"
    assert environment.ref.id.startswith("local:")
    assert isinstance(environment.ref.metadata["cwd"], str)


@pytest.mark.asyncio
async def test_context_projection_service_projects_committed_history() -> None:
    projection = _RecordingProjection()
    stream = _RecordingStream()
    services = ServiceRegistry()
    services.register(
        CONTEXT_PROJECTION_SERVICE,
        projection,
        ContextProjection,
        scope="session",
    )
    session = Session(SessionRuntimeConfig(
        stream_fn=stream,
        model=_model(),
        services=services,
    ))

    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(projection.calls) == 1
    projection_input, budget = projection.calls[0]
    assert projection_input.turns == ()
    assert projection_input.source == "turns"
    assert budget.max_input_tokens == _model().context_window
    assert stream.calls[0][0].content[0].text == "projected history"
    assert stream.calls[0][1].content[0].text == "go"


@pytest.mark.asyncio
async def test_file_tools_write_uses_active_resource_txn() -> None:
    writer = _TransactionalWriter()
    tool_call = AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id="call-1",
                name="write",
                arguments={
                    "path": "out.txt",
                    "content": "hello",
                    "rationale": "test write",
                },
            )
        ],
        timestamp=0.0,
    )
    final = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        timestamp=0.0,
    )
    session = await create_session(SessionBuildConfig(
        extensions=[("agentm.extensions.builtin.file_tools", {"tools": ["write"]})],
        stream_fn=_QueuedStream(tool_call, final),
        model=_model(),
        resource_writer=writer,
    ))

    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    assert len(writer.txns) == 1
    txn = writer.txns[0]
    assert txn.context.session_id == session.id
    assert txn.writes == [
        (ResourceRef(namespace="workspace", path="out.txt"), b"hello", "test write")
    ]
    assert txn.committed
    assert not txn.abandoned
    assert session.get_resource_txn() is None
    assert session.trajectory.turns[0].meta.resource_mutations == tuple(txn.mutations)


@pytest.mark.asyncio
async def test_resource_txn_abandons_on_turn_failure() -> None:
    writer = _TransactionalWriter()
    session = await create_session(SessionBuildConfig(
        extensions=[],
        stream_fn=_FailingStream(),
        model=_model(),
        resource_writer=writer,
    ))

    session.start()
    receipt = await session.prompt("go")
    with pytest.raises(RuntimeError, match="stream failed"):
        await receipt.wait()
    await session.shutdown()

    assert len(writer.txns) == 1
    assert writer.txns[0].abandoned
    assert not writer.txns[0].committed
    assert session.get_resource_txn() is None
