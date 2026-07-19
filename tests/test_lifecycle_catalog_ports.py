"""Focused coverage for lifecycle and catalog SDK ports."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

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
from agentm.core.abi.lifecycle import EffectTxn, EnvironmentFork
from agentm.core.abi.messages import (
    AssistantMessage,
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
from agentm.core.abi.stream import MessageEnd, Model, TextDelta
from agentm.core.abi.tool import ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import (
    ToolExecutionCapabilities,
    ToolExecutionRequest,
    ToolExecutionRequirements,
)
from agentm.core.abi.trajectory import Turn, TurnRef
from agentm.core.runtime.session import Session
from agentm.core.runtime.session_factory import create_from_config, create_session
from agentm.core.runtime.stores.memory import InMemoryTrajectoryStore
from agentm.config import DefaultSessionSpecResolver


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


class _RecordingEffectScope:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | int]] = []
        self.children: list[_RecordingEffectScope] = []

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
        child = _RecordingEffectScope()
        self.children.append(child)
        self.events.append(("fork", source_session_id, child_session_id))
        del ref
        return EnvironmentFork(effect_scope=child, cwd="")

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


@pytest.mark.asyncio
async def test_effect_scope_wraps_committed_turns() -> None:
    scope = _RecordingEffectScope()
    session = Session(stream_fn=_StaticStream(), model=_model(), system="test")
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
async def test_effect_scope_fork_and_resume() -> None:
    store = InMemoryTrajectoryStore()
    scope = _RecordingEffectScope()
    session = await create_session(
        extensions=[],
        stream_fn=_StaticStream(),
        model=_model(),
        store=store,
        effect_scope=scope,
    )

    session.start()
    await session.prompt("go")
    await _wait_turn(session)
    await session.shutdown()

    forked = await Session.fork(session, at=0, purpose="branch")
    assert scope.events[-1] == ("fork", session.id, forked.id)
    assert forked.get_effect_scope() is scope.children[0]

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


@pytest.mark.asyncio
async def test_atom_catalog_records_active_set_service() -> None:
    catalog = _RecordingCatalog()
    session = await create_session(
        extensions=[],
        stream_fn=_StaticStream(),
        model=_model(),
        atom_catalog=catalog,
    )

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

    resolved = Session(provider_resolver=_ProviderResolver("second"))
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
    module, config = resolved.provider
    assert module == "agentm.extensions.builtin.llm_openai"
    assert config == {
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
    session = await create_session(
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
    )

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
    session = Session(
        stream_fn=_QueuedStream(tool_call, final),
        model=_model(),
        tools=[_RequirementsTool()],
    )
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
    session = Session(
        stream_fn=_QueuedStream(tool_call, final),
        model=_model(),
        tools=[_RequirementsTool()],
    )

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
async def test_session_spec_resolver_records_resolved_spec() -> None:
    store = InMemoryTrajectoryStore()
    session = await create_from_config(
        AgentSessionConfig(
            spec_resolver=_Resolver(),
            stream_fn=_StaticStream(),
            model=_model(),
            store=store,
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
    session = await create_session(
        extensions=[("agentm.extensions.builtin.operations", {})],
        stream_fn=_StaticStream(),
        model=_model(),
        cwd="/tmp",
    )

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
    session = Session(
        stream_fn=stream,
        model=_model(),
        services=services,
    )

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
    session = await create_session(
        extensions=[("agentm.extensions.builtin.file_tools", {"tools": ["write"]})],
        stream_fn=_QueuedStream(tool_call, final),
        model=_model(),
        resource_writer=writer,
    )

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
    session = await create_session(
        extensions=[],
        stream_fn=_FailingStream(),
        model=_model(),
        resource_writer=writer,
    )

    session.start()
    receipt = await session.prompt("go")
    with pytest.raises(RuntimeError, match="stream failed"):
        await receipt.wait()
    await session.shutdown()

    assert len(writer.txns) == 1
    assert writer.txns[0].abandoned
    assert not writer.txns[0].committed
    assert session.get_resource_txn() is None
