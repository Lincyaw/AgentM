"""SDK-user behavior contracts for durable trajectory workflows."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agentm import (
    AgentSession,
    AgentSessionConfig,
    ExtensionSpec,
    Model,
    ProviderRequestFailed,
    TrajectoryStorage,
)
from agentm.core.abi.cancel import CancelSignal, cancel_reason
from agentm.core.abi.errors import ExtensionLoadError
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    text_message,
)
from agentm.core.abi.provider import ProviderPromptCacheRequest
from agentm.core.abi.store import (
    TrajectoryNodeQuery,
    TrajectoryNodeStore,
    TrajectoryStore,
)
from agentm.core.abi.stream import MessageEnd, TextDelta
from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.core.abi.trajectory import PromptCacheState
from agentm.core.abi.trigger import UserInput
from agentm.core.runtime.stores.jsonl import JsonlTrajectoryStore
from agentm.extensions.builtin.llm_openai import (
    OpenAIPromptCacheAdapter,
    OpenAIStreamFn,
)
from agentm.environments import LocalSnapshotEffectScope, LocalSnapshotStore
from agentm.scenarios import builtin_scenario_loader
from agentm.storage.resources import LocalResourceStore
from agentm.storage.trajectory import (
    JsonlTrajectoryNodeStore,
    PostgresTrajectoryNodeStore,
    PostgresTrajectoryStore,
)
from tests.fixtures.custom_trigger import CustomTrigger

_CONTEXT_PROJECTION = "agentm.extensions.builtin.context_projection"
_LLM_COMPACTION = "agentm.extensions.builtin.llm_compaction"
_MESSAGE_PATTERNS = "agentm.extensions.builtin.message_patterns"
_PROMPT_CACHE = "agentm.extensions.builtin.prompt_cache"
_OBSERVABLE_CACHE_ADAPTER = "tests.fixtures.prompt_cache_adapter"
_CUSTOM_TRIGGER = "tests.fixtures.custom_trigger"
_FILE_TOOLS = "agentm.extensions.builtin.file_tools"
_LOCAL_RESOURCES = "agentm.extensions.builtin.local_resources"
_OPERATIONS = "agentm.extensions.builtin.operations"
_BACKGROUND_EXEC = "agentm.extensions.builtin.background_exec"
_WAIT_FOR_CANCEL = object()


@dataclass(frozen=True, slots=True)
class _TrajectoryBackend:
    open_turn_store: Callable[[], TrajectoryStore]
    open_node_store: Callable[[], TrajectoryNodeStore]

    def open_storage(self) -> TrajectoryStorage:
        return TrajectoryStorage(
            turn_store=self.open_turn_store(),
            node_store=self.open_node_store(),
        )


def _jsonl_storage(
    turn_root: Path,
    node_root: Path,
) -> TrajectoryStorage:
    return TrajectoryStorage(
        turn_store=JsonlTrajectoryStore(turn_root),
        node_store=JsonlTrajectoryNodeStore(node_root),
    )


class _StubProvider:
    """Deterministic provider double at the public StreamFn boundary."""

    def __init__(self, *actions: str | AssistantMessage | object) -> None:
        self._actions = list(actions)
        self.requests: list[tuple[AgentMessage, ...]] = []
        self.stream_started = asyncio.Event()
        self.observed_cancel_reason: str | None = None

    async def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: str = "off",
    ) -> AsyncIterator[TextDelta | MessageEnd]:
        del model, tools, system, thinking
        self.requests.append(tuple(messages))
        if not self._actions:
            raise RuntimeError("stub provider has no queued response")
        action = self._actions.pop(0)
        if action is _WAIT_FOR_CANCEL:
            if signal is None:
                raise AssertionError("SDK did not pass a cancellation signal")
            self.stream_started.set()
            await signal.wait()
            self.observed_cancel_reason = cancel_reason(signal)
            raise RuntimeError("provider request cancelled")
        if isinstance(action, Exception):
            raise action
        if isinstance(action, AssistantMessage):
            yield MessageEnd(message=action)
            return
        if not isinstance(action, str):
            raise TypeError(f"unsupported provider action: {action!r}")
        response = AssistantMessage(
            role="assistant",
            content=(TextContent(type="text", text=action),),
            timestamp=0.0,
            stop_reason="end_turn",
        )
        yield TextDelta(text=action)
        yield MessageEnd(message=response)


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=(
            ToolCallBlock(
                type="tool_call",
                id=call_id,
                name=name,
                arguments=arguments,
            ),
        ),
        timestamp=0.0,
        stop_reason="tool_use",
    )


@pytest.fixture(params=("jsonl", "postgres"))
def trajectory_backend(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[_TrajectoryBackend]:
    if request.param == "jsonl":
        turn_root = tmp_path / "turns"
        node_root = tmp_path / "trajectory-nodes"
        yield _TrajectoryBackend(
            open_turn_store=lambda: JsonlTrajectoryStore(turn_root),
            open_node_store=lambda: JsonlTrajectoryNodeStore(node_root),
        )
        return

    database_url = os.environ.get("AGENTM_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("set AGENTM_TEST_POSTGRES_URL to run Postgres behavior contracts")
    psycopg = pytest.importorskip("psycopg")
    from agentm.storage.trajectory.psycopg import (
        PsycopgConnectionAdapter,
        connect as connect_postgres,
    )

    schema = f"agentm_test_{uuid.uuid4().hex}"
    admin = psycopg.connect(database_url)
    connections: list[PsycopgConnectionAdapter] = []
    with admin.cursor() as cursor:
        cursor.execute(f'CREATE SCHEMA "{schema}"')
    admin.commit()

    def open_turn_store() -> TrajectoryStore:
        connection = connect_postgres(database_url)
        connections.append(connection)
        return PostgresTrajectoryStore(connection, schema=schema)

    def open_node_store() -> TrajectoryNodeStore:
        connection = connect_postgres(database_url)
        connections.append(connection)
        return PostgresTrajectoryNodeStore(connection, schema=schema)

    try:
        yield _TrajectoryBackend(
            open_turn_store=open_turn_store,
            open_node_store=open_node_store,
        )
    finally:
        for connection in connections:
            connection.close()
        with admin.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.commit()
        admin.close()


def _model() -> Model:
    return Model(
        id="stub-model",
        provider="stub",
        context_window=128_000,
        max_output_tokens=4_096,
    )


@pytest.mark.asyncio
async def test_sdk_dsn_selects_one_paired_postgres_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = os.environ.get("AGENTM_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("set AGENTM_TEST_POSTGRES_URL to run the Postgres DSN contract")
    psycopg = pytest.importorskip("psycopg")
    from agentm.storage.trajectory.psycopg import connect as connect_postgres

    schema = f"agentm_test_{uuid.uuid4().hex}"
    monkeypatch.setenv("AGENTM_TRAJECTORY_DSN", database_url)
    monkeypatch.setenv("AGENTM_TRAJECTORY_SCHEMA", schema)
    monkeypatch.delenv("AGENTM_TRAJECTORY_DIR", raising=False)

    provider = _StubProvider("dsn-answer")
    admin = psycopg.connect(database_url)
    with admin.cursor() as cursor:
        cursor.execute(f'CREATE SCHEMA "{schema}"')
    admin.commit()
    try:
        session = await AgentSession.create(
            AgentSessionConfig(
                cwd=str(tmp_path),
                extensions=[],
                stream_fn=provider,
                model=_model(),
            )
        )
        session_id = session.session_id
        try:
            await session.run("dsn-question")
        finally:
            await session.shutdown()

        turn_connection = connect_postgres(database_url)
        node_connection = connect_postgres(database_url)
        try:
            turn_store = PostgresTrajectoryStore(
                turn_connection,
                schema=schema,
                create_schema=False,
            )
            node_store = PostgresTrajectoryNodeStore(
                node_connection,
                schema=schema,
                create_schema=False,
            )
            metadata, turns = turn_store.load(session_id)
            nodes = node_store.query_nodes(
                TrajectoryNodeQuery(session_id=session_id)
            )

            assert metadata.id == session_id
            assert len(turns) == 1
            assert turns[0].index == 0
            assert _text(
                [node.message for node in nodes if node.message is not None]
            ) == ["dsn-question", "dsn-answer"]
        finally:
            turn_connection.close()
            node_connection.close()
    finally:
        with admin.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.commit()
        admin.close()


def _trajectory_extensions(cache_key: str) -> list[tuple[str, dict[str, object]]]:
    return [
        (_CONTEXT_PROJECTION, {"mode": "exact_node_chain"}),
        (
            _PROMPT_CACHE,
            {
                "cache_key": cache_key,
                "tag_last_messages": 2,
                "register_provider_adapter": False,
            },
        ),
        (_OBSERVABLE_CACHE_ADAPTER, {}),
    ]


def _text(messages: Sequence[AgentMessage]) -> list[str]:
    return [
        block.text
        for message in messages
        for block in message.content
        if isinstance(block, TextContent)
    ]


def _cache_markers(messages: Sequence[AgentMessage]) -> set[str]:
    return {
        marker
        for message in messages
        if isinstance(
            marker := message.meta.tags.get("provider_cache_marker"),
            str,
        )
    }


def _write_local_scenario(root: Path, name: str) -> Path:
    scenario_dir = root / "contrib" / "scenarios" / name
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "manifest.yaml").write_text(
        "\n".join(
            (
                f"name: {name}",
                "extensions:",
                "  - local: local_echo",
                "",
            )
        ),
        encoding="utf-8",
    )
    source = scenario_dir / "local_echo.py"
    source.write_text(
        '''\
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.abi.messages import TextContent
from agentm.core.abi.tool import ToolResult


MANIFEST = ExtensionManifest(
    name="local_echo",
    description="Behavior-test local echo tool.",
    registers=("tool:local_echo",),
)


class LocalEcho:
    name = "local_echo"
    description = "Echo one value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    async def execute(self, args, *, signal=None):
        del signal
        return ToolResult(
            content=(TextContent(type="text", text=f"local:{args['value']}"),),
        )


def install(api, config):
    del config
    api.register_tool(LocalEcho())
''',
        encoding="utf-8",
    )
    return source


@pytest.mark.asyncio
async def test_sdk_scenario_local_extension_survives_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_local_scenario(tmp_path, "local_source")
    monkeypatch.chdir(tmp_path)
    provider = _StubProvider(
        "parent-answer",
        _tool_call("local-call", "local_echo", {"value": "fork"}),
        "fork-answer",
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            scenario="local_source",
            scenario_loader=builtin_scenario_loader,
            stream_fn=provider,
            model=_model(),
        )
    )
    forked: AgentSession | None = None
    try:
        await session.run("seed")
        forked = await AgentSession.fork(session, at=0, purpose="local-source-fork")
        await forked.run("use the local tool")
    finally:
        if forked is not None:
            await forked.shutdown()
        await session.shutdown()

    local_results = [
        block
        for message in provider.requests[-1]
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.tool_call_id == "local-call"
    ]
    assert len(local_results) == 1
    assert [
        content.text
        for content in local_results[0].content
        if isinstance(content, TextContent)
    ] == ["local:fork"]


@pytest.mark.asyncio
async def test_sdk_rejects_changed_scenario_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_local_scenario(tmp_path, "tampered_source")
    monkeypatch.chdir(tmp_path)
    spec = builtin_scenario_loader("tampered_source")
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# changed after resolution\n",
        encoding="utf-8",
    )

    with pytest.raises(ExtensionLoadError, match="source digest changed"):
        await AgentSession.create(
            AgentSessionConfig(
                extensions=list(spec.extensions),
                stream_fn=_StubProvider("unused"),
                model=_model(),
            )
        )


@pytest.mark.asyncio
async def test_sdk_resume_replays_history_and_durable_cache_state(
    trajectory_backend: _TrajectoryBackend,
) -> None:
    cache_key = "resume-prefix"
    provider = _StubProvider("answer-one", "answer-two")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=_trajectory_extensions(cache_key),
            stream_fn=provider,
            model=_model(),
            trajectory_storage=trajectory_backend.open_storage(),
        )
    )
    session_id = session.session_id
    try:
        await session.run("question-one")
        await session.run("question-two")
    finally:
        await session.shutdown()

    assert _cache_markers(provider.requests[1]) == {"resume-prefix:1"}

    resumed_provider = _StubProvider("answer-three")
    resumed = await AgentSession.resume(
        session_id,
        trajectory_backend.open_storage(),
        config=AgentSessionConfig(
            extensions=_trajectory_extensions(cache_key),
            stream_fn=resumed_provider,
            model=_model(),
        ),
    )
    try:
        transcript = await resumed.run("question-three")
    finally:
        await resumed.shutdown()

    assert _text(resumed_provider.requests[0]) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
        "question-three",
    ]
    assert _cache_markers(resumed_provider.requests[0]) == {"resume-prefix:2"}
    assert _text(transcript) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
        "question-three",
        "answer-three",
    ]


@pytest.mark.asyncio
async def test_sdk_persists_provider_failure_without_replaying_it(
    trajectory_backend: _TrajectoryBackend,
) -> None:
    provider = _StubProvider(RuntimeError("provider unavailable"))
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[
                (_CONTEXT_PROJECTION, {"mode": "exact_node_chain"}),
            ],
            stream_fn=provider,
            model=_model(),
            trajectory_storage=trajectory_backend.open_storage(),
        )
    )
    session_id = session.session_id
    try:
        with pytest.raises(RuntimeError, match="ProviderRequestFailed"):
            await session.run("failed-question")
    finally:
        await session.shutdown()

    _, failed_turns = trajectory_backend.open_turn_store().load(session_id)
    assert len(failed_turns) == 1
    failure = failed_turns[0].outcome.cause
    assert isinstance(failure, ProviderRequestFailed)
    assert failure.error_type == "RuntimeError"
    assert failure.detail == "provider unavailable"

    resumed_provider = _StubProvider("recovered-answer")
    resumed = await AgentSession.resume(
        session_id,
        trajectory_backend.open_storage(),
        config=AgentSessionConfig(
            extensions=[
                (_CONTEXT_PROJECTION, {"mode": "exact_node_chain"}),
            ],
            stream_fn=resumed_provider,
            model=_model(),
        ),
    )
    try:
        await resumed.run("retry-question")
    finally:
        await resumed.shutdown()

    assert _text(resumed_provider.requests[0]) == ["retry-question"]


@pytest.mark.asyncio
async def test_sdk_fork_replays_only_the_selected_prefix(
    trajectory_backend: _TrajectoryBackend,
) -> None:
    provider = _StubProvider("answer-one", "parent-answer", "branch-answer")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=_trajectory_extensions("fork-prefix"),
            stream_fn=provider,
            model=_model(),
            trajectory_storage=trajectory_backend.open_storage(),
        )
    )
    forked: AgentSession | None = None
    try:
        await session.run("question-one")
        await session.run("parent-only-question")
        forked = await AgentSession.fork(session, at=0, purpose="alternate")
        transcript = await forked.run("branch-only-question")
    finally:
        if forked is not None:
            await forked.shutdown()
        await session.shutdown()

    assert _text(provider.requests[2]) == [
        "question-one",
        "answer-one",
        "branch-only-question",
    ]
    assert _cache_markers(provider.requests[2]) == {"fork-prefix:2"}
    assert _text(transcript) == [
        "question-one",
        "answer-one",
        "branch-only-question",
        "branch-answer",
    ]


@pytest.mark.asyncio
async def test_sdk_fork_reinstalls_atoms_in_an_isolated_environment(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshots = LocalSnapshotStore(
        workspace_root=workspace,
        snapshot_root=tmp_path / "snapshots",
    )
    resources = LocalResourceStore(
        workspace_root=workspace,
        root=tmp_path / "resources",
    )
    provider = _StubProvider(
        _tool_call(
            "write-parent",
            "write",
            {"path": "parent.txt", "content": "parent"},
        ),
        "parent-written",
        _tool_call(
            "write-branch",
            "write",
            {"path": "branch.txt", "content": "branch"},
        ),
        "branch-written",
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(workspace),
            extensions=[
                (_OPERATIONS, {}),
                (_FILE_TOOLS, {"tools": ["write"]}),
            ],
            stream_fn=provider,
            model=_model(),
            trajectory_storage=_jsonl_storage(
                tmp_path / "environment-fork-turns",
                tmp_path / "environment-fork-nodes",
            ),
            resource_store=resources,
            resource_writer=resources,
            effect_scope=LocalSnapshotEffectScope(snapshotter=snapshots),
        )
    )
    forked: AgentSession | None = None
    try:
        await session.run("write the parent file")
        forked = await AgentSession.fork(session, at=0, purpose="isolated")
        await forked.run("write the branch file")

        child_workspace = Path(forked.cwd)
        assert child_workspace != workspace
        assert (child_workspace / "parent.txt").read_text() == "parent"
        assert (child_workspace / "branch.txt").read_text() == "branch"
        assert (workspace / "parent.txt").read_text() == "parent"
        assert not (workspace / "branch.txt").exists()
    finally:
        if forked is not None:
            await forked.shutdown()
        await session.shutdown()


@pytest.mark.asyncio
async def test_local_environment_restore_does_not_rewind_sdk_control_plane(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    control_plane = workspace / ".agentm"
    workspace.mkdir()
    control_plane.mkdir()
    world_file = workspace / "world.txt"
    control_file = control_plane / "state.json"
    world_file.write_text("before")
    control_file.write_text("before")
    snapshots = LocalSnapshotStore(
        workspace_root=workspace,
        snapshot_root=tmp_path / "snapshots",
    )
    before = await snapshots.snapshot(
        session_id="session-1",
        ref=0,
        metadata={"checkpoint": "before_turn", "turn_id": "turn-1"},
    )

    world_file.write_text("after")
    control_file.write_text("committed-control-state")
    await snapshots.restore_to(before)

    assert world_file.read_text() == "before"
    assert control_file.read_text() == "committed-control-state"


@pytest.mark.asyncio
async def test_sdk_file_toolbox_transactions_share_behavior_and_protect_constitution(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    protected = workspace / "src" / "agentm" / "core" / "abi"
    protected.mkdir(parents=True)
    (workspace / "core-manifest.yaml").write_text(
        "\n".join(
            (
                "version: 1",
                "constitution:",
                "  paths:",
                "    - src/agentm/core/**",
                "    - core-manifest.yaml",
                "managed:",
                "  globs: []",
                "",
            )
        )
    )
    (workspace / "note.txt").write_text("hello\n")
    provider = _StubProvider(
        _tool_call("read-note", "read", {"path": "note.txt"}),
        _tool_call(
            "edit-note-1",
            "edit",
            {
                "path": "note.txt",
                "old_string": "hello",
                "new_string": "world",
            },
        ),
        _tool_call(
            "edit-note-2",
            "edit",
            {
                "path": "note.txt",
                "old_string": "world",
                "new_string": "done",
            },
        ),
        "file-updated",
        _tool_call(
            "write-kernel",
            "write",
            {
                "path": "src/agentm/core/abi/hacked.py",
                "content": "unsafe",
            },
        ),
        "write-refused",
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(workspace),
            extensions=[
                (_LOCAL_RESOURCES, {}),
                (_FILE_TOOLS, {}),
            ],
            stream_fn=provider,
            model=_model(),
        )
    )
    try:
        await session.run("update the note twice")
        await session.run("modify the SDK kernel")
    finally:
        await session.shutdown()

    assert (workspace / "note.txt").read_text() == "done\n"
    assert not (protected / "hacked.py").exists()
    protected_results = [
        block
        for message in provider.requests[-1]
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.tool_call_id == "write-kernel"
    ]
    assert protected_results[0].is_error
    assert "constitution" in " ".join(
        content.text
        for content in protected_results[0].content
        if isinstance(content, TextContent)
    ).lower()


@pytest.mark.asyncio
async def test_sdk_interrupt_cancels_one_request_and_session_continues() -> None:
    provider = _StubProvider(_WAIT_FOR_CANCEL, "continued-answer")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[(_MESSAGE_PATTERNS, {})],
            stream_fn=provider,
            model=_model(),
        )
    )
    interrupted = asyncio.create_task(session.run("long-running-question"))
    try:
        await asyncio.wait_for(provider.stream_started.wait(), timeout=2.0)
        session.interrupt("user_cancel")
        await asyncio.wait_for(interrupted, timeout=2.0)
        transcript = await session.run("question-after-interrupt")
    finally:
        if not interrupted.done():
            interrupted.cancel()
        await session.shutdown()

    assert provider.observed_cancel_reason == "user_cancel"
    assert _text(provider.requests[1]) == [
        "long-running-question",
        "[Request interrupted by user]",
        "question-after-interrupt",
    ]
    assert _text(transcript)[-2:] == [
        "question-after-interrupt",
        "continued-answer",
    ]
    assert session.status()["phase"] == "closed"


@pytest.mark.asyncio
async def test_sdk_background_tool_owns_cancellation_after_detach() -> None:
    provider = _StubProvider(
        _tool_call("slow-call", "slow_tool", {}),
        _WAIT_FOR_CANCEL,
        "background-completion-observed",
    )
    completed = asyncio.Event()

    async def slow_tool(
        args: dict[str, object],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult:
        del args
        await asyncio.sleep(0.1)
        if signal is not None and signal.is_set():
            raise AssertionError("parent cancellation leaked into detached tool")
        completed.set()
        return ToolResult(
            content=(TextContent(type="text", text="background-success"),)
        )

    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[],
            extra_extensions=[
                ExtensionSpec.from_module(
                    _BACKGROUND_EXEC,
                    {"timeout": 0.01},
                )
            ],
            extra_tools=[
                FunctionTool(
                    name="slow_tool",
                    description="Complete after the foreground timeout.",
                    parameters={"type": "object", "properties": {}},
                    fn=slow_tool,
                )
            ],
            stream_fn=provider,
            model=_model(),
        )
    )
    interrupted = asyncio.create_task(session.run("detach-the-tool"))
    try:
        await asyncio.wait_for(provider.stream_started.wait(), timeout=2.0)
        session.interrupt("user_cancel")
        await asyncio.wait_for(interrupted, timeout=2.0)
        assert await session.idle(timeout=2.0)
    finally:
        if not interrupted.done():
            interrupted.cancel()
        await session.shutdown()

    assert provider.observed_cancel_reason == "user_cancel"
    assert completed.is_set()
    assert any(
        "Background task" in text and "background-success" in text
        for text in _text(provider.requests[-1])
    )


@pytest.mark.asyncio
async def test_sdk_checkpoints_materialized_steps_without_replaying_them(
    trajectory_backend: _TrajectoryBackend,
) -> None:
    storage = trajectory_backend.open_storage()
    store = storage.turn_store
    tool_response = _tool_call("blocking-call", "blocking_tool", {})
    provider = _StubProvider(tool_response)
    tool_started = asyncio.Event()

    async def blocking_tool(
        args: dict[str, object],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult:
        del args
        if signal is None:
            raise AssertionError("SDK did not pass a cancellation signal")
        tool_started.set()
        await signal.wait()
        return ToolResult(
            content=(TextContent(type="text", text="cancelled"),),
            is_error=True,
        )

    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[],
            stream_fn=provider,
            model=_model(),
            trajectory_storage=storage,
            extra_tools=[
                FunctionTool(
                    name="blocking_tool",
                    description="Wait for cancellation.",
                    parameters={"type": "object", "properties": {}},
                    fn=blocking_tool,
                )
            ],
        )
    )
    run_task = asyncio.create_task(session.run("checkpoint-question"))
    try:
        await asyncio.wait_for(tool_started.wait(), timeout=2.0)

        checkpoint = store.load_checkpoint(session.session_id)
        assert checkpoint is not None
        assert checkpoint.index == 0
        assert len(checkpoint.rounds) == 1
        assert checkpoint.rounds[0].response == tool_response
        assert checkpoint.rounds[0].tool_results == ()
        assert store.load(session.session_id)[1] == []

        session.interrupt("user_cancel")
        await asyncio.wait_for(run_task, timeout=2.0)
    finally:
        if not run_task.done():
            run_task.cancel()
        await session.shutdown()

    _, committed = store.load(session.session_id)
    assert len(committed) == 1
    assert store.load_checkpoint(session.session_id) is None


@pytest.mark.asyncio
async def test_sdk_child_cancellation_domain_is_explicit() -> None:
    inherited_provider = _StubProvider(_WAIT_FOR_CANCEL)
    parent = await AgentSession.create(
        AgentSessionConfig(
            extensions=[],
            stream_fn=_StubProvider("unused"),
            model=_model(),
        )
    )
    inherited = await parent.spawn(
        stream_fn=inherited_provider,
        model=_model(),
        parent_cancellation="inherit",
    )
    inherited_run = asyncio.create_task(inherited.run("foreground-work"))
    try:
        await asyncio.wait_for(
            inherited_provider.stream_started.wait(),
            timeout=2.0,
        )
        parent.interrupt("user_cancel")
        await asyncio.wait_for(inherited_run, timeout=2.0)
    finally:
        await inherited.shutdown()
        await parent.shutdown()
    assert inherited_provider.observed_cancel_reason == "user_cancel"

    independent_provider = _StubProvider(_WAIT_FOR_CANCEL)
    parent = await AgentSession.create(
        AgentSessionConfig(
            extensions=[],
            stream_fn=_StubProvider("unused"),
            model=_model(),
        )
    )
    independent = await parent.spawn(
        stream_fn=independent_provider,
        model=_model(),
        parent_cancellation="independent",
    )
    independent_run = asyncio.create_task(independent.run("background-work"))
    try:
        await asyncio.wait_for(
            independent_provider.stream_started.wait(),
            timeout=2.0,
        )
        parent.interrupt("user_cancel")
        await asyncio.sleep(0)
        assert not independent_run.done()
        independent.interrupt("task_stop")
        await asyncio.wait_for(independent_run, timeout=2.0)
    finally:
        await independent.shutdown()
        await parent.shutdown()
    assert independent_provider.observed_cancel_reason == "task_stop"


@pytest.mark.asyncio
async def test_sdk_compaction_persists_summary_across_resume(
    trajectory_backend: _TrajectoryBackend,
    tmp_path: Path,
) -> None:
    resource_root = tmp_path / "resources"

    def open_resources() -> LocalResourceStore:
        return LocalResourceStore(
            workspace_root=tmp_path,
            root=resource_root,
        )

    extensions = [
        (
            _LLM_COMPACTION,
            {
                "max_messages": 5,
                "keep_last_turns": 1,
            },
        )
    ]
    provider = _StubProvider(
        "answer-one",
        "answer-two",
        "answer-three",
        "durable-summary",
        "answer-four",
    )
    resources = open_resources()
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=extensions,
            stream_fn=provider,
            model=_model(),
            trajectory_storage=trajectory_backend.open_storage(),
            resource_store=resources,
            resource_writer=resources,
        )
    )
    session_id = session.session_id
    try:
        await session.run("question-one")
        await session.run("question-two")
        await session.run("question-three")
        await session.run("question-four")
    finally:
        await session.shutdown()

    assert _text(provider.requests[3]) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
    ]
    assert _text(provider.requests[4]) == [
        "<conversation-summary>\ndurable-summary\n</conversation-summary>",
        "question-three",
        "answer-three",
        "question-four",
    ]

    resumed_provider = _StubProvider("answer-five")
    resumed_resources = open_resources()
    resumed = await AgentSession.resume(
        session_id,
        trajectory_backend.open_storage(),
        config=AgentSessionConfig(
            extensions=extensions,
            stream_fn=resumed_provider,
            model=_model(),
            resource_store=resumed_resources,
            resource_writer=resumed_resources,
        ),
    )
    try:
        transcript = await resumed.run("question-five")
    finally:
        await resumed.shutdown()

    assert len(resumed_provider.requests) == 1
    assert _text(resumed_provider.requests[0]) == [
        "<conversation-summary>\ndurable-summary\n</conversation-summary>",
        "question-three",
        "answer-three",
        "question-four",
        "answer-four",
        "question-five",
    ]
    assert _text(transcript)[-2:] == ["question-five", "answer-five"]


@pytest.mark.asyncio
async def test_sdk_fork_inherits_compaction_at_matching_logical_leaf(
    trajectory_backend: _TrajectoryBackend,
    tmp_path: Path,
) -> None:
    provider = _StubProvider(
        "answer-one",
        "answer-two",
        "answer-three",
        "shared-summary",
        "parent-answer",
        "branch-answer",
    )
    resources = LocalResourceStore(
        workspace_root=tmp_path,
        root=tmp_path / "fork-resources",
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[
                (
                    _LLM_COMPACTION,
                    {
                        "max_messages": 5,
                        "keep_last_turns": 1,
                    },
                )
            ],
            stream_fn=provider,
            model=_model(),
            trajectory_storage=trajectory_backend.open_storage(),
            resource_store=resources,
            resource_writer=resources,
        )
    )
    forked: AgentSession | None = None
    try:
        await session.run("question-one")
        await session.run("question-two")
        await session.run("question-three")
        await session.run("parent-only-question")
        forked = await AgentSession.fork(session, at=2, purpose="alternate")
        transcript = await forked.run("branch-only-question")
    finally:
        if forked is not None:
            await forked.shutdown()
        await session.shutdown()

    assert len(provider.requests) == 6
    assert _text(provider.requests[5]) == [
        "<conversation-summary>\nshared-summary\n</conversation-summary>",
        "question-three",
        "answer-three",
        "branch-only-question",
    ]
    assert _text(transcript)[-2:] == [
        "branch-only-question",
        "branch-answer",
    ]


@pytest.mark.asyncio
async def test_sdk_interrupt_cancels_compaction_and_session_continues(
    tmp_path: Path,
) -> None:
    provider = _StubProvider(
        "answer-one",
        "answer-two",
        _WAIT_FOR_CANCEL,
        "durable-summary",
        "continued-answer",
    )
    resources = LocalResourceStore(
        workspace_root=tmp_path,
        root=tmp_path / "cancel-resources",
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[
                (
                    _LLM_COMPACTION,
                    {
                        "max_messages": 3,
                        "keep_last_turns": 1,
                    },
                )
            ],
            stream_fn=provider,
            model=_model(),
            trajectory_storage=_jsonl_storage(
                tmp_path / "cancel-turns",
                tmp_path / "cancel-nodes",
            ),
            resource_store=resources,
            resource_writer=resources,
        )
    )
    interrupted: asyncio.Task[list[AgentMessage]] | None = None
    try:
        await session.run("question-one")
        await session.run("question-two")
        interrupted = asyncio.create_task(session.run("question-three"))
        await asyncio.wait_for(provider.stream_started.wait(), timeout=2.0)
        session.interrupt("user_cancel")
        await asyncio.wait_for(interrupted, timeout=2.0)
        transcript = await session.run("question-after-interrupt")
    finally:
        if interrupted is not None and not interrupted.done():
            interrupted.cancel()
        await session.shutdown()

    assert provider.observed_cancel_reason == "user_cancel"
    assert _text(provider.requests[3]) == [
        "question-one",
        "answer-one",
        "question-two",
        "answer-two",
    ]
    assert _text(transcript)[-2:] == [
        "question-after-interrupt",
        "continued-answer",
    ]


class _EmptyOpenAIStream:
    def __aiter__(self) -> "_EmptyOpenAIStream":
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration

    async def close(self) -> None:
        return None


class _OpenAIChunkStream:
    def __init__(self, *chunks: object) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "_OpenAIChunkStream":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        return None


class _OpenAICompletionsStub:
    def __init__(self, stream: object | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._stream = stream

    async def create(self, **body: Any) -> object:
        self.requests.append(body)
        return (
            self._stream
            if self._stream is not None
            else _EmptyOpenAIStream()
        )


class _OpenAIClientStub:
    def __init__(self, stream: object | None = None) -> None:
        self.completions = _OpenAICompletionsStub(stream)
        self.chat = type("_Chat", (), {"completions": self.completions})()


@pytest.mark.asyncio
async def test_openai_provider_materializes_prompt_cache_request_fields() -> None:
    model = Model(
        id="gpt-5-mini",
        provider="openai",
        context_window=128_000,
        max_output_tokens=1_024,
    )
    adapter = OpenAIPromptCacheAdapter(retention="24h")
    adapted = adapter.apply_prompt_cache(
        ProviderPromptCacheRequest(
            messages=[text_message("cached-prefix")],
            model=model,
            state=PromptCacheState(cache_key="stable-sdk-session"),
        )
    )
    client = _OpenAIClientStub()
    stream_fn = OpenAIStreamFn(client=client)

    _ = [
        event
        async for event in stream_fn(
            messages=list(adapted.messages),
            model=model,
            tools=[],
        )
    ]

    assert client.completions.requests[0]["prompt_cache_key"] == (
        "stable-sdk-session"
    )
    assert client.completions.requests[0]["prompt_cache_retention"] == "24h"


@pytest.mark.asyncio
async def test_openai_provider_rejects_malformed_sdk_usage() -> None:
    usage = type(
        "_Usage",
        (),
        {
            "prompt_tokens": "12",
            "completion_tokens": 3,
            "prompt_tokens_details": None,
        },
    )()
    chunk = type("_Chunk", (), {"usage": usage, "choices": []})()
    stream_fn = OpenAIStreamFn(
        client=_OpenAIClientStub(_OpenAIChunkStream(chunk))
    )

    with pytest.raises(TypeError, match="prompt_tokens"):
        _ = [
            event
            async for event in stream_fn(
                messages=[text_message("hello")],
                model=Model(
                    id="gpt-test",
                    provider="openai",
                    context_window=8_192,
                    max_output_tokens=1_024,
                ),
                tools=[],
            )
        ]


@pytest.mark.asyncio
async def test_sdk_trigger_envelope_is_routed_and_persisted(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "trigger-envelope-turns"
    provider = _StubProvider("envelope-answer")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[],
            stream_fn=provider,
            model=_model(),
            trajectory_storage=_jsonl_storage(
                store_path,
                tmp_path / "trigger-envelope-nodes",
            ),
        )
    )
    session.start()
    try:
        receipt = session.push_trigger(
            UserInput(
                content=(TextContent(type="text", text="enveloped-question"),)
            ),
            target_session_id=session.session_id,
            target_agent_id=session.session_id,
            origin="channel",
            mode="task-notification",
            is_meta=True,
            meta={"request_id": "request-1"},
        )
        await receipt.wait()
        with pytest.raises(ValueError, match="route to the target session"):
            session.push_trigger(
                UserInput(
                    content=(TextContent(type="text", text="misrouted"),)
                ),
                target_session_id="another-session",
            )
        session_id = session.session_id
    finally:
        await session.shutdown()

    request_message = provider.requests[0][0]
    assert request_message.meta.origin == "channel"
    assert request_message.meta.mode == "task-notification"
    assert request_message.meta.visibility == "hidden"
    assert request_message.meta.tags["request_id"] == "request-1"

    _, turns = JsonlTrajectoryStore(store_path).load(session_id)
    assert turns[0].trigger_metadata is not None
    assert turns[0].trigger_metadata.target_session_id == session_id
    assert turns[0].trigger_metadata.meta["request_id"] == "request-1"


@pytest.mark.asyncio
async def test_sdk_resume_rehydrates_atom_defined_trigger(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "custom-trigger-turns"
    provider = _StubProvider("custom-answer")
    session = await AgentSession.create(
        AgentSessionConfig(
            extensions=[(_CUSTOM_TRIGGER, {})],
            stream_fn=provider,
            model=_model(),
            trajectory_storage=_jsonl_storage(
                store_path,
                tmp_path / "custom-trigger-nodes",
            ),
        )
    )
    session.start()
    try:
        await session.push_trigger(CustomTrigger("first")).wait()
        session_id = session.session_id
    finally:
        await session.shutdown()

    resumed_provider = _StubProvider("resumed-answer")
    resumed = await AgentSession.resume(
        session_id,
        _jsonl_storage(
            store_path,
            tmp_path / "custom-trigger-nodes",
        ),
        config=AgentSessionConfig(
            extensions=[(_CUSTOM_TRIGGER, {})],
            stream_fn=resumed_provider,
            model=_model(),
        ),
    )
    try:
        await resumed.run("after-resume")
    finally:
        await resumed.shutdown()

    assert _text(resumed_provider.requests[0]) == [
        "custom:first",
        "custom-answer",
        "after-resume",
    ]
