"""SDK-user behavior contracts for the optional Harbor host adapter."""

from __future__ import annotations

import asyncio
import contextvars
import os
import re
import shlex
import shutil
import signal
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pytest

pytest.importorskip("harbor")
pytest.importorskip("agentm_harbor")

from agentm import AgentSession, AgentSessionConfig, Model
from agentm.core.abi.cancel import CancelSignal, EventCancelSource
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.stream import AssistantStreamEvent, MessageEnd
from agentm.core.abi.tool import Tool
from agentm.control import SessionControlServer, send_interrupt
from agentm.storage.resources import LocalResourceStore
from agentm.storage.trajectory import JsonlTrajectoryStore
from agentm_harbor.external_agent import (
    SCENARIO,
    ExternalAgentMAgent,
    _load_scenario,
    _provision_remote_toolbox,
)
from agentm_harbor.harbor_ops import HarborOpsConfig, harbor_bindings
from harbor.environments.base import (
    BaseEnvironment,
    ExecResult as HarborExecResult,
    OutputStream,
)
from harbor.models.agent.context import AgentContext

_OutputCallback = Callable[[str, OutputStream], Awaitable[None]]


@dataclass
class _FakeArlStep:
    step_index: int


@dataclass
class _FakeArlInfo:
    session_id: str
    parent_session_id: str = ""
    fork_step: int = 0
    steps: list[_FakeArlStep] = field(default_factory=list)


class _FakeHarborEnvironment:
    """Local process double for Harbor's public environment boundary."""

    environment_name = "fake-harbor"
    environment_id = "fake-harbor-id"
    session_id = "fake-harbor-session"
    context_id = None

    def __init__(self, root: Path) -> None:
        self._root = root
        self._callback: contextvars.ContextVar[_OutputCallback | None] = (
            contextvars.ContextVar("fake_harbor_callback", default=None)
        )
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self.exec_started = asyncio.Event()
        self.exec_cwds: list[str | None] = []

    def _local_path(self, remote: str) -> Path:
        return self._root / remote.lstrip("/")

    @contextmanager
    def scoped_output_callback(
        self,
        callback: _OutputCallback | None,
    ) -> Iterator[None]:
        token = self._callback.set(callback)
        try:
            yield
        finally:
            self._callback.reset(token)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> HarborExecResult:
        del timeout_sec
        self.exec_cwds.append(cwd)
        command_parts = shlex.split(command)
        if len(command_parts) == 3 and command_parts[:2] == ["test", "-e"]:
            exists = self._local_path(command_parts[2]).exists()
            return HarborExecResult(
                stdout="",
                stderr="",
                return_code=0 if exists else 1,
            )
        cancel_match = re.search(
            r"^marker=['\"]?AGENTM_EXEC_ID=([0-9a-f]+)",
            command,
            flags=re.MULTILINE,
        )
        if cancel_match is not None:
            process = self._processes.get(cancel_match.group(1))
            if process is not None and process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            return HarborExecResult(stdout="", stderr="", return_code=0)

        local_cwd = self._local_path(cwd or "/")
        local_cwd.mkdir(parents=True, exist_ok=True)
        process_env = dict(os.environ)
        process_env.update(env or {})
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=local_cwd,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        execution_id = process_env.get("AGENTM_EXEC_ID")
        if execution_id is not None:
            self._processes[execution_id] = process
        self.exec_started.set()
        try:
            stdout, stderr = await process.communicate()
        finally:
            if execution_id is not None:
                self._processes.pop(execution_id, None)

        callback = self._callback.get()
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if callback is not None:
            if stdout_text:
                await callback(stdout_text, "stdout")
            if stderr_text:
                await callback(stderr_text, "stderr")
        return HarborExecResult(
            stdout=stdout_text,
            stderr=stderr_text,
            return_code=process.returncode,
        )

    async def upload_file(self, source_path: str, target_path: str) -> None:
        target = self._local_path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    async def download_file(self, source_path: str, target_path: str) -> None:
        shutil.copyfile(self._local_path(source_path), target_path)

    @property
    def has_running_processes(self) -> bool:
        return bool(self._processes)


class _InterruptProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[AgentMessage, ...]] = []
        self.started = asyncio.Event()
        self.cancel_reason: str | None = None

    async def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: Literal["off", "low", "medium", "high"] = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del model, tools, system, thinking
        self.requests.append(tuple(messages))
        texts = _texts(messages)
        if "child-request" in texts:
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=(
                        TextContent(
                            type="text",
                            text="child-completed",
                        ),
                    ),
                    timestamp=0.0,
                    stop_reason="end_turn",
                )
            )
            return
        if not any("operator-feedback" in text for text in texts):
            if signal is None:
                raise AssertionError("provider request has no cancellation signal")
            self.started.set()
            await signal.wait()
            self.cancel_reason = signal.reason
            raise RuntimeError("provider request cancelled")

        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=(
                    TextContent(
                        type="text",
                        text="feedback-applied",
                    ),
                ),
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )


class _FileToolProvider:
    """Drive a read/edit workflow through the public provider boundary."""

    def __init__(self) -> None:
        self.requests: list[tuple[AgentMessage, ...]] = []
        self._step = 0

    async def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: Literal["off", "low", "medium", "high"] = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del model, tools, system, signal, thinking
        self.requests.append(tuple(messages))
        content: tuple[ToolCallBlock | TextContent, ...]
        if self._step == 0:
            content = (
                ToolCallBlock(
                    type="tool_call",
                    id="read-note",
                    name="read",
                    arguments={"path": "note.txt"},
                ),
            )
            stop_reason = "tool_use"
        elif self._step == 1:
            content = (
                ToolCallBlock(
                    type="tool_call",
                    id="edit-note",
                    name="edit",
                    arguments={
                        "path": "note.txt",
                        "old_string": "before",
                        "new_string": "after",
                    },
                ),
            )
            stop_reason = "tool_use"
        else:
            content = (TextContent(type="text", text="remote-file-updated"),)
            stop_reason = "end_turn"
        self._step += 1
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=content,
                timestamp=0.0,
                stop_reason=stop_reason,
            )
        )


def _texts(messages: Sequence[AgentMessage]) -> list[str]:
    return [
        block.text
        for message in messages
        for block in message.content
        if isinstance(block, TextContent)
    ]


def _model() -> Model:
    return Model(
        id="harbor-stub",
        provider="stub",
        context_window=262144,
        max_output_tokens=512,
    )


def test_harbor_scenario_configures_policy_repository_scan() -> None:
    spec = _load_scenario(SCENARIO)
    policy = next(
        extension
        for extension in spec.extensions
        if extension.module_path == "policy_engine"
    )

    assert dict(policy.config) == {
        "policy_files": ("package:ifg_evidence.yaml",),
        "db_session_scoped": True,
        "ifg_realtime": True,
        "ifg_defer_projection": True,
        "ifg_repository_scan": True,
        "ifg_repository_search_roots": ("/repo",),
        "ifg_repository_worker_command": (
            "PYTHONPATH=/opt/agentm-toolbox python3 -m agentm_toolbox"
        ),
        "ifg_repository_remote_db_path": (
            "/logs/artifacts/agentm/policy/repository-index.sqlite"
        ),
    }


@pytest.mark.asyncio
async def test_harbor_setup_provisions_remote_toolbox_dependencies() -> None:
    class SetupEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.uploads: list[tuple[Path, str]] = []

        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            timeout_sec: int | None = None,
        ) -> HarborExecResult:
            assert cwd == "/"
            assert timeout_sec == 300
            self.commands.append(command)
            return HarborExecResult(stdout="", stderr="", return_code=0)

        async def upload_dir(self, source: Path, target: str) -> None:
            self.uploads.append((source, target))

    environment = SetupEnvironment()
    await _provision_remote_toolbox(environment)  # type: ignore[arg-type]

    assert len(environment.commands) == 3
    command, prepare, verify = environment.commands
    assert "command -v ast-grep" in command
    assert "python3 -m pip install --help" in command
    assert "python3 -m pip install" in command
    assert "--break-system-packages" in command
    assert "ast-grep-cli==0.44.1" in command
    assert 'ln -sf "$agentm_scripts_dir"/ast-grep /usr/local/bin/ast-grep' in command
    assert prepare == "mkdir -p -- /opt/agentm-toolbox"
    assert environment.uploads[0][0].name == "agentm_toolbox"
    assert environment.uploads[0][1] == "/opt/agentm-toolbox/agentm_toolbox"
    assert verify == (
        "PYTHONPATH=/opt/agentm-toolbox python3 -m agentm_toolbox "
        "repository-index --help >/dev/null"
    )


@pytest.mark.asyncio
async def test_harbor_operations_honor_stdin_cwd_and_cancellation(
    tmp_path: Path,
) -> None:
    fake = _FakeHarborEnvironment(tmp_path / "sandbox")
    operations, _ = harbor_bindings(
        cast(BaseEnvironment, fake),
        HarborOpsConfig(work_dir="/workspace"),
    )

    streamed: list[bytes] = []
    result = await operations.bash.exec(
        "cat",
        cwd="/task",
        stdin=b"stdin-payload",
        on_data=streamed.append,
    )
    assert result.exit_code == 0
    assert result.stdout == b"stdin-payload"
    assert b"".join(streamed) == b"stdin-payload"
    assert fake.exec_cwds[-1] == "/task"

    fake.exec_started.clear()
    cancellation = EventCancelSource()
    running = asyncio.create_task(
        operations.bash.exec(
            "sleep 30",
            cwd="/task",
            signal=cancellation,
        )
    )
    await asyncio.wait_for(fake.exec_started.wait(), timeout=2.0)
    cancellation.set("task_stop")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=3.0)
    assert not fake.has_running_processes


@pytest.mark.asyncio
async def test_harbor_file_tools_share_resource_backed_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    sandbox = tmp_path / "sandbox"
    workspace = sandbox / "workspace"
    workspace.mkdir(parents=True)
    note = workspace / "note.txt"
    note.write_text("before\n")
    fake = _FakeHarborEnvironment(sandbox)
    operations, writer = harbor_bindings(
        cast(BaseEnvironment, fake),
        HarborOpsConfig(work_dir="/workspace"),
    )
    provider = _FileToolProvider()
    resources = LocalResourceStore(
        workspace_root=tmp_path,
        root=tmp_path / "resources",
        discover_manifest=False,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd="/workspace",
            scenario=SCENARIO,
            scenario_loader=_load_scenario,
            stream_fn=provider,
            model=_model(),
            environment_operations=operations,
            resource_store=resources,
            resource_writer=writer,
            trajectory_store=JsonlTrajectoryStore(tmp_path / "trajectory"),
        )
    )
    try:
        transcript = await session.run("update the remote note")
    finally:
        await session.shutdown()

    assert note.read_text() == "after\n"
    assert "remote-file-updated" in _texts(transcript)
    tool_results = [
        block
        for message in provider.requests[-1]
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert [result.tool_call_id for result in tool_results] == [
        "read-note",
        "edit-note",
    ]
    assert all(not result.is_error for result in tool_results)


@pytest.mark.asyncio
async def test_harbor_host_interrupts_through_public_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    fake = _FakeHarborEnvironment(tmp_path / "sandbox")
    operations, writer = harbor_bindings(
        cast(BaseEnvironment, fake),
        HarborOpsConfig(work_dir="/"),
    )
    provider = _InterruptProvider()
    store = JsonlTrajectoryStore(tmp_path / "trajectory")
    resources = LocalResourceStore(
        workspace_root=tmp_path,
        root=tmp_path / "resources",
        discover_manifest=False,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd="/",
            scenario=SCENARIO,
            scenario_loader=_load_scenario,
            stream_fn=provider,
            model=_model(),
            environment_operations=operations,
            resource_store=resources,
            resource_writer=writer,
            trajectory_store=store,
        )
    )
    child: AgentSession | None = None
    with tempfile.TemporaryDirectory(prefix="agentm-harbor-") as inbox:
        server = SessionControlServer(
            session,
            inbox_root=Path(inbox),
        )
        await server.start()
        session.register_cleanup(server.stop)

        initial = asyncio.create_task(session.run("initial-request"))
        try:
            await asyncio.wait_for(provider.started.wait(), timeout=2.0)
            await send_interrupt(
                session.session_id,
                "operator-feedback",
                inbox_root=Path(inbox),
            )
            await asyncio.wait_for(initial, timeout=3.0)
            assert await session.idle(timeout=3.0)
            child = await session.spawn(purpose="harbor-child")
            child_transcript = await child.run("child-request")
            assert "child-completed" in _texts(child_transcript)
        finally:
            if not initial.done():
                initial.cancel()
            if child is not None:
                await child.shutdown()
            await session.shutdown()

    assert provider.cancel_reason == "submit_interrupt"
    assert any("operator-feedback" in _texts(request) for request in provider.requests)
    _, turns = store.load(session.session_id)
    assert len(turns) == 2
    assert store.load_checkpoint(session.session_id) is None


@pytest.mark.asyncio
async def test_harbor_external_agent_uses_isolated_resolver_inputs(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        "\n".join(
            (
                "[models.harbor-profile]",
                'provider = "tests.fixtures.harbor_provider"',
                'model = "stub-model"',
                "",
                "[atoms.llm_compaction]",
                "reserve_tokens = 1000",
            )
        ),
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    trajectory_path = tmp_path / "trajectory"
    fake = _FakeHarborEnvironment(tmp_path / "sandbox")
    fake.arl = _FakeArlInfo(
        session_id="arl-current",
        parent_session_id="arl-parent",
        fork_step=7,
        steps=[_FakeArlStep(step_index=11)],
    )
    agent = ExternalAgentMAgent(
        logs_dir=logs,
        model_name="harbor-profile",
        extra_env={
            "AGENTM_HOME": str(home),
            "AGENTM_TRAJECTORY_DIR": str(trajectory_path),
            "TRIAL_ONLY_VALUE": "trial-only-secret",
        },
    )
    context = AgentContext(metadata={"trial_key": "preserved"})
    process_env_before = dict(os.environ)

    await agent.run(
        "complete-through-harbor-host",
        cast(BaseEnvironment, fake),
        context,
    )

    assert dict(os.environ) == process_env_before
    store = JsonlTrajectoryStore(trajectory_path)
    sessions = store.list_sessions()
    assert len(sessions) == 1
    _, turns = store.load(sessions[0].id)
    assert len(turns) == 1
    assert turns[0].response is not None
    assert "harbor-provider-completed" in _texts((turns[0].response,))
    assert context.metadata == {
        "trial_key": "preserved",
        "agentm_session_id": sessions[0].id,
        "arl_session_id": "arl-current",
        "arl_parent_session_id": "arl-parent",
        "arl_fork_step": 7,
        "arl_step": 11,
    }

    await agent.resume(
        "continue-through-harbor-host",
        cast(BaseEnvironment, fake),
        context,
    )

    _, resumed_turns = store.load(sessions[0].id)
    assert len(resumed_turns) == 2
    assert resumed_turns[-1].response is not None
    assert "harbor-provider-completed" in _texts((resumed_turns[-1].response,))
