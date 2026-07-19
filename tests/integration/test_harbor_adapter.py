"""SDK-user behavior contracts for the optional Harbor host adapter."""

from __future__ import annotations

import asyncio
import contextvars
import os
import re
import shutil
import signal
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
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
)
from agentm.core.abi.stream import AssistantStreamEvent, MessageEnd
from agentm.core.abi.tool import Tool
from agentm.storage.trajectory import JsonlTrajectoryStore
from agentm_harbor.external_agent import (
    SCENARIO,
    ExternalAgentMAgent,
    _load_scenario,
)
from agentm_harbor.harbor_ops import HarborOpsConfig, harbor_bindings
from agentm_harbor.human_interrupt import HumanInterruptServer
from harbor.environments.base import (
    BaseEnvironment,
    ExecResult as HarborExecResult,
    OutputStream,
)
from harbor.models.agent.context import AgentContext

_OutputCallback = Callable[[str, OutputStream], Awaitable[None]]


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


def _texts(messages: Sequence[AgentMessage]) -> list[str]:
    return [
        block.text
        for message in messages
        for block in message.content
        if isinstance(block, TextContent)
    ]


async def _send_interrupt(path: Path, text: str) -> str:
    reader, writer = await asyncio.open_unix_connection(str(path))
    try:
        writer.write(text.encode("utf-8"))
        writer.write_eof()
        await writer.drain()
        return (await reader.readline()).decode("utf-8").strip()
    finally:
        writer.close()
        await writer.wait_closed()


def _model() -> Model:
    return Model(
        id="harbor-stub",
        provider="stub",
        context_window=4096,
        max_output_tokens=512,
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
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd="/",
            scenario=SCENARIO,
            scenario_loader=_load_scenario,
            stream_fn=provider,
            model=_model(),
            environment_operations=operations,
            resource_writer=writer,
            trajectory_store=store,
        )
    )
    child: AgentSession | None = None
    with tempfile.TemporaryDirectory(prefix="agentm-harbor-") as inbox:
        server = HumanInterruptServer(
            session,
            inbox_root=Path(inbox),
        )
        await server.start()
        session.register_cleanup(server.stop)

        initial = asyncio.create_task(session.run("initial-request"))
        try:
            await asyncio.wait_for(provider.started.wait(), timeout=2.0)
            assert await _send_interrupt(server.path, "operator-feedback") == "ok"
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
            )
        ),
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    trajectory_path = tmp_path / "trajectory"
    fake = _FakeHarborEnvironment(tmp_path / "sandbox")
    agent = ExternalAgentMAgent(
        logs_dir=logs,
        model_name="harbor-profile",
        extra_env={
            "AGENTM_HOME": str(home),
            "AGENTM_TRAJECTORY_DIR": str(trajectory_path),
            "TRIAL_ONLY_VALUE": "trial-only-secret",
        },
    )
    context = AgentContext()
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
    assert "harbor-provider-completed" in _texts(
        (turns[0].rounds[0].response,),
    )
