from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.core.kernel import TextContent
from agentm.core.operations import ExecResult
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class RecordingBashOps:
    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str, float | None]] = []

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_data: object | None = None,
        signal: object | None = None,
    ) -> ExecResult:
        del env, on_data, signal
        self.calls.append((cmd, cwd, timeout))
        return self.result


@pytest.mark.asyncio
async def test_tool_bash_install_smoke(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_bash", {})],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    assert [tool.name for tool in session.tools] == ["bash"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_bash_executes_via_bash_ops(tmp_path: Path) -> None:
    bash_ops = RecordingBashOps(
        ExecResult(stdout=b"ok\n", stderr=b"", exit_code=0, timed_out=False)
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_bash", {"bash_ops": bash_ops})],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute({"cmd": "pwd", "timeout": 9.0})

    assert bash_ops.calls == [("pwd", str(tmp_path), 9.0)]
    assert not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text) == {
        "exit_code": 0,
        "stderr": "",
        "stdout": "ok\n",
        "timed_out": False,
    }
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_bash_returns_error_result_for_non_zero_exit(tmp_path: Path) -> None:
    bash_ops = RecordingBashOps(
        ExecResult(stdout=b"", stderr=b"boom", exit_code=7, timed_out=False)
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_bash", {"bash_ops": bash_ops})],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute({"cmd": "false"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["exit_code"] == 7
    await session.shutdown()
