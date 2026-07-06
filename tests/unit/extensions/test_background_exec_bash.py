"""background_exec cancellation over real nested bash processes."""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any, cast

import pytest

from agentm.core.abi import (
    AgentStartEvent,
    ExtensionAPI,
    TextContent,
    ToolResult,
)
from agentm.extensions.builtin import background_exec
from agentm.extensions.builtin.bash.local import LocalBashOperations
from tests.unit.extensions._fake_api import FakeExtensionAPI


class _ShellTool:
    name = "bash"
    description = "Execute a shell command in the test cwd."
    parameters: dict[str, Any] = {}

    def __init__(self, cwd: Path) -> None:
        self._cwd = str(cwd)
        self._bash = LocalBashOperations()

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult:
        result = await self._bash.exec(
            str(args["cmd"]),
            cwd=self._cwd,
            timeout=float(args.get("timeout", 60.0)),
            signal=signal,
        )
        text = (
            f"exit={result.exit_code} timed_out={result.timed_out} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=result.exit_code != 0 or result.timed_out,
        )


@pytest.mark.asyncio
async def test_backgrounded_nested_bash_wait_timeout_then_cancel_kills_child(
    tmp_path: Path,
) -> None:
    api = FakeExtensionAPI()
    api.tools.append(_ShellTool(tmp_path))
    background_exec.install(
        cast(ExtensionAPI, api),
        background_exec.BackgroundExecConfig(
            timeout=0.1,
            heartbeat_interval=30.0,
            silence_warning=60.0,
            shutdown_grace_seconds=0.2,
        ),
    )
    api.fire(AgentStartEvent.CHANNEL, AgentStartEvent(messages=[]))

    bash_tool = _tool(api, "bash")
    wait_tool = _tool(api, "wait_background")
    cancel_tool = _tool(api, "cancel_background")
    check_tool = _tool(api, "check_background")
    tool_count = len(api.tools)

    pid_path = tmp_path / "nested-sleep.pid"
    done_path = tmp_path / "nested-sleep.done"
    cmd = (
        "sleep 30 & "
        f"echo $! > {shlex.quote(str(pid_path))}; "
        "wait $!; "
        f"echo done > {shlex.quote(str(done_path))}"
    )

    ticket = await bash_tool.execute({"cmd": cmd, "timeout": 60.0})
    task_id = str(ticket.extras["task_id"])
    assert ticket.extras["status"] == "running"
    await _wait_for_file(pid_path)
    nested_pid = int(pid_path.read_text(encoding="utf-8"))

    wait_result = await wait_tool.execute({"task_id": task_id, "timeout_s": 0.05})
    assert wait_result.extras["status"] == "running"
    assert "still running" in wait_result.extras["note"]
    assert len(api.tools) == tool_count

    check_result = await check_tool.execute({})
    tasks = check_result.extras["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == task_id
    assert tasks[0]["status"] == "running"

    cancel_result = await cancel_tool.execute({"task_id": task_id})
    assert cancel_result.extras == {"task_id": task_id, "status": "cancelling"}

    terminal = await _wait_for_terminal(check_tool, task_id)
    assert terminal["status"] == "cancelled"
    await _wait_for_pid_gone(nested_pid)
    assert not done_path.exists()
    assert await api.inbox.wait_no_pending_work(timeout=1.0)


def _tool(api: FakeExtensionAPI, name: str) -> Any:
    return next(tool for tool in api.tools if tool.name == name)


async def _wait_for_file(path: Path) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    while not path.exists():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        await asyncio.sleep(0.01)


async def _wait_for_terminal(tool: Any, task_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 2.0
    while True:
        result = await tool.execute({})
        tasks = result.extras["tasks"]
        assert len(tasks) == 1
        task = tasks[0]
        if task["task_id"] == task_id and task["status"] != "running":
            return cast(dict[str, Any], task)
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"task {task_id} did not become terminal")
        await asyncio.sleep(0.02)


async def _wait_for_pid_gone(pid: int) -> None:
    deadline = asyncio.get_running_loop().time() + 2.0
    while _pid_exists(pid):
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"pid {pid} still exists")
        await asyncio.sleep(0.02)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
