"""Tool executor fail-stop behavior."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    TOOL_EXECUTION_DOMAIN_METADATA_KEY,
    TOOL_EXECUTION_DOMAIN_PROCESS,
    TextContent,
    ToolProcessTerminated,
    ToolResult,
    execute_tool_call,
)


class _BlockingProcessTool:
    name = "blocking_process"
    description = "Block inside a child process until the parent kills it."
    parameters: dict[str, Any] = {}
    metadata = {
        TOOL_EXECUTION_DOMAIN_METADATA_KEY: TOOL_EXECUTION_DOMAIN_PROCESS,
    }

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult:
        del signal
        Path(str(args["pid_path"])).write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(float(args["sleep_seconds"]))
        Path(str(args["done_path"])).write_text("done", encoding="utf-8")
        return ToolResult(
            content=[TextContent(type="text", text="finished")],
            is_error=False,
        )


@pytest.mark.asyncio
async def test_process_tool_does_not_block_event_loop_and_kills_on_signal(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "child.pid"
    done_path = tmp_path / "child.done"
    signal = asyncio.Event()
    started_at = time.monotonic()

    task = asyncio.create_task(
        execute_tool_call(
            _BlockingProcessTool(),
            {
                "pid_path": str(pid_path),
                "done_path": str(done_path),
                "sleep_seconds": 5.0,
            },
            signal=signal,
        )
    )

    try:
        await asyncio.wait_for(_wait_for_file(pid_path), timeout=1.0)
        # If the process domain regresses to inline/event-loop execution, this
        # sleep cannot complete until the blocking tool returns.
        await asyncio.wait_for(asyncio.sleep(0.05), timeout=0.25)
        assert time.monotonic() - started_at < 1.0

        pid = int(pid_path.read_text(encoding="utf-8"))
        signal.set()

        with pytest.raises(ToolProcessTerminated):
            await asyncio.wait_for(task, timeout=2.0)

        assert not _pid_exists(pid)
        assert not done_path.exists()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _wait_for_file(path: Path) -> None:
    while not path.exists():
        await asyncio.sleep(0.01)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
