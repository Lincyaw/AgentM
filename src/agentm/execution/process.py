# code-health: ignore-file[AM025] -- execution worker and wire boundaries validate cross-process payloads
"""Process-isolated ``ToolExecutor`` for explicit entrypoint tools."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.tool import ToolMetadataProvider, ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import (
    ToolExecutionCapabilities,
    ToolExecutionRequest,
)
from agentm.execution.wire import decode_tool_output, encode_tool_arguments


PROCESS_ENTRYPOINT_METADATA_KEY = "process_entrypoint"


class ProcessToolExecutor:
    """Execute tools in a subprocess when they expose a module entrypoint.

    Generic in-memory ``Tool`` instances are not assumed to be pickleable or
    importable. A process-isolated tool must declare
    ``metadata['process_entrypoint'] = 'package.module:function'``. The
    function is imported in the worker process and called with the JSON args.
    """

    def capabilities(self) -> ToolExecutionCapabilities:
        return ToolExecutionCapabilities(
            isolation=("process",),
            filesystem=("none", "read", "write"),
            killable=True,
            network=True,
            concurrency=("exclusive", "parallel_safe"),
            interrupt=("cancel",),
        )

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        entrypoint = _process_entrypoint(request)
        if signal is not None and signal.is_set():
            raise asyncio.CancelledError("tool process interrupted before start")
        with tempfile.TemporaryDirectory(prefix="agentm-tool-") as tmp_dir_text:
            tmp_dir = Path(tmp_dir_text)
            args_path = tmp_dir / "args.json"
            result_path = tmp_dir / "result.json"
            args_path.write_text(
                encode_tool_arguments(request.args),
                encoding="utf-8",
            )
            env = dict(os.environ)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "agentm.execution.worker",
                entrypoint,
                str(args_path),
                str(result_path),
                cwd=request.cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            signal_task: asyncio.Task[object] | None = None
            wait_task = asyncio.create_task(
                process.communicate(),
                name=f"agentm-tool-process-{entrypoint}",
            )
            if signal is not None:
                signal_task = asyncio.create_task(
                    signal.wait(),
                    name=f"agentm-tool-process-signal-{entrypoint}",
                )
            try:
                done, _ = await asyncio.wait(
                    [task for task in (wait_task, signal_task) if task is not None],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if (
                    signal_task is not None
                    and signal_task in done
                    and not wait_task.done()
                ):
                    await _terminate_and_reap(process, wait_task)
                    raise asyncio.CancelledError("tool process interrupted")
                stdout, stderr = await wait_task
            except asyncio.CancelledError:
                await _terminate_and_reap(process, wait_task)
                raise
            finally:
                if signal_task is not None and not signal_task.done():
                    signal_task.cancel()
                    await asyncio.gather(signal_task, return_exceptions=True)
            if process.returncode != 0:
                return ToolResult(
                    content=[],
                    is_error=True,
                    extras={
                        "exit_code": process.returncode,
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )
            return decode_tool_output(result_path.read_text(encoding="utf-8"))


async def _terminate_and_reap(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    while not wait_task.done():
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            continue
    return wait_task.result()


def _process_entrypoint(request: ToolExecutionRequest) -> str:
    metadata = (
        request.tool.metadata if isinstance(request.tool, ToolMetadataProvider) else {}
    )
    entrypoint = metadata.get(PROCESS_ENTRYPOINT_METADATA_KEY)
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise RuntimeError(
            "process-isolated tools must declare metadata['process_entrypoint'] "
            "as 'module:function'"
        )
    return entrypoint


__all__ = ["PROCESS_ENTRYPOINT_METADATA_KEY", "ProcessToolExecutor"]
