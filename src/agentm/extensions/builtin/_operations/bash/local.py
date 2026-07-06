"""Local-stdlib implementation of ``BashOperations``.

Provides :func:`install_local` for use by the unified ``operations`` atom.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from signal import SIGKILL
from typing import Any

from agentm.core.abi import ExecResult, ExtensionAPI


class LocalBashOperations:
    """Default shell implementation backed by ``asyncio`` subprocesses."""

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_data: Callable[[bytes], None] | None = None,
        signal: asyncio.Event | None = None,
    ) -> ExecResult:
        process = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        timed_out = False

        async def _read_stream(
            stream: asyncio.StreamReader,
            sink: list[bytes],
            callback: Callable[[bytes], None] | None = None,
        ) -> None:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    return
                sink.append(chunk)
                if callback is not None:
                    callback(chunk)

        stdout_task = asyncio.create_task(
            _read_stream(process.stdout, stdout_chunks, on_data)
        )
        stderr_task = asyncio.create_task(_read_stream(process.stderr, stderr_chunks))

        signal_task: asyncio.Task[bool] | None = None
        if signal is not None:
            signal_task = asyncio.create_task(signal.wait())

        wait_task = asyncio.create_task(process.wait())
        try:
            done, _pending = await asyncio.wait(
                [task for task in (wait_task, signal_task) if task is not None],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if wait_task in done:
                await wait_task
            else:
                timed_out = (
                    timeout is not None
                    and signal_task not in done
                    and wait_task not in done
                )
                await self._terminate_process_group(process)
                await wait_task
        finally:
            if signal_task is not None and not signal_task.done():
                signal_task.cancel()
                await asyncio.gather(signal_task, return_exceptions=True)
            await asyncio.gather(stdout_task, stderr_task)

        return ExecResult(
            stdout=b"".join(stdout_chunks),
            stderr=b"".join(stderr_chunks),
            exit_code=process.returncode if process.returncode is not None else -SIGKILL,
            timed_out=timed_out,
        )

    async def _terminate_process_group(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode is not None:
            return

        pgid = process.pid
        if pgid is None:
            process.kill()
            return

        try:
            os.killpg(pgid, SIGKILL)
        except ProcessLookupError:
            return


def install_local(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    api.register_operations(bash=LocalBashOperations())
