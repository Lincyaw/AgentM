"""Builtin ``operations_local`` atom.

Registers the default :class:`agentm.core.abi.operations.Operations`
bundle: local-stdlib file I/O and ``asyncio``-subprocess shell exec.
See ``.claude/designs/pluggable-architecture.md`` §3.2 — Operations are
a pluggability axis like any other; the substrate no longer instantiates
a default bundle, so every scenario manifest that uses tool atoms must
list this atom (or an alternate ``operations_*`` atom such as an SSH /
sandbox / in-memory implementation).

§11 single-file contract: only stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*`` + ``agentm.core.abi.extension`` imports. No
atom-to-atom imports.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from signal import SIGKILL
from typing import Any

from agentm.core.abi.operations import ExecResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="operations_local",
    description=(
        "Registers the default local-FS / asyncio-subprocess Operations "
        "bundle. Listed first in every scenario that uses file/bash tools."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=(),
)


class LocalFileOperations:
    """Default filesystem implementation backed by local stdlib I/O.

    ``cwd`` anchors relative paths to the session's working directory.
    Without it, ``read("foo.py")`` resolves against the process cwd —
    typically wherever the operator launched ``agentm`` from — which
    diverges from the bash op's cwd handling and breaks any agent
    reasoning about its own workspace. ``cwd=None`` keeps the legacy
    "process cwd" behavior for tests and embedded uses that do not
    supply one.
    """

    def __init__(self, cwd: str | None = None) -> None:
        self._cwd = cwd

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or self._cwd is None:
            return candidate
        return Path(self._cwd) / candidate

    async def read_file(self, path: str) -> bytes:
        return await asyncio.to_thread(self._resolve(path).read_bytes)

    async def access(self, path: str) -> bool:
        def _access() -> bool:
            target = self._resolve(path)
            return target.exists() and os.access(target, os.R_OK)

        return await asyncio.to_thread(_access)

    async def is_dir(self, path: str) -> bool:
        return await asyncio.to_thread(self._resolve(path).is_dir)

    async def list_dir(self, path: str) -> list[str]:
        def _list_dir() -> list[str]:
            return sorted(entry.name for entry in self._resolve(path).iterdir())

        return await asyncio.to_thread(_list_dir)


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
                timed_out = wait_task not in done and timeout is not None
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    api.register_operations(
        file=LocalFileOperations(cwd=api.cwd),
        bash=LocalBashOperations(),
    )
