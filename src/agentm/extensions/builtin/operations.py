"""Local operations atom — registers shell execution services."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from signal import SIGKILL
from typing import IO, Any

from loguru import logger

from pydantic import BaseModel

from agentm.core.abi import (
    AtomInstallPriority,
    CancelSignal,
    EnvironmentRef,
    ExecResult,
)
from agentm.extensions import ExtensionManifest


class OperationsConfig(BaseModel):
    """Configuration placeholder for the local operations backend."""


MANIFEST = ExtensionManifest(
    name="operations",
    description="Registers local shell operations for SDK sessions.",
    registers=(),
    config_schema=OperationsConfig,
    requires=(),
    priority=AtomInstallPriority.SERVICE,
)


class LocalBashOperations:
    """Default shell implementation backed by ``asyncio`` subprocesses."""

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        on_data: Callable[[bytes], None] | None = None,
        signal: CancelSignal | None = None,
        log_path: str | None = None,
    ) -> ExecResult:
        process = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        log_file: IO[bytes] | None = None
        if log_path is not None:
            try:
                resolved = Path(log_path)
                if not resolved.is_absolute():
                    resolved = Path(cwd) / resolved
                resolved.parent.mkdir(parents=True, exist_ok=True)
                log_file = resolved.open("ab")
            except OSError as exc:
                logger.debug("local bash: cannot open log {}: {}", log_path, exc)
                log_file = None

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
                if log_file is not None:
                    try:
                        log_file.write(chunk)
                        log_file.flush()
                    except OSError:
                        pass
                if callback is not None:
                    callback(chunk)

        stdout_task = asyncio.create_task(
            _read_stream(process.stdout, stdout_chunks, on_data)
        )
        stderr_task = asyncio.create_task(_read_stream(process.stderr, stderr_chunks))
        stdin_task: asyncio.Task[None] | None = None
        if stdin is not None:
            assert process.stdin is not None
            stdin_task = asyncio.create_task(_write_stdin(process.stdin, stdin))

        signal_task: asyncio.Task[object] | None = None
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
            if stdin_task is not None:
                await asyncio.gather(stdin_task, return_exceptions=True)
            await asyncio.gather(stdout_task, stderr_task)
            if log_file is not None:
                try:
                    log_file.close()
                except OSError:
                    pass

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


class LocalEnvironmentOperations:
    """Default local process environment backend."""

    __slots__ = ("_bash", "_ref")

    def __init__(self, *, cwd: str, bash: LocalBashOperations | None = None) -> None:
        resolved_cwd = str(Path(cwd or os.getcwd()).resolve())
        self._bash = bash if bash is not None else LocalBashOperations()
        self._ref = EnvironmentRef(
            id=f"local:{resolved_cwd}",
            kind="local",
            metadata={"cwd": resolved_cwd},
        )

    @property
    def ref(self) -> EnvironmentRef:
        return self._ref

    @property
    def bash(self) -> LocalBashOperations:
        return self._bash

    async def snapshot(self) -> str | None:
        return None

    async def close(self) -> None:
        return None


async def _write_stdin(writer: asyncio.StreamWriter, payload: bytes) -> None:
    try:
        writer.write(payload)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            pass


def install(session: Any, config: OperationsConfig) -> None:
    del config
    bash = LocalBashOperations()
    cwd = getattr(getattr(session, "ctx", None), "cwd", "") or os.getcwd()
    environment = LocalEnvironmentOperations(cwd=cwd, bash=bash)
    session.register_operations(environment=environment, bash=bash)


__all__ = (
    "MANIFEST",
    "LocalEnvironmentOperations",
    "OperationsConfig",
    "install",
)
