# code-health: ignore-file[AM025] -- scenario adapter validates external command payloads
"""Harbor implementations of AgentM environment and workspace ports."""

from __future__ import annotations

import asyncio
import base64
import math
import posixpath
import shlex
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from agentm import (
    CancelSignal,
    EnvironmentRef,
    ExecResult,
)
from agentm.core.abi import (
    PathClass,
    WriterAuthor,
    WriteResult,
)
from harbor.environments.base import BaseEnvironment, OutputStream
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

_T = TypeVar("_T")
_CANCEL_GRACE_SECONDS = 15


class HarborOpsConfig(BaseModel):
    """Configuration for one Harbor-backed AgentM session."""

    model_config = ConfigDict(extra="forbid", strict=True)

    work_dir: str = "/"
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("work_dir")
    @classmethod
    def _absolute_work_dir(cls, value: str) -> str:
        normalized = posixpath.normpath(value)
        if not normalized.startswith("/"):
            raise ValueError("Harbor work_dir must be absolute")
        return normalized.rstrip("/") or "/"


def _abs(work_dir: str, path: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(work_dir, path))


def _command_script(
    cmd: str,
    *,
    stdin: bytes | None,
    log_path: str | None,
    work_dir: str,
    timeout: float | None,
) -> str:
    prelude: list[str] = ["set -o pipefail"]
    redirect = ""
    if stdin is not None:
        stdin_path = f"/tmp/agentm-stdin-{uuid.uuid4().hex}"
        encoded = base64.b64encode(stdin).decode("ascii")
        prelude.extend(
            (
                f"stdin_path={shlex.quote(stdin_path)}",
                "trap 'rm -f -- \"$stdin_path\"' EXIT",
                f'printf %s {shlex.quote(encoded)} | base64 -d > "$stdin_path"',
            )
        )
        redirect = ' < "$stdin_path"'

    command = f"(\n{cmd}\n){redirect}"
    if log_path is not None:
        resolved_log = _abs(work_dir, log_path)
        quoted_log = shlex.quote(resolved_log)
        prelude.append(f'mkdir -p -- "$(dirname -- {quoted_log})"')
        command = (
            f"{command} > >(tee -a {quoted_log} 2>/dev/null || cat) "
            f"2> >(tee -a {quoted_log} >&2 2>/dev/null || cat >&2)"
        )
    prelude.append(command)
    inner = "\n".join(prelude)
    shell = f"exec bash -lc {shlex.quote(inner)}"
    if timeout is not None:
        seconds = max(1, math.ceil(timeout))
        shell = (
            "if command -v timeout >/dev/null 2>&1; then "
            f"exec timeout -k 5s {seconds}s bash -lc {shlex.quote(inner)}; "
            f"else {shell}; fi"
        )
    return (
        "if command -v setsid >/dev/null 2>&1; then "
        f"exec setsid bash -lc {shlex.quote(shell)}; "
        f"else {shell}; fi"
    )


def _cancel_script(execution_id: str) -> str:
    marker = shlex.quote(f"AGENTM_EXEC_ID={execution_id}")
    return f"""\
set +e
marker={marker}
targets=""
for env_file in /proc/[0-9]*/environ; do
  pid="${{env_file#/proc/}}"
  pid="${{pid%/environ}}"
  case "$pid" in ''|*[!0-9]*) continue ;; esac
  if tr '\\000' '\\n' < "$env_file" 2>/dev/null | grep -qx -- "$marker"; then
    targets="$targets $pid"
  fi
done
for pid in $targets; do
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
done
sleep 1
for pid in $targets; do
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
done
"""


async def _await_uninterruptibly(task: asyncio.Task[_T]) -> _T:
    while not task.done():
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class HarborBashOperations:
    """Run shell commands inside one Harbor ``BaseEnvironment``."""

    __slots__ = ("_env", "_default_work_dir", "_default_timeout")

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        default_work_dir: str,
        default_timeout: float | None,
    ) -> None:
        self._env = env
        self._default_work_dir = default_work_dir
        self._default_timeout = default_timeout

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
        if not isinstance(cmd, str):
            raise TypeError("Harbor command must be a string")
        if not isinstance(cwd, str) or not cwd:
            raise TypeError("Harbor command cwd must be a non-empty string")
        if signal is not None and signal.is_set():
            raise asyncio.CancelledError("Harbor command interrupted before start")
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout is not None and (
            not math.isfinite(effective_timeout) or effective_timeout <= 0
        ):
            raise ValueError("Harbor command timeout must be finite and positive")

        execution_id = uuid.uuid4().hex
        environment = dict(env or {})
        environment["AGENTM_EXEC_ID"] = execution_id
        work_dir = _abs(self._default_work_dir, cwd)
        command = _command_script(
            cmd,
            stdin=stdin,
            log_path=log_path,
            work_dir=work_dir,
            timeout=effective_timeout,
        )
        host_timeout = (
            math.ceil(effective_timeout) + _CANCEL_GRACE_SECONDS
            if effective_timeout is not None
            else None
        )
        streamed_stdout = False

        async def stream_output(text: str, stream: OutputStream) -> None:
            nonlocal streamed_stdout
            if on_data is not None and stream == "stdout" and text:
                streamed_stdout = True
                on_data(text.encode("utf-8"))

        with self._env.scoped_output_callback(stream_output if on_data is not None else None):
            execution = asyncio.create_task(
                self._env.exec(
                    command=command,
                    cwd=work_dir,
                    env=environment,
                    timeout_sec=host_timeout,
                ),
                name=f"agentm-harbor-exec-{execution_id}",
            )
            signal_task = (
                asyncio.create_task(
                    signal.wait(),
                    name=f"agentm-harbor-signal-{execution_id}",
                )
                if signal is not None
                else None
            )
            cleaned = False
            try:
                wait_set: set[asyncio.Task[object]] = {execution}
                if signal_task is not None:
                    wait_set.add(signal_task)
                done, _ = await asyncio.wait(
                    wait_set,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if signal_task is not None and signal_task in done and not execution.done():
                    await self._cancel_and_reap(
                        execution,
                        execution_id=execution_id,
                        work_dir=work_dir,
                    )
                    cleaned = True
                    raise asyncio.CancelledError("Harbor command interrupted")
                result = await asyncio.shield(execution)
            except asyncio.CancelledError:
                if not cleaned and not execution.done():
                    await self._cancel_and_reap(
                        execution,
                        execution_id=execution_id,
                        work_dir=work_dir,
                    )
                raise
            finally:
                if signal_task is not None and not signal_task.done():
                    signal_task.cancel()
                    await asyncio.gather(signal_task, return_exceptions=True)

        stdout = (result.stdout or "").encode("utf-8")
        if on_data is not None and stdout and not streamed_stdout:
            on_data(stdout)
        return ExecResult(
            exit_code=result.return_code,
            stdout=stdout,
            stderr=(result.stderr or "").encode("utf-8"),
            timed_out=(effective_timeout is not None and result.return_code in {124, 137}),
        )

    async def _cancel_and_reap(
        self,
        execution: asyncio.Task[object],
        *,
        execution_id: str,
        work_dir: str,
    ) -> None:
        cancel_task = asyncio.create_task(
            self._env.exec(
                command=_cancel_script(execution_id),
                cwd=work_dir,
                timeout_sec=_CANCEL_GRACE_SECONDS,
            ),
            name=f"agentm-harbor-kill-{execution_id}",
        )
        try:
            await _await_uninterruptibly(cancel_task)
        except Exception as exc:
            logger.warning(
                "Harbor could not send cancellation for execution {}: {}",
                execution_id,
                exc,
            )
        try:
            await _await_uninterruptibly(execution)
        except BaseException as exc:
            logger.debug(
                "Harbor execution {} ended during cancellation: {}",
                execution_id,
                exc,
            )


class HarborEnvironmentOperations:
    """Environment identity and shell operations supplied by Harbor."""

    __slots__ = ("_bash", "_ref")

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        work_dir: str,
        timeout: float | None = None,
    ) -> None:
        self._bash = HarborBashOperations(
            env,
            default_work_dir=work_dir,
            default_timeout=timeout,
        )
        metadata: dict[str, str] = {
            "cwd": work_dir,
            "harbor_environment_name": env.environment_name,
            "harbor_environment_id": env.environment_id,
        }
        if env.context_id is not None:
            metadata["harbor_context_id"] = str(env.context_id)
        self._ref = EnvironmentRef(
            id=env.session_id,
            kind="sandbox",
            metadata=metadata,
        )

    @property
    def ref(self) -> EnvironmentRef:
        return self._ref

    @property
    def bash(self) -> HarborBashOperations:
        return self._bash

    async def snapshot(self) -> str | None:
        return None

    async def close(self) -> None:
        # Harbor owns the environment lifetime around the agent invocation.
        return None


class HarborResourceWriter:
    """Read and mutate workspace files through Harbor transfer APIs."""

    __slots__ = ("_env", "_work_dir")

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        work_dir: str,
    ) -> None:
        self._env = env
        self._work_dir = work_dir

    def _resolve(self, path: str) -> str:
        return _abs(self._work_dir, path)

    def classify(self, path: str) -> PathClass:
        del path
        return "managed"

    async def read(self, path: str) -> bytes:
        remote = self._resolve(path)
        with tempfile.NamedTemporaryFile() as target:
            try:
                await self._env.download_file(remote, target.name)
            except Exception as exc:
                message = str(exc).lower()
                if "no such file" in message or "not found" in message:
                    raise FileNotFoundError(path) from exc
                raise
            return Path(target.name).read_bytes()

    async def exists(self, path: str) -> bool:
        result = await self._env.exec(
            command=f"test -e {shlex.quote(self._resolve(path))}",
            cwd=self._work_dir,
        )
        return result.return_code == 0

    async def list_dir(self, path: str) -> list[str]:
        result = await self._env.exec(
            command=f"ls -1A -- {shlex.quote(self._resolve(path))}",
            cwd=self._work_dir,
        )
        if result.return_code != 0:
            error = result.stderr or ""
            if "no such file" in error.lower():
                raise FileNotFoundError(path)
            raise RuntimeError(f"list {path!r} failed: {error}")
        return sorted(line for line in (result.stdout or "").splitlines() if line)

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        remote = self._resolve(path)
        try:
            with tempfile.NamedTemporaryFile() as source:
                source.write(content)
                source.flush()
                await self._env.upload_file(source.name, remote)
        except Exception as exc:
            logger.warning("Harbor write failed for {}: {}", remote, exc)
            return WriteResult(path=path, path_class="managed", error=str(exc))
        return WriteResult(path=path, path_class="managed")

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        try:
            current = await self.read(path)
        except FileNotFoundError as exc:
            return WriteResult(path=path, path_class="managed", error=str(exc))
        if current != old:
            return WriteResult(
                path=path,
                path_class="managed",
                error=f"replace precondition failed for {path!r}",
            )
        return await self.write(
            path,
            new,
            rationale=rationale,
            author=author,
        )

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        result = await self._env.exec(
            command=f"rm -f -- {shlex.quote(self._resolve(path))}",
            cwd=self._work_dir,
        )
        if result.return_code != 0:
            return WriteResult(
                path=path,
                path_class="managed",
                error=result.stderr or "delete failed",
            )
        return WriteResult(path=path, path_class="managed")


def harbor_bindings(
    env: BaseEnvironment,
    config: HarborOpsConfig,
) -> tuple[HarborEnvironmentOperations, HarborResourceWriter]:
    """Build the two SDK host ports for one injected Harbor environment."""

    operations = HarborEnvironmentOperations(
        env,
        work_dir=config.work_dir,
        timeout=config.timeout,
    )
    writer = HarborResourceWriter(env, work_dir=config.work_dir)
    return operations, writer


__all__ = (
    "HarborBashOperations",
    "HarborEnvironmentOperations",
    "HarborOpsConfig",
    "HarborResourceWriter",
    "harbor_bindings",
)
