"""ARL agent-env sandbox: ``FileOperations``, ``BashOperations``, ``ResourceWriter``.

One sandbox per AgentM session. All config comes from :class:`AgentEnvConfig`
(scenario manifest). Connection parameters (``gateway_url``, ``api_key``) fall
through to the ARL SDK's own config chain when unset.
"""

from __future__ import annotations

import asyncio
import posixpath
import shlex
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    BatchHandle,
    ExecResult,
    ExtensionAPI,
    PathClass,
    SessionShutdownEvent,
    WriteResult,
    WriterAuthor,
)

if TYPE_CHECKING:
    from arl import AsyncSandboxSession

_AGENT_ENV_SESSION_SERVICE = "agent_env.session_id"
_AGENT_ENV_EXPERIMENT_SERVICE = "agent_env.experiment_id"
_OPERATION_POLL_INTERVAL_SECONDS = 2.0
_OPERATION_TIMEOUT_GRACE_SECONDS = 30.0
_OPERATION_STATUS_DONE = "done"
_OPERATION_STATUS_ERROR = "error"


class AgentEnvConfig(BaseModel):
    image: str | None = None
    experiment_id: str | None = None
    attach_session: str | None = None
    gateway_url: str | None = None
    api_key: str | None = None
    profile: str | None = None
    config_env: dict[str, Any] | None = None
    work_dir: str | None = None
    timeout: float | None = None
    idle_timeout_seconds: int | None = None
    max_lifetime_seconds: int | None = None
    create_timeout: float | None = None
    cpu_request: str | None = None
    cpu_limit: str | None = None
    memory_request: str | None = None
    memory_limit: str | None = None
    delete_on_shutdown: bool | None = None


def _normalize_work_dir(work_dir: str) -> str:
    normalized = posixpath.normpath(work_dir)
    if not normalized.startswith("/"):
        raise ValueError(f"agent_env work_dir must be absolute, got {work_dir!r}")
    return normalized.rstrip("/") or "/"


def _sandbox_abs(work_dir: str, path: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(work_dir, path))


def _is_in_work_dir(work_dir: str, path: str) -> bool:
    abs_path = _sandbox_abs(work_dir, path)
    if work_dir == "/":
        return abs_path.startswith("/")
    return abs_path == work_dir or abs_path.startswith(work_dir + "/")


def _workspace_relative_path(work_dir: str, path: str) -> str | None:
    abs_path = _sandbox_abs(work_dir, path)
    if work_dir == "/":
        rel_path = abs_path.lstrip("/")
    elif abs_path.startswith(work_dir + "/"):
        rel_path = abs_path[len(work_dir) + 1 :]
    else:
        return None
    return rel_path or None


def _step_timeout_budget(steps: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for step in steps:
        raw = step.get("timeoutSeconds", step.get("timeout"))
        if not isinstance(raw, bool) and isinstance(raw, (int, float)) and raw > 0:
            values.append(float(raw))
    if not values:
        return None
    return max(values) + _OPERATION_TIMEOUT_GRACE_SECONDS


async def _recover_execute_operation(
    session: AsyncSandboxSession,
    steps: list[dict[str, Any]],
    operation_id: str,
    started_at: float,
) -> Any:
    budget = _step_timeout_budget(steps)
    deadline = started_at + budget if budget is not None else None

    while True:
        operation = await session.get_execute_operation(operation_id)
        if operation.result is not None:
            return operation.result

        status = (operation.status or "").lower()
        if status == _OPERATION_STATUS_ERROR:
            raise RuntimeError(
                operation.error or f"ARL execute operation {operation_id} failed"
            )
        if status == _OPERATION_STATUS_DONE:
            raise RuntimeError(
                f"ARL execute operation {operation_id} finished without result"
            )

        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"ARL execute operation {operation_id} still pending after "
                    f"{budget:.0f}s"
                )
            sleep_for = min(_OPERATION_POLL_INTERVAL_SECONDS, remaining)
        else:
            sleep_for = _OPERATION_POLL_INTERVAL_SECONDS
        await asyncio.sleep(sleep_for)


async def _async_execute(
    session: AsyncSandboxSession,
    steps: list[dict[str, Any]],
    *,
    on_output: Callable[[str, str], None] | None = None,
) -> Any:
    from arl import GatewayOperationTimeout  # type: ignore[import-not-found]

    started_at = time.monotonic()
    try:
        return await session.execute(steps, on_output=on_output)
    except GatewayOperationTimeout as exc:
        return await _recover_execute_operation(
            session,
            steps,
            exc.operation_id,
            started_at,
        )


# Versioning-token script. Emits ``<mtime_ns>-<sha16>`` for files up to
# ``_MTIME_TOKEN_SIZE_CAP`` bytes, ``<mtime_ns>-size<size>`` for larger
# files. The mtime component uses GNU stat's ``%.Y`` format (fractional
# seconds) so we get nanosecond resolution; we strip the decimal point so
# the token is a plain ``<digits>-<rest>`` string. Invoked as
# ``bash -lc <script> bash <path>`` — the trailing ``bash`` sets ``$0``
# so ``$1`` is the path argument.
_MTIME_TOKEN_SIZE_CAP = 16 * 1024 * 1024
_MTIME_TOKEN_SCRIPT = (
    "set -e; "
    'P="$1"; '
    'MNS=$(stat -c %.Y -- "$P" | tr -d .); '
    'SZ=$(stat -c %s -- "$P"); '
    f'if [ "$SZ" -le {_MTIME_TOKEN_SIZE_CAP} ]; then '
    '  H=$(sha256sum -- "$P" | cut -c1-16); '
    '  printf "%s-%s" "$MNS" "$H"; '
    "else "
    '  printf "%s-size%s" "$MNS" "$SZ"; '
    "fi"
)


class _AgentEnvBashOperations:
    """``BashOperations`` impl that executes commands inside an ARL sandbox.

    Each call wraps ``cmd`` as ``bash -lc <cmd>`` and dispatches a single
    step through ``AsyncSandboxSession.execute``.
    """

    def __init__(
        self,
        session: AsyncSandboxSession,
        *,
        default_work_dir: str,
        default_timeout: float | None,
    ) -> None:
        self._session = session
        self._default_work_dir = default_work_dir
        self._default_timeout = default_timeout

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
        effective_timeout = timeout if timeout is not None else self._default_timeout
        work_dir = cwd or self._default_work_dir
        timeout_seconds = (
            max(1, int(effective_timeout)) if effective_timeout is not None else None
        )
        exec_id = f"agentm-{uuid.uuid4().hex}"
        quoted_cmd = shlex.quote(cmd)
        runner = f"exec bash -lc {quoted_cmd}"
        if timeout_seconds is not None:
            runner = (
                "if command -v timeout >/dev/null 2>&1; then "
                f"exec timeout -k 5s {timeout_seconds}s bash -lc {quoted_cmd}; "
                "else "
                f"{runner}; "
                "fi"
            )
        command = [
            "bash",
            "-lc",
            (
                "if command -v setsid >/dev/null 2>&1; then "
                f"exec setsid bash -lc {shlex.quote(runner)}; "
                "else "
                f"{runner}; "
                "fi"
            ),
        ]
        step_env = dict(env or {})
        step_env["AGENTM_EXEC_ID"] = exec_id
        step: dict[str, Any] = {
            "name": "agentm_bash",
            "command": command,
            "work_dir": work_dir,
            "env": step_env,
        }
        if timeout_seconds is not None:
            step["timeout"] = timeout_seconds + 15

        timed_out = False
        streamed_stdout = False

        def _stream_output(stdout: str, _stderr: str) -> None:
            nonlocal streamed_stdout
            if on_data is not None and stdout:
                streamed_stdout = True
                on_data(stdout.encode("utf-8"))

        execute_task = asyncio.create_task(
            _async_execute(
                self._session,
                [step],
                on_output=_stream_output if on_data is not None else None,
            )
        )
        signal_task: asyncio.Task[bool] | None = None
        if signal is not None:
            signal_task = asyncio.create_task(signal.wait())
        try:
            wait_set: set[asyncio.Task[Any]] = {execute_task}
            if signal_task is not None:
                wait_set.add(signal_task)
            done, _pending = await asyncio.wait(
                wait_set, return_when=asyncio.FIRST_COMPLETED
            )
            if (
                signal_task is not None
                and signal_task in done
                and not execute_task.done()
            ):
                await self._cancel_remote_exec(exec_id, work_dir)
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(execute_task), timeout=20
                    )
                except TimeoutError:
                    execute_task.add_done_callback(_consume_task_result)
                    return ExecResult(
                        stdout=b"",
                        stderr=b"agent-env execute cancelled\n",
                        exit_code=130,
                        timed_out=True,
                    )
                timed_out = True
            else:
                response = await execute_task
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent-env execute failed: {}", exc)
            stderr = f"agent-env execute failed: {exc}".encode()
            timed_out = effective_timeout is not None and "timeout" in str(exc).lower()
            return ExecResult(
                stdout=b"", stderr=stderr, exit_code=124, timed_out=timed_out
            )
        finally:
            if signal_task is not None and not signal_task.done():
                signal_task.cancel()
                await asyncio.gather(signal_task, return_exceptions=True)

        if not response.results:
            return ExecResult(
                stdout=b"",
                stderr=b"agent-env returned no results",
                exit_code=1,
                timed_out=False,
            )

        result = response.results[0]
        stdout_bytes = result.output.stdout.encode("utf-8")
        stderr_bytes = result.output.stderr.encode("utf-8")
        if on_data is not None and stdout_bytes and not streamed_stdout:
            on_data(stdout_bytes)
        if timeout_seconds is not None and result.output.exit_code in {124, 137}:
            timed_out = True
        if signal is not None and signal.is_set():
            timed_out = True
        return ExecResult(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            exit_code=result.output.exit_code,
            timed_out=timed_out,
        )

    async def _cancel_remote_exec(self, exec_id: str, work_dir: str) -> None:
        marker = f"AGENTM_EXEC_ID={exec_id}"
        script = f"""
set +e
marker={shlex.quote(marker)}
targets=""
for envf in /proc/[0-9]*/environ; do
  pid="${{envf#/proc/}}"
  pid="${{pid%/environ}}"
  case "$pid" in ''|*[!0-9]*) continue ;; esac
  if tr '\\000' '\\n' < "$envf" 2>/dev/null | grep -qx -- "$marker"; then
    targets="$targets $pid"
  fi
done
for pid in $targets; do
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 2
for pid in $targets; do
  kill -KILL "$pid" 2>/dev/null || true
done
printf 'cancelled %s pids:%s\\n' "$marker" "$targets"
"""
        step = {
            "name": "agentm_cancel_bash",
            "command": ["bash", "-lc", script],
            "work_dir": work_dir,
            "timeout": 15,
        }
        try:
            await self._session.execute([step])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent_env: failed to cancel remote exec {}: {}", exec_id, exc
            )


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        logger.debug("agent_env: cancelled execute task finished late: {}", exc)



class _AgentEnvResourceWriter:
    """``ResourceWriter`` impl whose writes land inside the ARL sandbox.

    Boundary contract: only paths *inside* ``work_dir`` (after resolving
    relative paths against it) are writable; everything else — including
    every host path under the AgentM tree — is treated as constitution
    and refused. The sandbox cannot see the host filesystem, so this is
    fail-safe by construction: the agent literally cannot mutate its own
    code from a sandbox session.

    Read and write use the ARL SDK file API to bypass the ~8KB stdout limit
    on ``session.execute``.
    """

    def __init__(
        self,
        session: AsyncSandboxSession,
        *,
        work_dir: str,
        gateway_url: str,
        api_key: str | None,
        session_id: str,
    ) -> None:
        self._session = session
        self._work_dir = _normalize_work_dir(work_dir)
        self._gateway_url = gateway_url
        self._api_key = api_key
        self._session_id = session_id

    # --- path classification ---------------------------------------------

    def _resolve(self, path: str) -> str:
        return _sandbox_abs(self._work_dir, path)

    def _relative(self, path: str) -> str | None:
        return _workspace_relative_path(self._work_dir, path)

    def _in_sandbox(self, path: str) -> bool:
        return _is_in_work_dir(self._work_dir, path)

    def classify(self, path: str) -> PathClass:
        return "managed" if self._in_sandbox(path) else "constitution"

    # --- ARL plumbing -----------------------------------------------------

    async def _run(self, command: list[str]) -> tuple[bytes, bytes, int]:
        step = {
            "name": "agentm_resource_writer",
            "command": command,
            "work_dir": self._work_dir,
        }
        response = await _async_execute(self._session, [step])
        if not response.results:
            return b"", b"no result", 1
        out = response.results[0].output
        return out.stdout.encode("utf-8"), out.stderr.encode("utf-8"), out.exit_code

    async def _mtime_token(self, path: str) -> str | None:
        stdout, _stderr, code = await self._run(
            ["bash", "-lc", _MTIME_TOKEN_SCRIPT, "bash", path]
        )
        if code != 0:
            return None
        text = stdout.decode("utf-8", "replace").strip()
        return text or None

    def _refuse(self, path: str) -> WriteResult:
        return WriteResult(
            path=path,
            path_class="constitution",
            committed=False,
            commit_sha_before=None,
            commit_sha_after=None,
            error=(
                f"Refusing to write {path!r}: agent-env sandbox can only "
                f"modify paths inside {self._work_dir!r}. Constitution / "
                f"host paths are off-limits from inside the sandbox."
            ),
        )

    async def _write_bytes(self, rel_path: str, content: bytes) -> tuple[bool, str]:
        try:
            await self._session.upload_file(rel_path, content)
        except Exception as exc:
            logger.warning("agent-env file upload failed: {}", exc)
            return False, str(exc)
        return True, ""

    # --- ResourceWriter API ----------------------------------------------

    async def read(self, path: str) -> bytes:
        import base64 as b64

        rel_path = self._relative(path)
        if rel_path is not None:
            try:
                return await self._session.download_file(rel_path)
            except Exception as exc:
                raise FileNotFoundError(str(exc)) from exc
        abs_path = self._resolve(path)
        stdout, stderr, code = await self._run(
            ["bash", "-c", f"base64 -w0 -- {shlex.quote(abs_path)}"],
        )
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        encoded = stdout.strip()
        if not encoded:
            return b""
        return b64.b64decode(encoded)

    async def exists(self, path: str) -> bool:
        _stdout, _stderr, code = await self._run(
            ["test", "-r", self._resolve(path)]
        )
        return code == 0

    async def list_dir(self, path: str) -> list[str]:
        abs_path = self._resolve(path)
        stdout, stderr, code = await self._run(["ls", "-1A", "--", abs_path])
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        text = stdout.decode("utf-8", "replace").strip("\n")
        return sorted(line for line in text.split("\n") if line)

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        rel_path = self._relative(path)
        if rel_path is None:
            return self._refuse(path)
        abs_path = self._resolve(path)
        before = await self._mtime_token(abs_path)
        ok, err = await self._write_bytes(rel_path, content)
        if not ok:
            return WriteResult(
                path=path,
                path_class="managed",
                committed=False,
                commit_sha_before=before,
                commit_sha_after=None,
                error=err or "sandbox write failed",
            )
        after = await self._mtime_token(abs_path)
        return WriteResult(
            path=path,
            path_class="managed",
            committed=True,
            commit_sha_before=before,
            commit_sha_after=after,
        )

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale
        if self._relative(path) is None:
            return self._refuse(path)
        try:
            current = await self.read(path)
        except FileNotFoundError as exc:
            return WriteResult(
                path=path,
                path_class="managed",
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
                error=str(exc),
            )
        if current != old:
            return WriteResult(
                path=path,
                path_class="managed",
                committed=False,
                commit_sha_before=await self._mtime_token(self._resolve(path)),
                commit_sha_after=None,
                error=(
                    f"replace precondition failed for {path!r}: file content "
                    f"differs from the supplied 'old' value"
                ),
            )
        return await self.write(path, new, rationale="replace", author=author)

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        if self._relative(path) is None:
            return self._refuse(path)
        abs_path = self._resolve(path)
        before = await self._mtime_token(abs_path)
        _stdout, stderr, code = await self._run(["rm", "-f", "--", abs_path])
        if code != 0:
            return WriteResult(
                path=path,
                path_class="managed",
                committed=False,
                commit_sha_before=before,
                commit_sha_after=None,
                error=stderr.decode("utf-8", "replace") or "sandbox delete failed",
            )
        return WriteResult(
            path=path,
            path_class="managed",
            committed=True,
            commit_sha_before=before,
            commit_sha_after=None,
        )

    def restore(self, path: "Path", version: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "agent-env writer does not support per-file restore; use "
            "SandboxSession.restore(snapshot_id) for whole-step rollback."
        )

    def current_version_for_path(self, path: str) -> str | None:
        from arl import GatewayClient  # type: ignore[import-not-found]

        client = GatewayClient(
            base_url=self._gateway_url,
            api_key=self._api_key,
        )
        try:
            response = client.execute(
                self._session_id,
                [
                    {
                        "name": "agentm_stat",
                        "command": [
                            "bash",
                            "-lc",
                            _MTIME_TOKEN_SCRIPT,
                            "bash",
                            self._resolve(path),
                        ],
                        "work_dir": self._work_dir,
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("agent-env mtime token fetch failed: {}", exc)
            return None
        finally:
            client.close()
        if not response.results:
            return None
        out = response.results[0].output
        if out.exit_code != 0:
            return None
        token = out.stdout.strip()
        return token or None

    def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> "AbstractAsyncContextManager[BatchHandle]":
        del rationale

        writer = self

        class _Batch:
            def __init__(self) -> None:
                self._ops: list[tuple[str, tuple[Any, ...]]] = []

            async def write(self, path: str, content: bytes) -> None:
                self._ops.append(("write", (path, content)))

            async def replace(self, path: str, old: bytes, new: bytes) -> None:
                self._ops.append(("replace", (path, old, new)))

            async def delete(self, path: str) -> None:
                self._ops.append(("delete", (path,)))

            async def flush(self) -> None:
                for kind, args in self._ops:
                    if kind == "write":
                        result = await writer.write(
                            args[0], args[1], rationale="batch", author=author
                        )
                    elif kind == "replace":
                        result = await writer.replace(
                            args[0],
                            args[1],
                            args[2],
                            rationale="batch",
                            author=author,
                        )
                    else:  # delete
                        result = await writer.delete(
                            args[0], rationale="batch", author=author
                        )
                    if result.error:
                        raise RuntimeError(
                            f"sandbox batch failed at {kind} {args[0]!r}: "
                            f"{result.error}"
                        )

        @asynccontextmanager
        async def _ctx():
            handle = _Batch()
            try:
                yield handle
            finally:
                await handle.flush()

        return _ctx()


async def _replay_fork_environment(
    api: ExtensionAPI,
    session: AsyncSandboxSession,
    gateway_url: str,
    api_key: str | None,
) -> None:
    """Replay the source session's full sandbox history into this fork's sandbox.

    Source ARL session_id is read from lineage (preferred) or resolved via
    experiment_id lookup (requires exactly one match). Replays the entire
    history — lineage turn indices are AgentM conversation turns, not ARL
    execute steps, so partial replay by turn count would be incorrect.
    """
    lineage = api.lineage
    if not isinstance(lineage, dict) or lineage.get("kind") != "fork":
        return

    source_arl_session = lineage.get("arl_session_id")
    if not isinstance(source_arl_session, str) or not source_arl_session:
        experiment_id = lineage.get("arl_experiment_id") or lineage.get(
            "source_session_id"
        )
        if not experiment_id:
            return

        from arl import AsyncGatewayClient as _AsyncGatewayClient  # type: ignore[import-not-found]

        client = _AsyncGatewayClient(base_url=gateway_url, api_key=api_key)
        try:
            arl_sessions = await client.list_experiment_sessions(experiment_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent_env: experiment lookup failed for {}: {}", experiment_id, exc
            )
            return
        finally:
            await client.aclose()

        if len(arl_sessions) != 1:
            logger.warning(
                "agent_env: expected 1 ARL session for experiment {}, got {} — skipping replay",
                experiment_id,
                len(arl_sessions),
            )
            return
        source_arl_session = arl_sessions[0].id

    logger.info(
        "agent_env: replaying {} into {}",
        source_arl_session,
        session.session_id,
    )
    try:
        result = await session.replay_from(source_session_id=source_arl_session)
        if result.errors:
            logger.warning(
                "agent_env: replay completed with {} errors out of {} steps",
                result.errors,
                result.steps_replayed,
            )
        else:
            logger.info("agent_env: replay complete — {} steps", result.steps_replayed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_env: ARL replay failed: {}", exc)


def _build_resources(config: AgentEnvConfig) -> Any:
    reqs = {
        k: v
        for k, v in {"cpu": config.cpu_request, "memory": config.memory_request}.items()
        if v
    }
    lims = {
        k: v
        for k, v in {"cpu": config.cpu_limit, "memory": config.memory_limit}.items()
        if v
    }
    if not reqs and not lims:
        return None
    from arl.types import ResourceRequirements  # type: ignore[import-not-found]

    return ResourceRequirements(requests=reqs, limits=lims)


async def install_agent_env(api: ExtensionAPI, config: AgentEnvConfig) -> None:
    try:
        import arl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "operations backend 'agent_env' requires the 'arl-env' package. "
            "Install with: uv sync --extra agent-env"
        ) from exc

    work_dir = config.work_dir or "/workspace"
    delete_on_shutdown = (
        config.delete_on_shutdown if config.delete_on_shutdown is not None else True
    )

    owned = True
    session: arl.AsyncSandboxSession
    if config.attach_session:
        session = await arl.AsyncSandboxSession.attach(
            config.attach_session,
            gateway_url=config.gateway_url or "",
            api_key=config.api_key,
            timeout=config.create_timeout or 600.0,
        )
        owned = False
        logger.info("agent_env: attached to existing sandbox {}", config.attach_session)
    elif config.image:
        session = arl.AsyncManagedSession(
            image=config.image,
            experiment_id=config.experiment_id or api.session_id,
            gateway_url=config.gateway_url or "",
            workspace_dir=work_dir,
            api_key=config.api_key,
            timeout=config.create_timeout or 600.0,
            resources=_build_resources(config),
            profile=config.profile or "default",
            config_env=config.config_env,
            idle_timeout_seconds=config.idle_timeout_seconds,
            max_lifetime_seconds=config.max_lifetime_seconds,
        )
    else:
        raise RuntimeError(
            "operations backend 'agent_env': 'image' or 'attach_session' required"
        )

    if owned:
        await session.create_sandbox()

    session_id = session.session_id or ""
    if session_id:
        api.set_service(_AGENT_ENV_SESSION_SERVICE, session_id)
    if owned and hasattr(session, "experiment_id"):
        api.set_service(_AGENT_ENV_EXPERIMENT_SERVICE, session.experiment_id)

    bash_ops = _AgentEnvBashOperations(
        session, default_work_dir=work_dir, default_timeout=config.timeout,
    )
    writer = _AgentEnvResourceWriter(
        session, work_dir=work_dir,
        gateway_url=config.gateway_url or "", api_key=config.api_key,
        session_id=session_id,
    )
    api.register_operations(bash=bash_ops)
    api.register_resource_writer(writer)

    if owned:
        await _replay_fork_environment(
            api, session, config.gateway_url or "", config.api_key
        )

    def _on_shutdown(_event: SessionShutdownEvent) -> None:
        if not owned:
            return
        if delete_on_shutdown:
            from arl import GatewayClient  # type: ignore[import-not-found]

            client = GatewayClient(
                base_url=config.gateway_url or "", api_key=config.api_key
            )
            try:
                client.delete_session(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "agent_env: sandbox deletion failed on shutdown: {}", exc
                )
            finally:
                client.close()
        else:
            logger.info(
                "agent_env: keeping sandbox {} for external cleanup", session_id
            )

    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
