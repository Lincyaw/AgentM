"""ARL sandbox ``BashOperations`` implementation."""

from __future__ import annotations

import asyncio
import shlex
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentm.core.abi import ExecResult

from ..agent_env import async_execute

if TYPE_CHECKING:
    from arl import SandboxSession as ArlSandboxSession


class AgentEnvBashOperations:
    """``BashOperations`` impl that executes commands inside an ARL sandbox.

    Each call wraps ``cmd`` as ``bash -lc <cmd>`` and dispatches a single
    step through the ARL sandbox session's ``execute`` method.
    """

    def __init__(
        self,
        session: "ArlSandboxSession",
        *,
        default_work_dir: str,
        default_timeout: float | None,
        gateway_url: str,
        api_key: str | None,
    ) -> None:
        self._session = session
        self._default_work_dir = default_work_dir
        self._default_timeout = default_timeout
        self._gateway_url = gateway_url
        self._api_key = api_key

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
            async_execute(
                self._session,
                [step],
                on_output=_stream_output if on_data is not None else None,
                gateway_url=self._gateway_url,
                api_key=self._api_key,
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
            await async_execute(
                self._session,
                [step],
                gateway_url=self._gateway_url,
                api_key=self._api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent_env: failed to cancel remote exec {}: {}", exec_id, exc
            )


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        logger.debug("agent_env: cancelled execute task finished late: {}", exc)
