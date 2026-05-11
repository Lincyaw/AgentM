"""``operations_agent_env`` atom — backs Operations with the ARL agent-env sandbox.

Replaces the local-stdlib :class:`Operations` bundle (``operations_local``) with
one whose ``BashOperations`` and ``FileOperations`` run inside an ARL sandbox via
``arl.SandboxSession``. Use this when the agent must execute in an isolated
Kubernetes-backed environment instead of the operator's host shell.

Lifecycle: ``install`` creates one sandbox per AgentM session; the sandbox is
deleted on ``SessionShutdownEvent``. Each ``BashOperations.exec`` maps to one
``session.execute`` call; ``FileOperations`` are expressed as ``cat`` / ``test``
/ ``ls`` steps so semantics stay aligned with the sandbox's view of the world.

Config (all optional except ``pool_ref``, with env-var fallbacks):
- ``pool_ref``      — WarmPool to allocate from (env: ``AGENTM_AGENT_ENV_POOL_REF``)
- ``gateway_url``   — ARL Gateway base URL (env: ``AGENTM_AGENT_ENV_GATEWAY_URL``,
                      default ``http://localhost:8080``)
- ``namespace``     — Kubernetes namespace (env: ``AGENTM_AGENT_ENV_NAMESPACE``,
                      default ``default``)
- ``work_dir``      — Default cwd inside the sandbox (default ``/workspace``)
- ``timeout``       — Per-step timeout seconds; ``None`` means no timeout
- ``idle_timeout_seconds`` — Sandbox idle TTL on the gateway

§11 single-file contract: only stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*`` + ``arl`` (optional 3rd-party). No atom-to-atom imports.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.operations import ExecResult
from agentm.extensions import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="operations_agent_env",
    description=(
        "Registers an Operations bundle backed by ARL agent-env sandboxes. "
        "Drop-in replacement for operations_local when the scenario should "
        "run tool calls inside a Kubernetes-isolated environment."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {
            "pool_ref": {"type": "string"},
            "gateway_url": {"type": "string"},
            "namespace": {"type": "string"},
            "work_dir": {"type": "string"},
            "timeout": {"type": ["number", "null"]},
            "idle_timeout_seconds": {"type": ["integer", "null"]},
        },
        "additionalProperties": False,
    },
    requires=(),
    conflicts=("operations_local",),
)


def _resolve(config: dict[str, Any], key: str, env_var: str, default: str | None) -> str | None:
    value = config.get(key)
    if isinstance(value, str) and value:
        return value
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    return default


class _AgentEnvBashOperations:
    """``BashOperations`` impl that executes commands inside an ARL sandbox.

    Each call wraps ``cmd`` as ``bash -lc <cmd>`` and dispatches a single
    step through ``SandboxSession.execute``. The ARL SDK is synchronous;
    we run the call in a worker thread to keep AgentM's event loop free.
    Streaming via ``on_data`` is best-effort: the SDK returns the full
    stdout/stderr after the step completes, so we deliver the entire
    stdout blob as a single chunk after the call returns.
    """

    def __init__(
        self,
        session: Any,
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
        step: dict[str, Any] = {
            "name": "agentm_bash",
            "command": ["bash", "-lc", cmd],
            "work_dir": work_dir,
        }
        if env:
            step["env"] = dict(env)
        if effective_timeout is not None:
            step["timeout"] = max(1, int(effective_timeout))

        timed_out = False
        try:
            response = await asyncio.to_thread(self._session.execute, [step])
        except Exception as exc:  # noqa: BLE001
            stderr = f"agent-env execute failed: {exc}".encode()
            return ExecResult(stdout=b"", stderr=stderr, exit_code=124, timed_out=False)

        if not response.results:
            return ExecResult(stdout=b"", stderr=b"agent-env returned no results", exit_code=1, timed_out=False)

        result = response.results[0]
        stdout_bytes = result.output.stdout.encode("utf-8")
        stderr_bytes = result.output.stderr.encode("utf-8")
        if on_data is not None and stdout_bytes:
            on_data(stdout_bytes)
        if signal is not None and signal.is_set():
            timed_out = True
        return ExecResult(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            exit_code=result.output.exit_code,
            timed_out=timed_out,
        )


class _AgentEnvFileOperations:
    """``FileOperations`` impl that reads through the sandbox's shell.

    Files live inside the sandbox, so the local FS is the wrong source of
    truth — every read is expressed as a ``cat`` / ``test`` / ``ls -1A`` step
    and the stdout is decoded back into the Protocol's expected shape.
    """

    def __init__(self, session: Any, *, default_work_dir: str) -> None:
        self._session = session
        self._default_work_dir = default_work_dir

    def _abs(self, path: str) -> str:
        return path if path.startswith("/") else f"{self._default_work_dir}/{path}"

    async def _run(self, command: list[str]) -> tuple[bytes, bytes, int]:
        step = {
            "name": "agentm_fs",
            "command": command,
            "work_dir": self._default_work_dir,
        }
        response = await asyncio.to_thread(self._session.execute, [step])
        if not response.results:
            return b"", b"no result", 1
        out = response.results[0].output
        return out.stdout.encode("utf-8"), out.stderr.encode("utf-8"), out.exit_code

    async def read_file(self, path: str) -> bytes:
        stdout, stderr, code = await self._run(["cat", "--", self._abs(path)])
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        return stdout

    async def access(self, path: str) -> bool:
        _stdout, _stderr, code = await self._run(["test", "-r", self._abs(path)])
        return code == 0

    async def is_dir(self, path: str) -> bool:
        _stdout, _stderr, code = await self._run(["test", "-d", self._abs(path)])
        return code == 0

    async def list_dir(self, path: str) -> list[str]:
        stdout, stderr, code = await self._run(["ls", "-1A", "--", self._abs(path)])
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        text = stdout.decode("utf-8", "replace").strip("\n")
        return sorted(line for line in text.split("\n") if line)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Deferred import keeps the SDK truly optional — atoms that never run
    # under agent-env shouldn't fail to load just because ``arl`` is absent.
    try:
        from arl import SandboxSession  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - install-time surface
        raise RuntimeError(
            "operations_agent_env requires the 'arl-env' package. "
            "Install with: uv sync --extra agent-env"
        ) from exc

    pool_ref = _resolve(config, "pool_ref", "AGENTM_AGENT_ENV_POOL_REF", None)
    if not pool_ref:
        raise RuntimeError(
            "operations_agent_env: 'pool_ref' is required (set in atom "
            "config or via AGENTM_AGENT_ENV_POOL_REF env var)"
        )
    gateway_url = _resolve(
        config, "gateway_url", "AGENTM_AGENT_ENV_GATEWAY_URL", "http://localhost:8080"
    )
    namespace = _resolve(
        config, "namespace", "AGENTM_AGENT_ENV_NAMESPACE", "default"
    ) or "default"
    work_dir = (config.get("work_dir") or "/workspace") if isinstance(config.get("work_dir"), str) else "/workspace"
    timeout = config.get("timeout")
    timeout_value: float | None = float(timeout) if isinstance(timeout, (int, float)) else None
    idle = config.get("idle_timeout_seconds")
    idle_value: int | None = int(idle) if isinstance(idle, int) else None

    session = SandboxSession(
        pool_ref=pool_ref,
        namespace=namespace,
        gateway_url=gateway_url or "http://localhost:8080",
        keep_alive=False,
        idle_timeout_seconds=idle_value,
    )
    session.create_sandbox()

    api.register_operations(
        file=_AgentEnvFileOperations(session, default_work_dir=work_dir),
        bash=_AgentEnvBashOperations(
            session,
            default_work_dir=work_dir,
            default_timeout=timeout_value,
        ),
    )

    def _on_shutdown(_event: SessionShutdownEvent) -> None:
        try:
            session.delete_sandbox()
        except Exception:  # noqa: BLE001
            pass
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass

    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
