"""``operations_agent_env`` atom — backs Operations with the ARL agent-env sandbox.

Replaces the local-stdlib :class:`Operations` bundle (``operations_local``) with
one whose ``BashOperations`` and ``FileOperations`` run inside an ARL sandbox.
The atom now defaults to ``arl.ManagedSession(image=...)`` — the server-side
managed pool flow, where the gateway provisions/scales pods automatically — and
falls back to ``arl.SandboxSession(pool_ref=...)`` for callers that still pin a
pre-created WarmPool. Use this whenever the agent must execute in an isolated
Kubernetes-backed environment instead of the operator's host shell.

Lifecycle: ``install`` creates one sandbox per AgentM session; the sandbox is
deleted on ``SessionShutdownEvent``. Each ``BashOperations.exec`` maps to one
``session.execute`` call; ``FileOperations`` are expressed as ``cat`` / ``test``
/ ``ls`` steps so semantics stay aligned with the sandbox's view of the world.

The atom *also* replaces the session's :class:`ResourceWriter` (via
``api.register_resource_writer``) with a sandbox-backed implementation so
``tool_write`` / ``tool_edit`` land inside the sandbox too — keeping read and
write semantics consistent with bash. The sandbox writer refuses any path
outside ``work_dir`` (including every host path), so an agent in a sandboxed
session cannot modify its own AgentM code.

Config (env-var fallbacks shown). Exactly one of ``image`` / ``pool_ref`` is
required; if both are set, ``image`` wins (managed pool path):

- ``image``         — Container image for the managed pool (env:
                      ``AGENTM_AGENT_ENV_IMAGE``). When set, the atom uses
                      ``arl.ManagedSession`` and the server provisions the pool.
- ``experiment_id`` — Logical experiment grouping for managed sessions (env:
                      ``AGENTM_AGENT_ENV_EXPERIMENT_ID``, default
                      ``agentm-default``). Lets you bulk-delete all sandboxes
                      spawned by one AgentM workload via
                      ``GatewayClient.delete_experiment``.
- ``pool_ref``      — Pre-created WarmPool to allocate from (env:
                      ``AGENTM_AGENT_ENV_POOL_REF``). Legacy / advanced path,
                      kept for backward compatibility.
- ``gateway_url``   — ARL Gateway base URL (env: ``AGENTM_AGENT_ENV_GATEWAY_URL``,
                      default ``http://localhost:8080``)
- ``namespace``     — Kubernetes namespace (env: ``AGENTM_AGENT_ENV_NAMESPACE``,
                      default ``default``)
- ``work_dir``      — Default cwd inside the sandbox (default ``/workspace``)
- ``timeout``       — Per-step timeout seconds; ``None`` means no timeout
- ``idle_timeout_seconds`` — Sandbox idle TTL on the gateway (legacy path only;
                      ManagedSession handles idle policy server-side).

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
from agentm.core.abi.resource import (
    BatchHandle,
    PathClass,
    WriteResult,
    WriterAuthor,
)
from agentm.extensions import ExtensionManifest

from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path


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
            "image": {"type": "string"},
            "experiment_id": {"type": "string"},
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


class _AgentEnvResourceWriter:
    """``ResourceWriter`` impl whose writes land inside the ARL sandbox.

    Boundary contract: only paths *inside* ``work_dir`` (after resolving
    relative paths against it) are writable; everything else — including
    every host path under the AgentM tree — is treated as constitution
    and refused. The sandbox cannot see the host filesystem, so this is
    fail-safe by construction: the agent literally cannot mutate its own
    code from a sandbox session.

    Writes go through ARL ``execute`` steps. ``write`` uploads the bytes
    via a base64-encoded ``tee`` pipeline so binary content survives;
    ``replace`` reads-modifies-writes through the same path; ``delete``
    runs ``rm -f``. Version tokens are best-effort: we use ``stat -c %Y``
    (mtime) which is enough for ``current_version_for_path`` callers that
    want an opaque "was-it-the-same-write" check. ``restore`` is not
    supported — the sandbox doesn't keep per-file history outside of
    ARL's own snapshot_id mechanism, which is per-step rather than
    per-file.
    """

    def __init__(self, session: Any, *, work_dir: str) -> None:
        self._session = session
        self._work_dir = work_dir.rstrip("/") or "/"

    # --- path classification ---------------------------------------------

    def _resolve(self, path: str) -> str:
        return path if path.startswith("/") else f"{self._work_dir}/{path}"

    def _in_sandbox(self, path: str) -> bool:
        resolved = self._resolve(path)
        prefix = self._work_dir
        return resolved == prefix or resolved.startswith(prefix + "/")

    def classify(self, path: str) -> PathClass:
        # In-sandbox paths are managed (we track an mtime-token version);
        # everything else is treated as constitution so the writer refuses.
        return "managed" if self._in_sandbox(path) else "constitution"

    # --- ARL plumbing -----------------------------------------------------

    async def _run(self, command: list[str]) -> tuple[bytes, bytes, int]:
        step = {
            "name": "agentm_resource_writer",
            "command": command,
            "work_dir": self._work_dir,
        }
        response = await asyncio.to_thread(self._session.execute, [step])
        if not response.results:
            return b"", b"no result", 1
        out = response.results[0].output
        return out.stdout.encode("utf-8"), out.stderr.encode("utf-8"), out.exit_code

    async def _mtime_token(self, path: str) -> str | None:
        stdout, _stderr, code = await self._run(["stat", "-c", "%Y", "--", path])
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

    async def _write_bytes(self, abs_path: str, content: bytes) -> tuple[bool, str]:
        import base64

        encoded = base64.b64encode(content).decode("ascii")
        # mkdir -p the parent, then atomically replace via temp + mv so
        # partial writes never leave a half-written file. Using base64
        # avoids any shell quoting hazard on arbitrary bytes.
        script = (
            f"set -e; mkdir -p \"$(dirname -- {_sh_quote(abs_path)})\"; "
            f"tmp=\"$(mktemp -- {_sh_quote(abs_path + '.XXXXXX')})\"; "
            f"printf %s {_sh_quote(encoded)} | base64 -d > \"$tmp\"; "
            f"mv -- \"$tmp\" {_sh_quote(abs_path)}"
        )
        _stdout, stderr, code = await self._run(["bash", "-lc", script])
        return code == 0, stderr.decode("utf-8", "replace")

    # --- ResourceWriter API ----------------------------------------------

    async def read(self, path: str) -> bytes:
        abs_path = self._resolve(path)
        if not self._in_sandbox(path):
            raise FileNotFoundError(
                f"agent-env writer cannot read {path!r}: outside {self._work_dir!r}"
            )
        stdout, stderr, code = await self._run(["cat", "--", abs_path])
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        return stdout

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author  # accepted for protocol parity; sandbox has no audit log
        if not self._in_sandbox(path):
            return self._refuse(path)
        abs_path = self._resolve(path)
        before = await self._mtime_token(abs_path)
        ok, err = await self._write_bytes(abs_path, content)
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
        if not self._in_sandbox(path):
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
        if not self._in_sandbox(path):
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
        # Per-file restore would need bookkeeping the sandbox doesn't keep.
        # ARL has session-wide snapshot_id (per execute step) which would
        # need a separate "session.restore" call — out of scope for the
        # ResourceWriter Protocol. Atoms that need rollback should use
        # ARL's SandboxSession.restore() directly.
        raise NotImplementedError(
            "agent-env writer does not support per-file restore; use "
            "SandboxSession.restore(snapshot_id) for whole-step rollback."
        )

    def current_version_for_path(self, path: str) -> str | None:
        # Synchronous caller surface (the ResourceWriter Protocol matches
        # GitBackedResourceWriter, which can read mtime synchronously). The
        # ARL SDK is sync, so we shell out directly via a single execute()
        # without involving asyncio.
        try:
            response = self._session.execute(
                [
                    {
                        "name": "agentm_stat",
                        "command": ["stat", "-c", "%Y", "--", self._resolve(path)],
                        "work_dir": self._work_dir,
                    }
                ]
            )
        except Exception:  # noqa: BLE001
            return None
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
        del rationale  # accepted for protocol parity

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
                # Sandbox doesn't expose multi-step atomicity, so we
                # replay sequentially. First failure aborts the rest —
                # mirrors the in-memory single-step semantics callers see.
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


def _sh_quote(value: str) -> str:
    """POSIX-safe single-quote escape for a string interpolated into a shell command."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Deferred import keeps the SDK truly optional — atoms that never run
    # under agent-env shouldn't fail to load just because ``arl`` is absent.
    try:
        import arl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - install-time surface
        raise RuntimeError(
            "operations_agent_env requires the 'arl-env' package. "
            "Install with: uv sync --extra agent-env"
        ) from exc

    image = _resolve(config, "image", "AGENTM_AGENT_ENV_IMAGE", None)
    pool_ref = _resolve(config, "pool_ref", "AGENTM_AGENT_ENV_POOL_REF", None)
    if not image and not pool_ref:
        raise RuntimeError(
            "operations_agent_env: one of 'image' (managed pool, default) or "
            "'pool_ref' (legacy pre-created WarmPool) is required. Set the "
            "atom config field, or use AGENTM_AGENT_ENV_IMAGE / "
            "AGENTM_AGENT_ENV_POOL_REF."
        )
    gateway_url = _resolve(
        config, "gateway_url", "AGENTM_AGENT_ENV_GATEWAY_URL", "http://localhost:8080"
    ) or "http://localhost:8080"
    namespace = _resolve(
        config, "namespace", "AGENTM_AGENT_ENV_NAMESPACE", "default"
    ) or "default"
    work_dir = (config.get("work_dir") or "/workspace") if isinstance(config.get("work_dir"), str) else "/workspace"
    timeout = config.get("timeout")
    timeout_value: float | None = float(timeout) if isinstance(timeout, (int, float)) else None
    idle = config.get("idle_timeout_seconds")
    idle_value: int | None = int(idle) if isinstance(idle, int) else None

    session: Any
    if image:
        # Managed pool path: the server creates and scales the pool from
        # ``image``; ``experiment_id`` groups sandboxes for bulk cleanup.
        experiment_id = _resolve(
            config,
            "experiment_id",
            "AGENTM_AGENT_ENV_EXPERIMENT_ID",
            "agentm-default",
        ) or "agentm-default"
        session = arl.ManagedSession(
            image=image,
            experiment_id=experiment_id,
            namespace=namespace,
            gateway_url=gateway_url,
            workspace_dir=work_dir,
        )
    else:
        # Legacy pre-created WarmPool path. ``pool_ref`` is guaranteed
        # non-empty by the selection guard above.
        assert pool_ref is not None
        session = arl.SandboxSession(
            pool_ref=pool_ref,
            namespace=namespace,
            gateway_url=gateway_url,
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
    # Redirect tool_write / tool_edit / tool_propose_change writes into the
    # sandbox. The writer rejects any path outside ``work_dir``, so the agent
    # cannot mutate its own code from inside a sandboxed session.
    api.register_resource_writer(
        _AgentEnvResourceWriter(session, work_dir=work_dir)
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
