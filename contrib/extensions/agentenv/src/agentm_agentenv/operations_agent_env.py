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
- ``sync_cwd``      — When true (default false), seed the sandbox ``work_dir``
                      from the host cwd's git HEAD before the run and sync the
                      agent's diff back to the host cwd on shutdown. Lets an
                      upstream dispatcher (e.g. workbuddy) hand the agent a real
                      repo to edit and recover the diff for commit/push *without*
                      putting any VCS credentials inside the sandbox. The host
                      cwd must be a git work tree.
- ``host_workspace`` — Host directory to seed from / sync back to (default:
                      ``os.getcwd()``, which is the dispatcher-provided ``--cwd``).

§11 single-file contract: only stdlib + ``agentm.core.abi.*`` +
``agentm.extensions.*`` + ``arl`` (optional 3rd-party). No atom-to-atom imports.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import subprocess
import sys
import urllib.request
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
            "sync_cwd": {"type": "boolean"},
            "host_workspace": {"type": "string"},
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
        # Use base64 encoding to avoid stdout truncation at the ARL gateway's
        # buffer limit (~8KB). Raw `cat` output gets clipped for files > 8KB.
        abs_path = self._abs(path)
        stdout, stderr, code = await self._run(
            ["bash", "-c", f"base64 -w0 -- {shlex.quote(abs_path)}"],
        )
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        try:
            return base64.b64decode(stdout)
        except Exception:
            # Fallback: maybe base64 not available, try raw cat
            stdout2, stderr2, code2 = await self._run(["cat", "--", abs_path])
            if code2 != 0:
                raise FileNotFoundError(stderr2.decode("utf-8", "replace") or path)
            return stdout2

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

    Read and write use the ARL gateway HTTP file API (``/v1/sessions/{id}/files``)
    to bypass the ~8KB stdout limit on ``session.execute``. Mtime tokens and
    directory operations still use execute (small outputs).
    """

    def __init__(self, session: Any, *, work_dir: str, gateway_url: str) -> None:
        self._session = session
        self._work_dir = work_dir.rstrip("/") or "/"
        self._gateway_url = gateway_url.rstrip("/")

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
        # Versioning token: ``<mtime_ns>-<sha16>`` for files <= the size
        # cap, ``<mtime_ns>-<size>`` otherwise. Combining mtime with a
        # content digest defeats the 1-second-resolution collision that
        # ``stat -c '%Y'`` alone suffered (two writes inside the same
        # second produced identical tokens, so
        # ``current_version_for_path`` reported "unchanged" when content
        # had actually changed). ``stat -c '%.Y'`` gives fractional
        # seconds on GNU coreutils, which we encode as nanoseconds. The
        # 16 MiB cap keeps the digest cost bounded; above it we fall
        # back to ``(mtime_ns, size)`` — collisions are still possible
        # for same-second writes that preserve size, documented limit.
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

    async def _write_bytes(self, abs_path: str, content: bytes) -> tuple[bool, str]:
        # Use the HTTP file API to bypass the execute stdout/command size limit.
        session_id = getattr(self._session, "session_id", None)
        if not session_id:
            return False, "no session id"
        # Ensure parent directory exists.
        await self._run(["bash", "-lc", f"mkdir -p \"$(dirname -- {_sh_quote(abs_path)})\""])
        # Compute the relative path from work_dir for the upload API.
        rel_path = abs_path
        if rel_path.startswith(self._work_dir + "/"):
            rel_path = rel_path[len(self._work_dir) + 1:]
        elif rel_path.startswith(self._work_dir):
            rel_path = rel_path[len(self._work_dir):]
        try:
            _upload_to_pod(self._gateway_url, session_id, rel_path, content)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    # --- ResourceWriter API ----------------------------------------------

    async def read(self, path: str) -> bytes:
        abs_path = self._resolve(path)
        if not self._in_sandbox(path):
            raise FileNotFoundError(
                f"agent-env writer cannot read {path!r}: outside {self._work_dir!r}"
            )
        # The ARL gateway truncates stdout at ~8KB. Read in 6KB chunks
        # via `dd` to bypass the limit. Each chunk stays well under 8KB.
        # First get the file size.
        stdout, stderr, code = await self._run(
            ["bash", "-c", f"wc -c < {shlex.quote(abs_path)}"],
        )
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        total = int(stdout.strip())
        if total == 0:
            return b""
        chunks: list[bytes] = []
        offset = 0
        chunk_size = 6000
        while offset < total:
            stdout2, _, code2 = await self._run(
                ["bash", "-c", f"dd if={shlex.quote(abs_path)} bs=1 skip={offset} count={chunk_size} 2>/dev/null"],
            )
            if code2 != 0 or not stdout2:
                break
            chunks.append(stdout2)
            offset += len(stdout2)
        return b"".join(chunks)

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
        #
        # Uses the same ``<mtime_ns>-<sha16>`` token shape as the async
        # ``_mtime_token`` so a write's ``commit_sha_after`` and a later
        # ``current_version_for_path`` compare equal when (and only when)
        # the file is byte-identical to the just-written content.
        try:
            response = self._session.execute(
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


# --- cwd↔sandbox workspace sync (opt-in via ``sync_cwd``) -------------------
#
# Closes the dispatcher gap where an agent edits the sandbox ``work_dir`` but
# the dispatcher publishes from a host working tree the pod never touches. The
# pod stays credential-free: the host clones (with its creds), we seed the pod
# from the host's committed tree, and we apply the agent's diff back to the host
# so the dispatcher's own commit/push path ships it.
_SEED_ARCHIVE_NAME = ".wb_seed.tar.gz"
_BASELINE_TAG = "wb-baseline"
_SYNC_GIT_IDENT = ("-c", "user.email=workbuddy@local", "-c", "user.name=workbuddy")


def _exclude_host_paths(host_dir: str, patterns: tuple[str, ...]) -> None:
    """Append ``patterns`` to the host work tree's ``.git/info/exclude``.

    Local-only ignore: never committed, never touches the repo's tracked
    ``.gitignore``. Best-effort — failures just mean the dispatcher might
    commit a stray file; not worth aborting the run.
    """
    located = _run_host_git(host_dir, "rev-parse", "--git-path", "info/exclude")
    if located.returncode != 0:
        return
    exclude_path = located.stdout.decode().strip()
    if not exclude_path:
        return
    if not os.path.isabs(exclude_path):
        exclude_path = os.path.join(host_dir, exclude_path)
    try:
        existing = ""
        if os.path.exists(exclude_path):
            with open(exclude_path, encoding="utf-8") as fh:
                existing = fh.read()
        missing = [p for p in patterns if p not in existing]
        if not missing:
            return
        os.makedirs(os.path.dirname(exclude_path), exist_ok=True)
        with open(exclude_path, "a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write("# added by operations_agent_env sync_cwd\n")
            fh.write("\n".join(missing) + "\n")
    except OSError as exc:
        print(f"WARNING: [agent_env_sync] could not update git exclude: {exc}", file=sys.stderr)


def _run_host_git(host_dir: str, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603,S607
        ["git", "-C", host_dir, *args],
        input=stdin,
        capture_output=True,
        check=False,
    )


def _pod_exec(session: Any, cmd: str, work_dir: str) -> tuple[str, str, int]:
    """Run ``bash -lc cmd`` in the sandbox; return (stdout, stderr, exit_code)."""
    resp = session.execute([{"name": "wb_workspace_sync", "command": ["bash", "-lc", cmd], "work_dir": work_dir}])
    if not getattr(resp, "results", None):
        return "", "agent-env returned no results", 1
    out = resp.results[0].output
    return out.stdout, out.stderr, out.exit_code


def _upload_to_pod(gateway_url: str, session_id: str, rel_path: str, payload: bytes) -> None:
    """Upload bytes to ``rel_path`` (relative to work_dir) via the gateway."""
    body = json.dumps(
        {"path": rel_path, "content": base64.b64encode(payload).decode("ascii"), "encoding": "base64"}
    ).encode("utf-8")
    url = gateway_url.rstrip("/") + f"/v1/sessions/{session_id}/files"
    req = urllib.request.Request(  # noqa: S310
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
        resp.read()


_BASE_TAG = "main-base"


def _seed_sandbox_from_host(session: Any, gateway_url: str, host_dir: str, work_dir: str) -> None:
    """Seed the sandbox ``work_dir`` from ``host_dir``'s git HEAD and baseline it.

    Tracked files only (``git archive`` respects ``.gitignore``); the sandbox
    gets the code, never the remote or credentials. A baseline commit inside the
    pod gives :func:`_sync_sandbox_to_host` a precise diff target.

    When ``AGENTM_GIT_BASE_REF`` is set (e.g. ``origin/main``), the sandbox
    gets a two-commit history: base_ref content as ``main-base``, then HEAD
    content overlaid as ``pr-changes``.  This lets review/merge agents run
    ``git diff main-base..HEAD`` to see what the PR changed.
    """
    inside = _run_host_git(host_dir, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        raise RuntimeError(
            f"operations_agent_env: sync_cwd requires a git work tree at {host_dir!r}"
        )
    session_id = getattr(session, "session_id", None)
    if not session_id:
        raise RuntimeError("operations_agent_env: sync_cwd seed failed: sandbox has no session id")

    base_ref = os.environ.get("AGENTM_GIT_BASE_REF")
    ident = " ".join(_SYNC_GIT_IDENT)

    if base_ref:
        base_archive = _run_host_git(host_dir, "archive", "--format=tar.gz", base_ref)
        if base_archive.returncode != 0:
            print(
                f"WARNING: [agent_env_sync] git archive {base_ref} failed, "
                "falling back to single-stage seed",
                file=sys.stderr,
            )
            base_ref = None

    # Exclude .agentm/ from sandbox git so uploaded skills and observability
    # data don't pollute the sync-back diff (which uses git diff --cached).
    _SANDBOX_GITIGNORE = "echo '.agentm/' >> .gitignore; "

    if base_ref:
        # Two-stage seed: base_ref → HEAD gives the agent real diff context.
        _upload_to_pod(gateway_url, session_id, _SEED_ARCHIVE_NAME, base_archive.stdout)
        base_cmd = (
            f"set -e; cd {work_dir}; "
            f"tar -xzf {_SEED_ARCHIVE_NAME}; rm -f {_SEED_ARCHIVE_NAME}; "
            "git init -q; "
            + _SANDBOX_GITIGNORE +
            f"git {ident} add -A; "
            f"git {ident} commit -q -m 'main (base)' --allow-empty; "
            f"git tag -f {_BASE_TAG}"
        )
        out, err, code = _pod_exec(session, base_cmd, work_dir)
        if code != 0:
            raise RuntimeError(
                f"operations_agent_env: sync_cwd base seed failed (exit {code}): {err or out}"
            )
        head_archive = _run_host_git(host_dir, "archive", "--format=tar.gz", "HEAD")
        if head_archive.returncode != 0:
            raise RuntimeError(
                "operations_agent_env: sync_cwd seed failed (git archive HEAD): "
                + head_archive.stderr.decode(errors="replace").strip()
            )
        _upload_to_pod(gateway_url, session_id, _SEED_ARCHIVE_NAME, head_archive.stdout)
        overlay_cmd = (
            f"set -e; cd {work_dir}; "
            f"tar -xzf {_SEED_ARCHIVE_NAME}; rm -f {_SEED_ARCHIVE_NAME}; "
            f"git {ident} add -A; "
            f"git {ident} commit -q -m 'workbuddy PR changes' --allow-empty; "
            f"git tag -f {_BASELINE_TAG}"
        )
        out, err, code = _pod_exec(session, overlay_cmd, work_dir)
        if code != 0:
            raise RuntimeError(
                f"operations_agent_env: sync_cwd overlay seed failed (exit {code}): {err or out}"
            )
        print(
            f"INFO: [agent_env_sync] two-stage seed: {base_ref} → HEAD into {work_dir}",
            file=sys.stderr,
        )
    else:
        # Single-stage seed (original path, used by dev agents).
        archive = _run_host_git(host_dir, "archive", "--format=tar.gz", "HEAD")
        if archive.returncode != 0:
            raise RuntimeError(
                "operations_agent_env: sync_cwd seed failed (git archive HEAD): "
                + archive.stderr.decode(errors="replace").strip()
            )
        _upload_to_pod(gateway_url, session_id, _SEED_ARCHIVE_NAME, archive.stdout)
        seed_cmd = (
            f"set -e; cd {work_dir}; "
            f"tar -xzf {_SEED_ARCHIVE_NAME}; rm -f {_SEED_ARCHIVE_NAME}; "
            "git init -q; "
            + _SANDBOX_GITIGNORE +
            f"git {ident} add -A; "
            f"git {ident} commit -q -m wb-baseline --allow-empty; "
            f"git tag -f {_BASELINE_TAG}"
        )
        out, err, code = _pod_exec(session, seed_cmd, work_dir)
        if code != 0:
            raise RuntimeError(
                f"operations_agent_env: sync_cwd seed failed in sandbox (exit {code}): {err or out}"
            )
        print(
            f"INFO: [agent_env_sync] seeded sandbox {work_dir} from {host_dir} ({len(archive.stdout)} bytes)",
            file=sys.stderr,
        )
    # Keep AgentM's own host-side runtime droppings (``.agentm/`` observability,
    # catalog) and our transient seed archive out of the dispatcher's commit:
    # they are written into the host cwd but are not part of the repo. A local
    # ``.git/info/exclude`` entry is non-invasive (never committed, never
    # touches the repo's tracked ``.gitignore``).
    _exclude_host_paths(host_dir, (".agentm/", _SEED_ARCHIVE_NAME))
    print(
        f"INFO: [agent_env_sync] seeded sandbox {work_dir} from {host_dir} ({len(archive.stdout)} bytes)",
        file=sys.stderr,
    )


def _sync_sandbox_to_host(session: Any, host_dir: str, work_dir: str) -> None:
    """Apply the agent's sandbox diff (vs the seed baseline) back onto ``host_dir``.

    Best-effort and idempotent: an empty diff (e.g. a review agent that changed
    nothing) is a no-op. Runs at shutdown, before the sandbox is deleted, so the
    dispatcher's subsequent commit/push sees the changes in its work tree.
    """
    ident = " ".join(_SYNC_GIT_IDENT)
    diff_cmd = (
        f"cd {work_dir} 2>/dev/null || exit 0; "
        "git rev-parse --git-dir >/dev/null 2>&1 || exit 0; "
        f"git {ident} add -A; "
        # Diff against the immovable baseline tag, not HEAD: the agent may have
        # committed inside the sandbox, which would move HEAD and hide its work.
        f"git diff --cached --binary {_BASELINE_TAG} | base64 -w0"
    )
    out, err, code = _pod_exec(session, diff_cmd, work_dir)
    if code != 0:
        print(f"ERROR: [agent_env_sync] sandbox diff failed (exit {code}): {err}", file=sys.stderr)
        return
    encoded = out.strip()
    if not encoded:
        return
    try:
        patch = base64.b64decode(encoded)
    except ValueError as exc:  # binascii.Error is a ValueError subclass
        print(f"ERROR: [agent_env_sync] could not decode sandbox patch: {exc}", file=sys.stderr)
        return
    if not patch.strip():
        return
    applied = _run_host_git(host_dir, "apply", "--binary", "--whitespace=nowarn", stdin=patch)
    if applied.returncode != 0:
        print(
            "ERROR: [agent_env_sync] git apply failed on host work tree "
            f"{host_dir}: {applied.stderr.decode(errors='replace').strip()}",
            file=sys.stderr,
        )
        return
    print(
        f"INFO: [agent_env_sync] applied {len(patch)} bytes of sandbox changes to {host_dir}",
        file=sys.stderr,
    )


def _upload_skills_to_sandbox(session: Any, gateway_url: str, work_dir: str) -> None:
    """Upload SKILL.md files from ``AGENTM_SKILLS_DIR`` into the sandbox.

    Skills are stored on the host PVC (persistent across sessions). This
    function copies them into ``<work_dir>/.agentm/skills/`` inside the
    sandbox so ``skill_loader`` can discover them at its normal path.
    """
    skills_dir = os.environ.get("AGENTM_SKILLS_DIR")
    if not skills_dir or not os.path.isdir(skills_dir):
        return
    session_id = getattr(session, "session_id", None)
    if not session_id:
        return
    target_base = ".agentm/skills"
    _pod_exec(session, f"mkdir -p {work_dir}/{target_base}", work_dir)
    count = 0
    for entry in sorted(os.listdir(skills_dir)):
        entry_path = os.path.join(skills_dir, entry)
        if os.path.isdir(entry_path):
            skill_file = os.path.join(entry_path, "SKILL.md")
            if os.path.isfile(skill_file):
                target = f"{target_base}/{entry}/SKILL.md"
                _pod_exec(session, f"mkdir -p {work_dir}/{target_base}/{entry}", work_dir)
                try:
                    with open(skill_file, "rb") as fh:
                        _upload_to_pod(gateway_url, session_id, target, fh.read())
                    count += 1
                except Exception as exc:
                    print(f"WARNING: [agent_env_sync] skill upload failed for {entry}: {exc}", file=sys.stderr)
        elif entry.endswith(".md") and os.path.isfile(entry_path):
            try:
                with open(entry_path, "rb") as fh:
                    _upload_to_pod(gateway_url, session_id, f"{target_base}/{entry}", fh.read())
                count += 1
            except Exception as exc:
                print(f"WARNING: [agent_env_sync] skill upload failed for {entry}: {exc}", file=sys.stderr)
    if count:
        print(f"INFO: [agent_env_sync] uploaded {count} skill(s) to {target_base}/", file=sys.stderr)


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
    sync_cwd = bool(config.get("sync_cwd"))
    host_workspace = config.get("host_workspace")
    host_dir = host_workspace if isinstance(host_workspace, str) and host_workspace else os.getcwd()

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

    # Seed the sandbox from the host work tree before any tool runs. A seed
    # failure is fatal: continuing would let the agent edit an empty workspace
    # and silently produce an empty diff/PR.
    if sync_cwd:
        _seed_sandbox_from_host(session, gateway_url, host_dir, work_dir)

    # Upload skill files from host PVC into the sandbox so skill_loader can
    # discover them. Non-fatal: missing dir or upload errors are logged, not raised.
    _upload_skills_to_sandbox(session, gateway_url, work_dir)

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
        _AgentEnvResourceWriter(session, work_dir=work_dir, gateway_url=gateway_url)
    )

    def _on_shutdown(_event: SessionShutdownEvent) -> None:
        # Recover the agent's diff into the host work tree BEFORE tearing the
        # sandbox down, so the dispatcher's commit/push step sees the changes.
        if sync_cwd:
            try:
                _sync_sandbox_to_host(session, host_dir, work_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: [agent_env_sync] sync-back raised: {exc}", file=sys.stderr)
        try:
            session.delete_sandbox()
        except Exception:  # noqa: BLE001
            pass
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass

    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
