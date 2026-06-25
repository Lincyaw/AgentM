"""ARL agent-env sandbox implementations of ``FileOperations`` and ``BashOperations``.

Moved from the former ``operations_agent_env`` builtin atom. Provides
:func:`install_agent_env` for use by the unified ``operations`` atom entry point.

Lifecycle: ``install_agent_env`` creates one sandbox per AgentM session; the sandbox
is deleted on ``SessionShutdownEvent``. Each ``BashOperations.exec`` maps to one
``session.execute`` call; ``FileOperations`` are expressed as ``cat`` / ``test``
/ ``ls`` steps so semantics stay aligned with the sandbox's view of the world.

The function *also* replaces the session's :class:`ResourceWriter` (via
``api.register_resource_writer``) with a sandbox-backed implementation so
``write`` / ``edit`` (from ``file_tools``) land inside the sandbox too — keeping
read and write semantics consistent with bash. The sandbox writer refuses any path
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
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import urllib.request
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    BatchHandle,
    ExecResult,
    ExtensionAPI,
    FileStat,
    PathClass,
    SessionShutdownEvent,
    WriteResult,
    WriterAuthor,
)

class AgentEnvConfig(BaseModel):
    image: str | None = None
    experiment_id: str | None = None
    pool_ref: str | None = None
    attach_session: str | None = None
    gateway_url: str | None = None
    api_key: str | None = None
    namespace: str | None = None
    work_dir: str | None = None
    timeout: float | None = None
    idle_timeout_seconds: int | None = None

def _resolve_str(value: str | None, env_var: str, default: str | None) -> str | None:
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
        # ARL gateway truncates raw `cat` stdout for files > ~8KB.
        # base64 output (ASCII) is NOT truncated, so we use that.
        abs_path = self._abs(path)
        stdout, stderr, code = await self._run(
            ["bash", "-c", f"base64 -w0 -- {shlex.quote(abs_path)}"],
        )
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        encoded = stdout.strip()
        if not encoded:
            return b""
        return base64.b64decode(encoded)

    async def access(self, path: str) -> bool:
        _stdout, _stderr, code = await self._run(["test", "-r", self._abs(path)])
        return code == 0

    async def is_dir(self, path: str) -> bool:
        _stdout, _stderr, code = await self._run(["test", "-d", self._abs(path)])
        return code == 0

    async def is_file(self, path: str) -> bool:
        _stdout, _stderr, code = await self._run(["test", "-f", self._abs(path)])
        return code == 0

    async def list_dir(self, path: str) -> list[str]:
        stdout, stderr, code = await self._run(["ls", "-1A", "--", self._abs(path)])
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        text = stdout.decode("utf-8", "replace").strip("\n")
        return sorted(line for line in text.split("\n") if line)

    async def stat(self, path: str) -> FileStat:
        abs_path = self._abs(path)
        stdout, stderr, code = await self._run(
            ["stat", "-c", "%s %Y %F", "--", abs_path]
        )
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        parts = stdout.decode().strip().split(None, 2)
        size = int(parts[0])
        mtime_s = int(parts[1])
        ftype = parts[2] if len(parts) > 2 else ""
        return FileStat(
            size=size,
            mtime_ns=mtime_s * 1_000_000_000,
            is_file="regular" in ftype,
            is_dir="directory" in ftype,
        )

    async def write_file(self, path: str, data: bytes) -> None:
        abs_path = self._abs(path)
        encoded = base64.b64encode(data).decode("ascii")
        _, stderr, code = await self._run(
            ["bash", "-c", f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(abs_path)}"]
        )
        if code != 0:
            raise OSError(f"write_file failed: {stderr.decode('utf-8', 'replace')}")

    async def makedirs(self, path: str, exist_ok: bool = True) -> None:
        abs_path = self._abs(path)
        flag = "-p" if exist_ok else ""
        cmd = ["mkdir", "--", abs_path] if not flag else ["mkdir", flag, "--", abs_path]
        _, stderr, code = await self._run(cmd)
        if code != 0 and not exist_ok:
            raise OSError(f"makedirs failed: {stderr.decode('utf-8', 'replace')}")

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
        # ARL gateway truncates raw `cat` stdout for files > ~8KB.
        # base64 output (ASCII) is NOT truncated, so we use that.
        stdout, stderr, code = await self._run(
            ["bash", "-c", f"base64 -w0 -- {shlex.quote(abs_path)}"],
        )
        if code != 0:
            raise FileNotFoundError(stderr.decode("utf-8", "replace") or path)
        encoded = stdout.strip()
        if not encoded:
            return b""
        return base64.b64decode(encoded)

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
        raise NotImplementedError(
            "agent-env writer does not support per-file restore; use "
            "SandboxSession.restore(snapshot_id) for whole-step rollback."
        )

    def current_version_for_path(self, path: str) -> str | None:
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

def _clone_repo_into_sandbox(session: Any, work_dir: str) -> None:
    """Clone the target repo into the sandbox and checkout the issue branch."""
    repo = os.environ.get("WORKBUDDY_REPO")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    issue_num = os.environ.get("WORKBUDDY_ISSUE_NUM")
    if not repo or not token:
        logger.warning("[agent_env_sync] WORKBUDDY_REPO or GH_TOKEN not set, skipping clone")
        return
    clone_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    branch = f"workbuddy/issue-{issue_num}" if issue_num else ""
    clone_cmd = (
        f"set -e; "
        f"git clone --quiet {shlex.quote(clone_url)} {work_dir} 2>/dev/null || "
        f"  (rm -rf {work_dir} && git clone --quiet {shlex.quote(clone_url)} {work_dir}); "
        f"cd {work_dir}; "
        "git config user.email 'workbuddy@local'; "
        "git config user.name 'workbuddy'; "
    )
    if branch:
        clone_cmd += (
            f"git fetch origin {shlex.quote(branch)} 2>/dev/null && "
            f"git checkout {shlex.quote(branch)} 2>/dev/null || true"
        )
    out, err, code = _pod_exec(session, clone_cmd, "/")
    if code != 0:
        raise RuntimeError(
            f"operations agent_env: git clone failed (exit {code}): {err or out}"
        )
    _pod_exec(
        session,
        f"echo '.agentm/' >> {work_dir}/.gitignore",
        work_dir,
    )
    logger.info("[agent_env_sync] cloned {repo} into {work_dir}", repo=repo, work_dir=work_dir)

def _inject_gh_token(session: Any, work_dir: str) -> None:
    """Make GH_TOKEN available in the sandbox so agents can use `gh` CLI."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    setup_cmd = (
        f"echo 'export GH_TOKEN={shlex.quote(token)}' > /etc/profile.d/gh_token.sh; "
        f"echo 'export GITHUB_TOKEN={shlex.quote(token)}' >> /etc/profile.d/gh_token.sh; "
        "chmod 644 /etc/profile.d/gh_token.sh"
    )
    _pod_exec(session, setup_cmd, work_dir)
    logger.info("[agent_env_sync] injected GH_TOKEN into sandbox")

def _upload_skills_to_sandbox(session: Any, gateway_url: str, work_dir: str) -> None:
    """Upload SKILL.md files from ``AGENTM_SKILLS_DIR`` into the sandbox."""
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
                    logger.warning("[agent_env_sync] skill upload failed for {entry}: {exc}", entry=entry, exc=exc)
        elif entry.endswith(".md") and os.path.isfile(entry_path):
            try:
                with open(entry_path, "rb") as fh:
                    _upload_to_pod(gateway_url, session_id, f"{target_base}/{entry}", fh.read())
                count += 1
            except Exception as exc:
                logger.warning("[agent_env_sync] skill upload failed for {entry}: {exc}", entry=entry, exc=exc)
    if count:
        logger.info("[agent_env_sync] uploaded {count} skill(s) to {target_base}/", count=count, target_base=target_base)

def _replay_fork_environment(api: ExtensionAPI, arl_session: Any) -> None:
    """If this session is a fork, replay the source session's side-effect
    tool calls up to the fork turn to restore the sandbox environment.

    ``arl_session`` must be an ``arl.ManagedSession`` or ``arl.SandboxSession``
    (has ``_client`` and ``_session_id``).

    Reads ``api.lineage`` for ``kind: "fork"`` + ``source_session_id`` +
    ``fork_point.turn_index``. No-op if lineage is absent or not a fork.
    """
    lineage = api.lineage
    if not isinstance(lineage, dict) or lineage.get("kind") != "fork":
        return
    source_id = lineage.get("source_session_id")
    fork_point = lineage.get("fork_point") or {}
    turn_index = fork_point.get("turn_index") or fork_point.get("up_to")
    if not source_id:
        return

    # Find the source session's ARL sandbox by experiment_id convention:
    # agent_env uses the agentm session_id as the ARL experiment_id, so
    # the source's ARL session is the one with experiment_id == source_id.
    try:
        arl_sessions = arl_session._client.list_experiment_sessions(source_id)
        if not arl_sessions:
            logger.warning(
                "agent_env: no ARL session found for experiment {} — "
                "cannot replay (was the source run with agent_env?)", source_id,
            )
            return
        source_arl_session = arl_sessions[0].id
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_env: could not query ARL for source {}: {}", source_id, exc)
        return

    logger.info(
        "agent_env: replaying ARL session {} to turn {} into {}",
        source_arl_session, turn_index, arl_session._session_id,
    )

    try:
        up_to = int(turn_index) if turn_index is not None else None
        result = arl_session._client.replay_from(
            arl_session._session_id,
            source_session_id=source_arl_session,
            up_to_step=up_to,
        )
        logger.info(
            "agent_env: replay complete — {} steps replayed, {} errors",
            result.get("stepsReplayed", 0), result.get("errors", 0),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_env: ARL replay failed: {}", exc)


def install_agent_env(api: ExtensionAPI, config: AgentEnvConfig) -> None:
    # Deferred import keeps the SDK truly optional — atoms that never run
    # under agent-env shouldn't fail to load just because ``arl`` is absent.
    try:
        import arl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - install-time surface
        raise RuntimeError(
            "operations backend 'agent_env' requires the 'arl-env' package. "
            "Install with: uv sync --extra agent-env"
        ) from exc

    image = _resolve_str(config.image, "AGENTM_AGENT_ENV_IMAGE", None)
    pool_ref = _resolve_str(config.pool_ref, "AGENTM_AGENT_ENV_POOL_REF", None)
    attach_session = _resolve_str(
        config.attach_session, "AGENTM_AGENT_ENV_ATTACH_SESSION", None
    )
    gateway_url = _resolve_str(
        config.gateway_url, "AGENTM_AGENT_ENV_GATEWAY_URL", "http://localhost:8080"
    ) or "http://localhost:8080"
    namespace = _resolve_str(
        config.namespace, "AGENTM_AGENT_ENV_NAMESPACE", "default"
    ) or "default"
    work_dir = config.work_dir or "/workspace"
    timeout_value: float | None = config.timeout
    idle_value: int | None = config.idle_timeout_seconds
    api_key = _resolve_str(config.api_key, "AGENTM_AGENT_ENV_API_KEY", None)
    owned = True
    session: Any
    if attach_session:
        session = arl.SandboxSession.attach(
            attach_session, gateway_url=gateway_url, api_key=api_key,
        )
        owned = False
        logger.info("agent_env: attached to existing sandbox {}", attach_session)
    elif image:
        # Default experiment_id to agentm session_id so fork can look up
        # the source ARL session by experiment_id == source agentm session_id.
        experiment_id = _resolve_str(
            config.experiment_id,
            "AGENTM_AGENT_ENV_EXPERIMENT_ID",
            None,
        ) or api.session_id
        session = arl.ManagedSession(
            image=image,
            experiment_id=experiment_id,
            namespace=namespace,
            gateway_url=gateway_url,
            workspace_dir=work_dir,
            api_key=api_key,
        )
    elif pool_ref:
        session = arl.SandboxSession(
            pool_ref=pool_ref,
            namespace=namespace,
            gateway_url=gateway_url,
            keep_alive=False,
            idle_timeout_seconds=idle_value,
            api_key=api_key,
        )
    else:
        raise RuntimeError(
            "operations backend 'agent_env': one of 'attach_session', 'image', "
            "or 'pool_ref' is required. Set the atom config field or use "
            "AGENTM_AGENT_ENV_ATTACH_SESSION / AGENTM_AGENT_ENV_IMAGE / "
            "AGENTM_AGENT_ENV_POOL_REF."
        )
    if owned:
        session.create_sandbox()
        _inject_gh_token(session, work_dir)
        _clone_repo_into_sandbox(session, work_dir)
        _upload_skills_to_sandbox(session, gateway_url, work_dir)

    file_ops = _AgentEnvFileOperations(session, default_work_dir=work_dir)
    bash_ops = _AgentEnvBashOperations(
        session,
        default_work_dir=work_dir,
        default_timeout=timeout_value,
    )
    writer = _AgentEnvResourceWriter(session, work_dir=work_dir, gateway_url=gateway_url)

    api.register_operations(file=file_ops, bash=bash_ops)
    api.register_resource_writer(writer)

    if owned:
        _replay_fork_environment(api, session)

    def _on_shutdown(_event: SessionShutdownEvent) -> None:
        if not owned:
            return
        try:
            session.delete_sandbox()
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent_env: sandbox deletion failed on shutdown: {}", exc)
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent_env: session close failed on shutdown: {}", exc)

    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
