"""ARL agent-env ``ResourceWriter`` implementation.

Writes land inside the ARL sandbox. Only paths inside ``work_dir`` are
writable; everything else is treated as constitution and refused.
"""

from __future__ import annotations

import shlex
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentm.core.abi import (
    BatchHandle,
    PathClass,
    WriteResult,
    WriterAuthor,
)
from agentm.extensions.builtin._agent_env import (
    _async_execute,
    _call_maybe_async,
    _is_in_work_dir,
    _normalize_work_dir,
    _sandbox_abs,
    _workspace_relative_path,
)

if TYPE_CHECKING:
    from arl import SandboxSession as ArlSandboxSession


# Versioning-token script. Emits ``<mtime_ns>-<sha16>`` for files up to
# ``_MTIME_TOKEN_SIZE_CAP`` bytes, ``<mtime_ns>-size<size>`` for larger
# files. The mtime component uses GNU stat's ``%.Y`` format (fractional
# seconds) so we get nanosecond resolution; we strip the decimal point so
# the token is a plain ``<digits>-<rest>`` string. Invoked as
# ``bash -lc <script> bash <path>`` -- the trailing ``bash`` sets ``$0``
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


class AgentEnvResourceWriter:
    """``ResourceWriter`` impl whose writes land inside the ARL sandbox.

    Boundary contract: only paths *inside* ``work_dir`` (after resolving
    relative paths against it) are writable; everything else -- including
    every host path under the AgentM tree -- is treated as constitution
    and refused. The sandbox cannot see the host filesystem, so this is
    fail-safe by construction: the agent literally cannot mutate its own
    code from a sandbox session.

    Read and write use the ARL SDK file API to bypass the ~8KB stdout limit
    on ``session.execute``.
    """

    def __init__(
        self,
        session: "ArlSandboxSession",
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
        response = await _async_execute(
            self._session,
            [step],
            gateway_url=self._gateway_url,
            api_key=self._api_key,
        )
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
            await _call_maybe_async(self._session.upload_file, rel_path, content)
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
                return await _call_maybe_async(self._session.download_file, rel_path)
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
