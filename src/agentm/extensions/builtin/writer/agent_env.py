"""ARL agent-env ``ResourceWriter`` implementation.

Writes land inside the ARL sandbox.  All paths are writable — the
container itself is the isolation boundary.
"""

from __future__ import annotations

import shlex
from contextlib import AbstractAsyncContextManager, asynccontextmanager
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
    _normalize_work_dir,
    _sandbox_abs,
)

if TYPE_CHECKING:
    from arl import SandboxSession as ArlSandboxSession


def _is_enoent(message: str) -> bool:
    """Whether an error message from the sandbox means "file does not exist".

    This module owns the commands it runs (``base64``, ``ls``) and the
    gateway file API, so matching their ENOENT phrasing here is the one
    legitimate place; callers dispatch on ``FileNotFoundError`` only.
    """
    lowered = message.lower()
    return "no such file or directory" in lowered or "code = notfound" in lowered


class AgentEnvResourceWriter:
    """``ResourceWriter`` impl whose writes land inside the ARL sandbox.

    The sandbox container is the isolation boundary — all paths inside
    the container are writable.  Reads and writes use the ARL SDK file
    API to bypass the ~8 KB stdout limit on ``session.execute``.
    """

    def __init__(
        self,
        session: "ArlSandboxSession",
        *,
        work_dir: str,
    ) -> None:
        self._session = session
        self._work_dir = _normalize_work_dir(work_dir)

    # --- path helpers --------------------------------------------------------

    def _resolve(self, path: str) -> str:
        return _sandbox_abs(self._work_dir, path)

    def classify(self, path: str) -> PathClass:
        return "managed"

    # --- ARL plumbing --------------------------------------------------------

    async def _run(self, command: list[str]) -> tuple[bytes, bytes, int]:
        step = {
            "name": "agentm_resource_writer",
            "command": command,
            "work_dir": self._work_dir,
        }
        response = await _async_execute(
            self._session,
            [step],
        )
        if not response.results:
            return b"", b"no result", 1
        out = response.results[0].output
        return out.stdout.encode("utf-8"), out.stderr.encode("utf-8"), out.exit_code

    async def _write_bytes(self, path: str, content: bytes) -> tuple[bool, str]:
        try:
            await _call_maybe_async(self._session.upload_file, path, content)
        except Exception as exc:
            logger.warning("agent-env file upload failed: {}", exc)
            return False, str(exc)
        return True, ""

    # --- ResourceWriter API --------------------------------------------------

    async def read(self, path: str) -> bytes:
        abs_path = self._resolve(path)
        try:
            return await _call_maybe_async(self._session.download_file, abs_path)
        except Exception as exc:
            if _is_enoent(str(exc)):
                raise FileNotFoundError(str(exc)) from exc
            raise

    async def exists(self, path: str) -> bool:
        _stdout, _stderr, code = await self._run(
            ["test", "-r", self._resolve(path)]
        )
        return code == 0

    async def list_dir(self, path: str) -> list[str]:
        abs_path = self._resolve(path)
        stdout, stderr, code = await self._run(["ls", "-1A", "--", abs_path])
        if code != 0:
            message = stderr.decode("utf-8", "replace") or path
            if _is_enoent(message):
                raise FileNotFoundError(message)
            raise RuntimeError(f"list {path!r} failed: {message}")
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
        abs_path = self._resolve(path)
        ok, err = await self._write_bytes(abs_path, content)
        if not ok:
            return WriteResult(
                path=path,
                path_class="managed",
                error=err or "sandbox write failed",
            )
        return WriteResult(
            path=path,
            path_class="managed",
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
        try:
            current = await self.read(path)
        except FileNotFoundError as exc:
            return WriteResult(
                path=path,
                path_class="managed",
                error=str(exc),
            )
        if current != old:
            return WriteResult(
                path=path,
                path_class="managed",
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
        abs_path = self._resolve(path)
        _stdout, stderr, code = await self._run(["rm", "-f", "--", abs_path])
        if code != 0:
            return WriteResult(
                path=path,
                path_class="managed",
                error=stderr.decode("utf-8", "replace") or "sandbox delete failed",
            )
        return WriteResult(
            path=path,
            path_class="managed",
        )

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
            except BaseException:
                raise
            else:
                await handle.flush()

        return _ctx()
