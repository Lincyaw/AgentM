"""Operations atom bridging Harbor's BaseEnvironment to AgentM's tool layer.

The agent runs locally; bash and file operations execute in Harbor's sandbox
via ``environment.exec()`` / ``upload_file()`` / ``download_file()``.
The sandbox backend is pluggable (Docker, Modal, E2B, GKE, Daytona, etc.)
-- this atom only talks to the ``BaseEnvironment`` interface.

Thread-local holder lets concurrent trials coexist safely.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import shlex
import tempfile
import threading
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentm.core.abi import (
    AtomAPI,
    BatchHandle,
    ExecResult,
    ExtensionManifest,
    PathClass,
    WriterAuthor,
    WriteResult,
)
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


# ---------------------------------------------------------------------------
# Thread-local environment holder
# ---------------------------------------------------------------------------

_holder: threading.local = threading.local()


def set_harbor_environment(env: "BaseEnvironment") -> None:
    """Inject a Harbor BaseEnvironment for the current thread.

    Must be called before ``AgentSession.create()`` so the atom's
    ``install()`` can pick it up.
    """
    _holder.environment = env


def _get_environment() -> "BaseEnvironment":
    env = getattr(_holder, "environment", None)
    if env is None:
        raise RuntimeError(
            "Harbor environment not set — call set_harbor_environment() "
            "before AgentSession.create()"
        )
    return env


# ---------------------------------------------------------------------------
# Config + manifest
# ---------------------------------------------------------------------------


class HarborOpsConfig(BaseModel):
    work_dir: str = "/"
    timeout: float | None = None


MANIFEST = ExtensionManifest(
    name="harbor_ops",
    description="Routes bash/file operations through Harbor BaseEnvironment.",
    registers=(),
    config_schema=HarborOpsConfig,
    requires=(),
    api_version=1,
    tier=1,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _abs(work_dir: str, path: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(work_dir, path))


# ---------------------------------------------------------------------------
# BashOperations
# ---------------------------------------------------------------------------


class HarborBashOperations:
    """Wraps ``BaseEnvironment.exec()`` into AgentM's ``BashOperations``."""

    __slots__ = ("_env", "_default_work_dir", "_default_timeout")

    def __init__(
        self,
        env: "BaseEnvironment",
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
        on_data: Callable[[bytes], None] | None = None,
        signal: asyncio.Event | None = None,
        log_path: str | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else self._default_timeout
        timeout_sec = int(effective_timeout) if effective_timeout else None

        result = await self._env.exec(
            command=cmd,
            cwd=cwd or self._default_work_dir or None,
            env=env,
            timeout_sec=timeout_sec,
        )

        return ExecResult(
            exit_code=result.return_code,
            stdout=(result.stdout or "").encode("utf-8"),
            stderr=(result.stderr or "").encode("utf-8"),
            timed_out=False,
        )


# ---------------------------------------------------------------------------
# ResourceWriter
# ---------------------------------------------------------------------------


class HarborResourceWriter:
    """File I/O via ``BaseEnvironment.upload_file()`` / ``download_file()``."""

    __slots__ = ("_env", "_work_dir")

    def __init__(
        self,
        env: "BaseEnvironment",
        *,
        work_dir: str,
    ) -> None:
        self._env = env
        self._work_dir = work_dir

    def _resolve(self, path: str) -> str:
        return _abs(self._work_dir, path)

    def classify(self, path: str) -> PathClass:
        return "managed"

    async def read(self, path: str) -> bytes:
        abs_path = self._resolve(path)
        fd, tmp = tempfile.mkstemp()
        os.close(fd)
        try:
            await self._env.download_file(abs_path, tmp)
            return Path(tmp).read_bytes()
        except Exception as exc:
            msg = str(exc).lower()
            if "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(str(exc)) from exc
            raise
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    async def exists(self, path: str) -> bool:
        abs_path = self._resolve(path)
        result = await self._env.exec(
            command=f"test -e {shlex.quote(abs_path)}",
        )
        return result.return_code == 0

    async def list_dir(self, path: str) -> list[str]:
        abs_path = self._resolve(path)
        result = await self._env.exec(
            command=f"ls -1A -- {shlex.quote(abs_path)}",
        )
        if result.return_code != 0:
            err = result.stderr or ""
            if "no such file" in err.lower():
                raise FileNotFoundError(err)
            raise RuntimeError(f"list {path!r} failed: {err}")
        text = (result.stdout or "").strip("\n")
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
        try:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(content)
                tmp = f.name
            try:
                await self._env.upload_file(tmp, abs_path)
            finally:
                os.unlink(tmp)
        except Exception as exc:
            logger.warning("harbor write failed for {}: {}", abs_path, exc)
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
        result = await self._env.exec(
            command=f"rm -f -- {shlex.quote(abs_path)}",
        )
        if result.return_code != 0:
            return WriteResult(
                path=path,
                path_class="managed",
                error=(result.stderr or "delete failed"),
            )
        return WriteResult(path=path, path_class="managed")

    def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> AbstractAsyncContextManager[BatchHandle]:
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
                        r = await writer.write(
                            args[0], args[1], rationale="batch", author=author,
                        )
                    elif kind == "replace":
                        r = await writer.replace(
                            args[0], args[1], args[2], rationale="batch", author=author,
                        )
                    else:
                        r = await writer.delete(
                            args[0], rationale="batch", author=author,
                        )
                    if r.error:
                        raise RuntimeError(f"batch {kind} {args[0]!r}: {r.error}")

        @asynccontextmanager
        async def _ctx():
            handle = _Batch()
            yield handle
            await handle.flush()

        return _ctx()


# ---------------------------------------------------------------------------
# Atom entry point
# ---------------------------------------------------------------------------


async def install(api: AtomAPI, config: HarborOpsConfig) -> None:
    env = _get_environment()
    work_dir = config.work_dir.rstrip("/") or "/"

    bash_ops = HarborBashOperations(
        env, default_work_dir=work_dir, default_timeout=config.timeout,
    )
    writer = HarborResourceWriter(env, work_dir=work_dir)
    api.register_operations(bash=bash_ops)
    api.register_resource_writer(writer)
    logger.info("harbor_ops: operations registered (work_dir={})", work_dir)


__all__ = (
    "MANIFEST",
    "HarborOpsConfig",
    "install",
    "set_harbor_environment",
)
