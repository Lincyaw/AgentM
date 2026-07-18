"""Default local-filesystem :class:`ResourceWriter` implementation.

The writer is a bootstrap runtime service: atom loading and catalog wiring need
it before any atom can install.  Alternative environments may replace it
through the session services after bootstrap.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path, PurePosixPath

from loguru import logger

from agentm.core._internal.catalog.manifest import (
    CoreManifestPathUnsetError,
    is_constitution_path,
    load_core_manifest,
    matches_manifest_glob,
)
from agentm.core.abi.resource import (
    BatchHandle,
    PathClass,
    ResourceWriter,
    WriteResult,
    WriterAuthor,
)
from agentm.core.lib.paths import expand_path


def _resolve_path(cwd: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (cwd / candidate).resolve()


class LocalResourceWriter:
    """Local-filesystem resource writer.

    The implementation performs plain file I/O.
    Every mutation is classified against both the caller-visible path and the
    fully resolved physical target so symlinks cannot cross the constitution
    boundary in either direction.
    """

    def __init__(self, *, cwd: str) -> None:
        self._cwd = expand_path(cwd).resolve()

    async def read(self, path: str) -> bytes:
        resolved = _resolve_path(self._cwd, path)
        return await asyncio.to_thread(resolved.read_bytes)

    async def exists(self, path: str) -> bool:
        resolved = _resolve_path(self._cwd, path)
        return await asyncio.to_thread(
            lambda: resolved.exists() and os.access(resolved, os.R_OK)
        )

    async def list_dir(self, path: str) -> list[str]:
        resolved = _resolve_path(self._cwd, path)
        return await asyncio.to_thread(
            lambda: sorted(entry.name for entry in resolved.iterdir())
        )

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        resolved = _resolve_path(self._cwd, path)
        path_class = self._classify_resolved(path, resolved)
        if path_class == "constitution":
            return self._constitution_error(path)
        try:
            await asyncio.to_thread(self._write_bytes, resolved, content)
        except Exception as exc:  # noqa: BLE001
            logger.debug("local writer: write failed for {}: {}", path, exc)
            return WriteResult._error(path, path_class, str(exc))
        return WriteResult._success(path, path_class)

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        resolved = _resolve_path(self._cwd, path)
        path_class = self._classify_resolved(path, resolved)
        if path_class == "constitution":
            return self._constitution_error(path)
        try:
            current = await asyncio.to_thread(resolved.read_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.debug("local writer: read for compare failed for {}: {}", path, exc)
            return WriteResult._error(path, path_class, str(exc))
        if current != old:
            return WriteResult._error(
                path,
                path_class,
                f"Current bytes for {path!r} no longer match expected content",
            )
        try:
            await asyncio.to_thread(self._write_bytes, resolved, new)
        except Exception as exc:  # noqa: BLE001
            logger.debug("local writer: write failed for {}: {}", path, exc)
            return WriteResult._error(path, path_class, str(exc))
        return WriteResult._success(path, path_class)

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        resolved = _resolve_path(self._cwd, path)
        path_class = self._classify_resolved(path, resolved)
        if path_class == "constitution":
            return self._constitution_error(path)
        try:
            await asyncio.to_thread(resolved.unlink)
        except Exception as exc:  # noqa: BLE001
            logger.debug("local writer: delete failed for {}: {}", path, exc)
            return WriteResult._error(path, path_class, str(exc))
        return WriteResult._success(path, path_class)

    def classify(self, path: str) -> PathClass:
        resolved = _resolve_path(self._cwd, path)
        return self._classify_resolved(path, resolved)

    def _classify_resolved(self, path: str, resolved: Path) -> PathClass:
        """Classify the exact target a mutation will touch.

        Checking both path representations is intentional:

        - a harmless-looking symlink may resolve into constitution;
        - a constitution path may itself be a symlink to an external target.

        Either representation crossing the protected namespace is sufficient
        to reject the mutation.
        """

        try:
            if is_constitution_path(path) or is_constitution_path(str(resolved)):
                return "constitution"
            managed_globs = load_core_manifest().managed_globs
        except CoreManifestPathUnsetError:
            managed_globs = ()

        try:
            relative = resolved.relative_to(self._cwd)
        except ValueError:
            return "unmanaged"

        rel_posix = PurePosixPath(relative).as_posix()
        if any(
            matches_manifest_glob(pattern, path)
            or matches_manifest_glob(pattern, rel_posix)
            for pattern in managed_globs
        ):
            return "managed"
        return "unmanaged"

    @staticmethod
    def _constitution_error(path: str) -> WriteResult:
        return WriteResult._error(
            path,
            "constitution",
            f"Refusing to modify constitution path {path!r}",
        )

    def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> AbstractAsyncContextManager[BatchHandle]:
        del rationale

        writer = self

        class _Batch:
            def __init__(self) -> None:
                self._ops: list[tuple[str, tuple[object, ...]]] = []

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
                            str(args[0]),
                            args[1],  # type: ignore[arg-type]
                            rationale="batch",
                            author=author,
                        )
                    elif kind == "replace":
                        result = await writer.replace(
                            str(args[0]),
                            args[1],  # type: ignore[arg-type]
                            args[2],  # type: ignore[arg-type]
                            rationale="batch",
                            author=author,
                        )
                    else:
                        result = await writer.delete(
                            str(args[0]),
                            rationale="batch",
                            author=author,
                        )
                    if result.error:
                        raise RuntimeError(
                            f"batch failed at {kind} {args[0]!r}: {result.error}"
                        )

        @asynccontextmanager
        async def _ctx() -> AsyncIterator[_Batch]:
            handle = _Batch()
            try:
                yield handle
            except BaseException:
                raise
            else:
                await handle.flush()

        return _ctx()

    @staticmethod
    def _write_bytes(resolved: Path, content: bytes) -> None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        try:
            mode = resolved.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o644
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{resolved.name}.",
            dir=resolved.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(mode)
            os.replace(temporary, resolved)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = [
    "LocalResourceWriter",
    "ResourceWriter",
    "WriteResult",
]
