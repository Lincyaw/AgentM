"""Local filesystem ``ResourceWriter`` — no git, no subprocess."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path, PurePosixPath

from agentm.core.abi.resource import (
    BatchHandle,
    PathClass,
    WriteResult,
    WriterAuthor,
)
from agentm.core._internal.catalog.manifest import (
    CoreManifestPathUnsetError,
    is_constitution_path,
    load_core_manifest,
    matches_manifest_glob,
)
from agentm.core.lib.paths import expand_path


def _resolve_path(cwd: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (cwd / candidate).resolve()


class GitBackedResourceWriter:
    """Local-filesystem resource writer.

    Historical name kept for backward compatibility — this implementation
    does plain file I/O with no git operations.  Path classification
    (constitution / managed / unmanaged) is still enforced; writes to
    constitution paths are refused.
    """

    def __init__(
        self,
        *,
        cwd: str,
        session_id: str = "",
        bus: object = None,
        auto_commit: bool = True,
        protected_branches: object = None,
    ) -> None:
        del session_id, bus, auto_commit, protected_branches
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
            lambda: sorted(e.name for e in resolved.iterdir())
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
        path_class = self.classify(path)
        if path_class == "constitution":
            return WriteResult._error(
                path, path_class, f"Refusing to modify constitution path {path!r}"
            )
        try:
            await asyncio.to_thread(self._write_bytes, resolved, content)
        except Exception as exc:  # noqa: BLE001
            return WriteResult._error(path, path_class, str(exc))
        return WriteResult._uncommitted(path, path_class)

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
        path_class = self.classify(path)
        if path_class == "constitution":
            return WriteResult._error(
                path, path_class, f"Refusing to modify constitution path {path!r}"
            )
        try:
            current = await asyncio.to_thread(resolved.read_bytes)
        except Exception as exc:  # noqa: BLE001
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
            return WriteResult._error(path, path_class, str(exc))
        return WriteResult._uncommitted(path, path_class)

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        resolved = _resolve_path(self._cwd, path)
        path_class = self.classify(path)
        if path_class == "constitution":
            return WriteResult._error(
                path, path_class, f"Refusing to modify constitution path {path!r}"
            )
        try:
            await asyncio.to_thread(resolved.unlink)
        except Exception as exc:  # noqa: BLE001
            return WriteResult._error(path, path_class, str(exc))
        return WriteResult._uncommitted(path, path_class)

    def classify(self, path: str) -> PathClass:
        try:
            if is_constitution_path(path):
                return "constitution"
            managed_globs = load_core_manifest().managed_globs
        except CoreManifestPathUnsetError:
            managed_globs = ()

        resolved = _resolve_path(self._cwd, path)
        try:
            relative = resolved.relative_to(self._cwd)
        except ValueError:
            return "unmanaged"

        rel_posix = PurePosixPath(relative).as_posix()
        if any(matches_manifest_glob(pattern, rel_posix) for pattern in managed_globs):
            return "managed"
        return "unmanaged"

    def restore(self, path: Path, version: str) -> None:  # noqa: ARG002
        raise NotImplementedError("local writer has no version tracking")

    def current_version_for_path(self, path: str) -> str | None:  # noqa: ARG002
        return None

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
                            str(args[0]), args[1],  # type: ignore[arg-type]
                            rationale="batch", author=author,
                        )
                    elif kind == "replace":
                        result = await writer.replace(
                            str(args[0]), args[1], args[2],  # type: ignore[arg-type]
                            rationale="batch", author=author,
                        )
                    else:
                        result = await writer.delete(
                            str(args[0]), rationale="batch", author=author,
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
            finally:
                await handle.flush()

        return _ctx()

    @staticmethod
    def _write_bytes(resolved: Path, content: bytes) -> None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)


DEFAULT_PROTECTED_BRANCHES: frozenset[str] = frozenset({"main", "master"})


class ProtectedBranchError(RuntimeError):
    pass


class GitOperationError(RuntimeError):
    pass
