"""Git-backed chokepoint for writes to managed harness resources."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from agentm.core.abi import EventBus
from agentm.core._internal.catalog.manifest import (
    is_constitution_path,
    load_core_manifest,
    matches_manifest_glob,
)
from agentm.harness.events import ResourceWriteEvent

WriterAuthor = Literal["agent", "human", "indexer"]
PathClass = Literal["managed", "unmanaged", "constitution"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WriteResult:
    path: str
    path_class: PathClass
    committed: bool
    commit_sha_before: str | None
    commit_sha_after: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    stdout: str
    stderr: str


class GitOperationError(RuntimeError):
    """Raised when a git subprocess exits non-zero."""

    def __init__(
        self,
        *,
        args: tuple[str, ...],
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> None:
        message = (
            f"git {' '.join(args)!s} failed with exit code {exit_code}: "
            f"{stderr.strip() or stdout.strip() or '<no output>'}"
        )
        super().__init__(message)
        self.args_tuple = args
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class BatchHandle(Protocol):
    async def write(self, path: str, content: bytes) -> None: ...

    async def replace(self, path: str, old: bytes, new: bytes) -> None: ...

    async def delete(self, path: str) -> None: ...


class ResourceWriter(Protocol):
    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult: ...

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult: ...

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult: ...

    def classify(self, path: str) -> PathClass: ...

    def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> AbstractAsyncContextManager[BatchHandle]: ...


@dataclass(slots=True)
class _PendingBatchOp:
    kind: Literal["write", "replace", "delete"]
    path: str
    content: bytes | None = None
    old: bytes | None = None
    new: bytes | None = None


@dataclass(frozen=True, slots=True)
class _RestoreState:
    resolved: Path
    existed: bool
    content: bytes | None


class _BatchImpl(BatchHandle):
    def __init__(self) -> None:
        self.pending: list[_PendingBatchOp] = []

    async def write(self, path: str, content: bytes) -> None:
        self.pending.append(_PendingBatchOp(kind="write", path=path, content=content))

    async def replace(self, path: str, old: bytes, new: bytes) -> None:
        self.pending.append(
            _PendingBatchOp(kind="replace", path=path, old=old, new=new)
        )

    async def delete(self, path: str) -> None:
        self.pending.append(_PendingBatchOp(kind="delete", path=path))


class GitBackedResourceWriter:
    """Single chokepoint for managed-resource writes."""

    def __init__(
        self,
        *,
        cwd: str,
        session_id: str,
        bus: EventBus,
    ) -> None:
        self._cwd = Path(cwd).resolve()
        self._session_id = session_id
        self._bus = bus
        self._git_prefix: tuple[str, ...] = ()
        self._shadow_git_dir = self._cwd / ".agentm" / "repo"
        self._advisory_mode = False
        self._warned_advisory = False
        self._initial_snapshot_needed = False

        git_bin = shutil.which("git")
        if git_bin is None:
            self._enter_advisory_mode("git binary missing")
            return

        if (self._cwd / ".git").exists():
            self._initial_snapshot_needed = not self._has_head_sync()
            return

        self._shadow_git_dir.parent.mkdir(parents=True, exist_ok=True)
        if self._shadow_git_dir.exists():
            self._git_prefix = (
                f"--git-dir={self._shadow_git_dir}",
                f"--work-tree={self._cwd}",
            )
            self._initial_snapshot_needed = not self._has_head_sync()
            return

        try:
            completed = subprocess.run(
                ["git", "init", "--bare", "-q", str(self._shadow_git_dir)],
                cwd=self._cwd,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise GitOperationError(
                    args=("init", "--bare", "-q", str(self._shadow_git_dir)),
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
        except GitOperationError as exc:
            self._enter_advisory_mode("shadow repo init failed", error=exc)
            return
        self._git_prefix = (
            f"--git-dir={self._shadow_git_dir}",
            f"--work-tree={self._cwd}",
        )
        self._initial_snapshot_needed = True

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        path_class = self.classify(path)
        resolved = self._resolve_path(path)
        if path_class == "constitution":
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
                error=f"Refusing to modify constitution path {path!r}",
            )
        if path_class == "unmanaged" or self._advisory_mode:
            try:
                await asyncio.to_thread(self._write_bytes, resolved, content)
            except Exception as exc:  # noqa: BLE001
                return WriteResult(
                    path=path,
                    path_class=path_class,
                    committed=False,
                    commit_sha_before=None,
                    commit_sha_after=None,
                    error=str(exc),
                )
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
            )

        return await self._commit_single_path(
            path=path,
            rationale=rationale,
            author=author,
            write_op=lambda resolved: self._write_bytes(resolved, content),
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
        path_class = self.classify(path)
        resolved = self._resolve_path(path)
        if path_class == "constitution":
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
                error=f"Refusing to modify constitution path {path!r}",
            )
        try:
            current = await asyncio.to_thread(resolved.read_bytes)
        except Exception as exc:  # noqa: BLE001
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
                error=str(exc),
            )
        if current != old:
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
                error=f"Current bytes for {path!r} no longer match expected content",
            )
        if path_class == "unmanaged" or self._advisory_mode:
            try:
                await asyncio.to_thread(self._write_bytes, resolved, new)
            except Exception as exc:  # noqa: BLE001
                return WriteResult(
                    path=path,
                    path_class=path_class,
                    committed=False,
                    commit_sha_before=None,
                    commit_sha_after=None,
                    error=str(exc),
                )
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
            )

        return await self._commit_single_path(
            path=path,
            rationale=rationale,
            author=author,
            write_op=lambda _resolved: self._write_bytes(resolved, new),
        )

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        path_class = self.classify(path)
        resolved = self._resolve_path(path)
        if path_class == "constitution":
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
                error=f"Refusing to modify constitution path {path!r}",
            )
        if path_class == "unmanaged" or self._advisory_mode:
            try:
                await asyncio.to_thread(self._delete_path, resolved)
            except Exception as exc:  # noqa: BLE001
                return WriteResult(
                    path=path,
                    path_class=path_class,
                    committed=False,
                    commit_sha_before=None,
                    commit_sha_after=None,
                    error=str(exc),
                )
            return WriteResult(
                path=path,
                path_class=path_class,
                committed=False,
                commit_sha_before=None,
                commit_sha_after=None,
            )

        return await self._commit_single_path(
            path=path,
            rationale=rationale,
            author=author,
            write_op=lambda resolved: self._delete_path(resolved),
        )

    def classify(self, path: str) -> PathClass:
        if is_constitution_path(path):
            return "constitution"

        resolved = self._resolve_path(path)
        try:
            relative = resolved.relative_to(self._cwd)
        except ValueError:
            return "unmanaged"

        rel_posix = PurePosixPath(relative).as_posix()
        managed_globs = load_core_manifest().managed_globs
        if any(matches_manifest_glob(pattern, rel_posix) for pattern in managed_globs):
            return "managed"
        return "unmanaged"

    def batch(
        self,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> AbstractAsyncContextManager[BatchHandle]:
        @asynccontextmanager
        async def _manager() -> AsyncIterator[BatchHandle]:
            handle = _BatchImpl()
            yield handle
            await self._apply_batch(handle.pending, rationale=rationale, author=author)

        return _manager()

    async def _apply_batch(
        self,
        pending: list[_PendingBatchOp],
        *,
        rationale: str,
        author: WriterAuthor,
    ) -> None:
        if not pending:
            return

        managed_ops: list[tuple[_PendingBatchOp, Path, str]] = []
        unmanaged_ops: list[tuple[_PendingBatchOp, Path]] = []

        for op in pending:
            path_class = self.classify(op.path)
            if path_class == "constitution":
                raise RuntimeError(f"Refusing to modify constitution path {op.path!r}")

            resolved = self._resolve_path(op.path)
            if path_class == "managed" and not self._advisory_mode:
                relative = resolved.relative_to(self._cwd)
                relative_posix = PurePosixPath(relative).as_posix()
                managed_ops.append((op, resolved, relative_posix))
            else:
                unmanaged_ops.append((op, resolved))

        unmanaged_restore: list[_RestoreState] = []
        try:
            for op, resolved in unmanaged_ops:
                unmanaged_restore.append(await asyncio.to_thread(self._capture_restore_state, resolved))
                await asyncio.to_thread(self._apply_pending_op, op, resolved)
        except Exception:  # noqa: BLE001
            await asyncio.to_thread(self._restore_paths, unmanaged_restore)
            raise

        if not managed_ops:
            return

        await asyncio.to_thread(self._ensure_git_ready)
        pre_sha = await asyncio.to_thread(self._head_sha)
        managed_restore = await asyncio.to_thread(
            self._capture_restore_states,
            [resolved for _, resolved, _ in managed_ops],
        )

        try:
            await asyncio.to_thread(
                self._snapshot_human_for_dirty_paths,
                [relative_posix for _, _, relative_posix in managed_ops],
            )
            for op, resolved, _ in managed_ops:
                await asyncio.to_thread(self._apply_pending_op, op, resolved)

            relative_paths = [relative_posix for _, _, relative_posix in managed_ops]
            await asyncio.to_thread(self._stage_paths, relative_paths)
            if await asyncio.to_thread(self._is_index_clean_for_paths, relative_paths):
                return

            await asyncio.to_thread(self._commit, rationale, author)
            post_sha = await asyncio.to_thread(self._head_sha)
        except Exception:
            await asyncio.to_thread(
                self._restore_after_failure_batch,
                managed_restore,
                pre_sha,
                [relative_posix for _, _, relative_posix in managed_ops],
            )
            await asyncio.to_thread(self._restore_paths, unmanaged_restore)
            raise

        for _, _, relative_posix in managed_ops:
            await self._bus.emit(
                "resource_write",
                ResourceWriteEvent(
                    path=relative_posix,
                    pre_sha=pre_sha,
                    post_sha=post_sha,
                    rationale=rationale,
                    author=author,
                ),
            )

    async def _commit_single_path(
        self,
        *,
        path: str,
        rationale: str,
        author: WriterAuthor,
        write_op: Callable[[Path], None],
    ) -> WriteResult:
        resolved = self._resolve_path(path)
        relative = resolved.relative_to(self._cwd)
        relative_posix = PurePosixPath(relative).as_posix()
        await asyncio.to_thread(self._ensure_git_ready)
        pre_sha = await asyncio.to_thread(self._head_sha)
        restore_exists = resolved.exists()
        restore_bytes = await asyncio.to_thread(resolved.read_bytes) if restore_exists else None
        try:
            await asyncio.to_thread(self._snapshot_human_if_dirty, relative_posix)
            await asyncio.to_thread(write_op, resolved)
            await asyncio.to_thread(self._stage_paths, [relative_posix])
            if await asyncio.to_thread(self._is_index_clean_for_paths, [relative_posix]):
                current_sha = await asyncio.to_thread(self._head_sha)
                return WriteResult(
                    path=path,
                    path_class="managed",
                    committed=False,
                    commit_sha_before=pre_sha,
                    commit_sha_after=current_sha,
                )
            await asyncio.to_thread(self._commit, rationale, author)
            post_sha = await asyncio.to_thread(self._head_sha)
        except Exception as exc:  # noqa: BLE001
            await asyncio.to_thread(
                self._restore_after_failure,
                resolved,
                restore_exists,
                restore_bytes,
                pre_sha,
                relative_posix,
            )
            return WriteResult(
                path=path,
                path_class="managed",
                committed=False,
                commit_sha_before=pre_sha,
                commit_sha_after=None,
                error=str(exc),
            )

        await self._bus.emit(
            "resource_write",
            ResourceWriteEvent(
                path=relative_posix,
                pre_sha=pre_sha,
                post_sha=post_sha,
                rationale=rationale,
                author=author,
            ),
        )
        return WriteResult(
            path=path,
            path_class="managed",
            committed=True,
            commit_sha_before=pre_sha,
            commit_sha_after=post_sha,
        )

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self._cwd / candidate).resolve()

    def _write_bytes(self, path: str | Path, content: bytes) -> None:
        Path(path).write_bytes(content)

    def _delete_path(self, path: str | Path) -> None:
        Path(path).unlink()

    def _enter_advisory_mode(
        self,
        reason: str,
        *,
        error: GitOperationError | None = None,
    ) -> None:
        self._advisory_mode = True
        if self._warned_advisory:
            return
        payload: dict[str, object] = {
            "cwd": str(self._cwd),
            "reason": reason,
            "session_id": self._session_id,
        }
        if error is not None:
            payload["stderr"] = error.stderr
            payload["stdout"] = error.stdout
            payload["exit_code"] = error.exit_code
        logger.warning("resource writer advisory mode enabled", extra=payload)
        self._warned_advisory = True

    def _ensure_git_ready(self) -> None:
        if self._initial_snapshot_needed:
            self._run_git_sync(("add", "-A"))
            if self._is_index_clean():
                self._commit("agentm: initial snapshot", "agent", allow_empty=True)
            else:
                self._commit("agentm: initial snapshot", "agent")
            self._initial_snapshot_needed = False

    def _has_head_sync(self) -> bool:
        try:
            self._head_sha()
        except GitOperationError:
            return False
        return True

    def _head_sha(self) -> str:
        result = self._run_git_sync(("rev-parse", "HEAD"))
        return result.stdout.strip()

    def _snapshot_human_if_dirty(self, relative_path: str) -> None:
        if not self._has_uncommitted_diff(relative_path):
            return
        self._stage_paths([relative_path])
        self._commit("auto: pre-agent snapshot", "human")

    def _snapshot_human_for_dirty_paths(self, relative_paths: list[str]) -> None:
        dirty_paths = [
            relative_path
            for relative_path in relative_paths
            if self._has_uncommitted_diff(relative_path)
        ]
        if not dirty_paths:
            return
        self._stage_paths(dirty_paths)
        self._commit("auto: pre-agent snapshot", "human")

    def _has_uncommitted_diff(self, relative_path: str) -> bool:
        try:
            self._run_git_sync(("diff", "--quiet", "--", relative_path))
        except GitOperationError as exc:
            if exc.exit_code == 1:
                return True
            raise
        return False

    def _stage_paths(self, relative_paths: list[str]) -> None:
        self._run_git_sync(("add", "--", *relative_paths))

    def _is_index_clean(self) -> bool:
        try:
            self._run_git_sync(("diff", "--cached", "--quiet"))
        except GitOperationError as exc:
            if exc.exit_code == 1:
                return False
            raise
        return True

    def _is_index_clean_for_paths(self, relative_paths: list[str]) -> bool:
        try:
            self._run_git_sync(("diff", "--cached", "--quiet", "--", *relative_paths))
        except GitOperationError as exc:
            if exc.exit_code == 1:
                return False
            raise
        return True

    def _commit(
        self,
        message: str,
        author: WriterAuthor,
        *,
        allow_empty: bool = False,
    ) -> None:
        name, email = self._author_identity(author)
        args = ["commit", "-m", message, f"--author={name} <{email}>"]
        if allow_empty:
            args.append("--allow-empty")
        env = {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
        self._run_git_sync(tuple(args), env=env)

    def _restore_after_failure(
        self,
        resolved: Path,
        restore_exists: bool,
        restore_bytes: bytes | None,
        pre_sha: str | None,
        relative_path: str,
    ) -> None:
        try:
            if restore_exists:
                assert restore_bytes is not None
                resolved.write_bytes(restore_bytes)
            else:
                resolved.unlink(missing_ok=True)
            if pre_sha is not None:
                self._run_git_sync(("restore", "--source", pre_sha, "--", relative_path))
                if self._head_sha() != pre_sha:
                    self._run_git_sync(("reset", "--hard", pre_sha))
        except Exception:  # noqa: BLE001
            logger.exception("failed to restore resource after git write failure")

    def _restore_after_failure_batch(
        self,
        restore_states: list[_RestoreState],
        pre_sha: str | None,
        relative_paths: list[str],
    ) -> None:
        try:
            self._restore_paths(restore_states)
            if pre_sha is not None:
                self._run_git_sync(("restore", "--source", pre_sha, "--", *relative_paths))
                if self._head_sha() != pre_sha:
                    self._run_git_sync(("reset", "--hard", pre_sha))
        except Exception:  # noqa: BLE001
            logger.exception("failed to restore resource batch after git write failure")

    def _capture_restore_state(self, resolved: Path) -> _RestoreState:
        existed = resolved.exists()
        content = resolved.read_bytes() if existed else None
        return _RestoreState(resolved=resolved, existed=existed, content=content)

    def _capture_restore_states(self, resolved_paths: list[Path]) -> list[_RestoreState]:
        return [self._capture_restore_state(path) for path in resolved_paths]

    def _restore_paths(self, restore_states: list[_RestoreState]) -> None:
        for state in restore_states:
            if state.existed:
                assert state.content is not None
                state.resolved.parent.mkdir(parents=True, exist_ok=True)
                state.resolved.write_bytes(state.content)
            else:
                state.resolved.unlink(missing_ok=True)

    def _apply_pending_op(self, op: _PendingBatchOp, resolved: Path) -> None:
        if op.kind == "write":
            assert op.content is not None
            self._write_bytes(resolved, op.content)
            return
        if op.kind == "replace":
            assert op.old is not None
            assert op.new is not None
            current = resolved.read_bytes()
            if current != op.old:
                raise RuntimeError(
                    f"Current bytes for {op.path!r} no longer match expected content"
                )
            self._write_bytes(resolved, op.new)
            return
        self._delete_path(resolved)

    def _author_identity(self, author: WriterAuthor) -> tuple[str, str]:
        if author == "agent":
            return "agent", f"{self._session_id}@agentm"
        if author == "human":
            return "human", "human@agentm"
        return "indexer", "indexer@agentm"

    def _run_git_sync(
        self,
        args: tuple[str, ...],
        *,
        env: dict[str, str] | None = None,
    ) -> GitCommandResult:
        command = ["git", *self._git_prefix, *args]
        merged_env = os.environ.copy()
        if env is not None:
            merged_env.update(env)
        completed = subprocess.run(
            command,
            cwd=self._cwd,
            env=merged_env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise GitOperationError(
                args=args,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        return GitCommandResult(stdout=completed.stdout, stderr=completed.stderr)


__all__ = [
    "BatchHandle",
    "GitBackedResourceWriter",
    "GitOperationError",
    "PathClass",
    "ResourceWriter",
    "WriteResult",
    "WriterAuthor",
]
