"""Git-backed chokepoint for writes to managed runtime resources."""

from __future__ import annotations

import asyncio
from loguru import logger
import os
import shutil
import subprocess
from collections.abc import Callable
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from agentm.core.abi import EventBus
from agentm.core.abi.resource import (
    BatchHandle,
    PathClass,
    ResourceWriter,
    WriteResult,
    WriterAuthor,
)
from agentm.core._internal.catalog.manifest import (
    CoreManifestPathUnsetError,
    is_constitution_path,
    load_core_manifest,
    matches_manifest_glob,
)
from agentm.core.abi.events import ResourceWriteEvent
from agentm.core.lib.paths import expand_path


# Author / committer identity overrides leak from the user's shell into git
# subprocesses by default; per `git-commit(1)` docs, GIT_AUTHOR_NAME silently
# overrides the `--author=` flag. We strip them so the writer's identity
# (`agent <session_id@agentm>`) is preserved verbatim. GPG_TTY is stripped so
# a global `commit.gpgsign=true` can't block on a passphrase prompt — the
# matching `commit.gpgsign=false` override is set per-call where applicable.
_GIT_ENV_BLOCKLIST: tuple[str, ...] = (
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_AUTHOR_DATE",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
    "GIT_COMMITTER_DATE",
    "GPG_TTY",
)


DEFAULT_PROTECTED_BRANCHES: frozenset[str] = frozenset({"main", "master"})


class ProtectedBranchError(RuntimeError):
    """Raised internally to signal a refusal to commit on a protected branch."""

    def __init__(self, branch: str) -> None:
        super().__init__(
            f"refusing to auto-commit to protected branch {branch!r}; "
            "switch to a non-protected branch, or construct the session with "
            "auto_commit=False to run the writer in advisory (no-commit) mode"
        )
        self.branch = branch


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


def _resolve_path(cwd: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (cwd / candidate).resolve()


def _write_bytes(path: str | Path, content: bytes) -> None:
    Path(path).write_bytes(content)


def _delete_path(path: str | Path) -> None:
    Path(path).unlink()


def _author_identity(session_id: str, author: WriterAuthor) -> tuple[str, str]:
    if author == "agent":
        return "agent", f"{session_id}@agentm"
    if author == "human":
        return "human", "human@agentm"
    return "indexer", "indexer@agentm"


def _capture_restore_state(resolved: Path) -> _RestoreState:
    existed = resolved.exists()
    content = resolved.read_bytes() if existed else None
    return _RestoreState(resolved=resolved, existed=existed, content=content)


def _capture_restore_states(resolved_paths: list[Path]) -> list[_RestoreState]:
    return [_capture_restore_state(path) for path in resolved_paths]


def _restore_paths(restore_states: list[_RestoreState]) -> None:
    for state in restore_states:
        if state.existed:
            assert state.content is not None
            state.resolved.parent.mkdir(parents=True, exist_ok=True)
            state.resolved.write_bytes(state.content)
        else:
            state.resolved.unlink(missing_ok=True)


def _apply_pending_op(op: _PendingBatchOp, resolved: Path) -> None:
    if op.kind == "write":
        assert op.content is not None
        _write_bytes(resolved, op.content)
        return
    if op.kind == "replace":
        assert op.old is not None
        assert op.new is not None
        current = resolved.read_bytes()
        if current != op.old:
            raise RuntimeError(
                f"Current bytes for {op.path!r} no longer match expected content"
            )
        _write_bytes(resolved, op.new)
        return
    _delete_path(resolved)


@dataclass(slots=True)
class _GitClient:
    cwd: Path
    session_id: str
    protected_branches: frozenset[str]
    prefix: tuple[str, ...] = ()

    def is_real_user_repo(self) -> bool:
        """True when commits would land in the user's real repo."""
        return self.prefix == ()

    def current_branch(self) -> str | None:
        """Resolve current branch via symbolic-ref. Returns None on detached HEAD."""
        try:
            result = self.run(("symbolic-ref", "--short", "HEAD"))
        except GitOperationError:
            return None
        branch = result.stdout.strip()
        return branch or None

    def check_protected_branch(self) -> None:
        """Raise ProtectedBranchError if we'd commit on a protected user branch."""
        if not self.is_real_user_repo():
            return
        if not self.protected_branches:
            return
        branch = self.current_branch()
        if branch is not None and branch in self.protected_branches:
            raise ProtectedBranchError(branch)

    def has_head(self) -> bool:
        try:
            self.head_sha()
        except GitOperationError:
            return False
        return True

    def head_sha(self) -> str:
        result = self.run(("rev-parse", "HEAD"))
        return result.stdout.strip()

    def snapshot_human_if_dirty(self, relative_path: str) -> None:
        if not self.has_uncommitted_diff(relative_path):
            return
        self.stage_paths([relative_path])
        self.commit("auto: pre-agent snapshot", "human")

    def snapshot_human_for_dirty_paths(self, relative_paths: list[str]) -> None:
        dirty_paths = [
            relative_path
            for relative_path in relative_paths
            if self.has_uncommitted_diff(relative_path)
        ]
        if not dirty_paths:
            return
        self.stage_paths(dirty_paths)
        self.commit("auto: pre-agent snapshot", "human")

    def has_uncommitted_diff(self, relative_path: str) -> bool:
        try:
            self.run(("diff", "--quiet", "--", relative_path))
        except GitOperationError as exc:
            if exc.exit_code == 1:
                return True
            raise
        return False

    def stage_paths(self, relative_paths: list[str]) -> None:
        self.run(("add", "--", *relative_paths))

    def is_index_clean(self) -> bool:
        try:
            self.run(("diff", "--cached", "--quiet"))
        except GitOperationError as exc:
            if exc.exit_code == 1:
                return False
            raise
        return True

    def is_index_clean_for_paths(self, relative_paths: list[str]) -> bool:
        try:
            self.run(("diff", "--cached", "--quiet", "--", *relative_paths))
        except GitOperationError as exc:
            if exc.exit_code == 1:
                return False
            raise
        return True

    def commit(
        self,
        message: str,
        author: WriterAuthor,
        *,
        allow_empty: bool = False,
    ) -> None:
        name, email = _author_identity(self.session_id, author)
        args = ["commit", "-m", message, f"--author={name} <{email}>"]
        if allow_empty:
            args.append("--allow-empty")
        env = {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
        self.run(tuple(args), env=env)

    def run(
        self,
        args: tuple[str, ...],
        *,
        env: dict[str, str] | None = None,
    ) -> GitCommandResult:
        command = ["git", *self.prefix, *args]
        merged_env = os.environ.copy()
        # Strip identity / signing leaks from the user's shell so the
        # writer's `--author` and synthetic committer identity are not
        # silently overridden (see `_GIT_ENV_BLOCKLIST` docstring).
        for key in _GIT_ENV_BLOCKLIST:
            merged_env.pop(key, None)
        # For the shadow-repo path we own the repo and don't want the
        # user's ~/.gitconfig (commit.gpgsign, core.hooksPath, signingkey,
        # ...) to interfere with internal commits. For the real-user-repo
        # path, leave global config alone — that's the user's repo and
        # their conventions apply.
        if not self.is_real_user_repo():
            merged_env.setdefault("GIT_CONFIG_GLOBAL", "/dev/null")
            merged_env.setdefault("GIT_CONFIG_SYSTEM", "/dev/null")
        if env is not None:
            merged_env.update(env)
        completed = subprocess.run(
            command,
            cwd=self.cwd,
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


def _restore_after_failure(
    git: _GitClient,
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
            git.run(("restore", "--source", pre_sha, "--", relative_path))
            if git.head_sha() != pre_sha:
                git.run(("reset", "--hard", pre_sha))
    except Exception:  # noqa: BLE001
        logger.exception("failed to restore resource after git write failure")


def _restore_after_failure_batch(
    git: _GitClient,
    restore_states: list[_RestoreState],
    pre_sha: str | None,
    relative_paths: list[str],
) -> None:
    try:
        _restore_paths(restore_states)
        if pre_sha is not None:
            git.run(("restore", "--source", pre_sha, "--", *relative_paths))
            if git.head_sha() != pre_sha:
                git.run(("reset", "--hard", pre_sha))
    except Exception:  # noqa: BLE001
        logger.exception("failed to restore resource batch after git write failure")


class GitBackedResourceWriter:
    """Single chokepoint for managed-resource writes."""

    def __init__(
        self,
        *,
        cwd: str,
        session_id: str,
        bus: EventBus,
        auto_commit: bool = True,
        protected_branches: frozenset[str] = DEFAULT_PROTECTED_BRANCHES,
    ) -> None:
        # Lazy-init contract: __init__ is a pure data-copy. No subprocess
        # spawns, no stat calls, no mkdir. All disk probing and shadow-repo
        # creation is deferred to `_lazy_setup()`, which fires on the first
        # mutating operation (write/replace/delete/batch/restore). Scenarios
        # that only read or classify pay zero filesystem cost.
        self._cwd = expand_path(cwd).resolve()
        self._session_id = session_id
        self._bus = bus
        self._auto_commit = auto_commit
        self._shadow_git_dir = self._cwd / ".agentm" / "repo"
        self._git = _GitClient(
            cwd=self._cwd,
            session_id=session_id,
            protected_branches=protected_branches,
        )
        self._advisory_mode = False
        self._warned_advisory = False
        self._initial_snapshot_needed = False
        self._setup_done = False

    def _lazy_setup(self) -> None:
        """Perform the disk-touching init work on first mutation.

        Idempotent: subsequent calls no-op via ``_setup_done``. Notes on
        deferred semantics vs. eager init: the ``auto_commit=False`` /
        ``git missing`` advisory-mode warnings now fire on the first write
        attempt rather than at construction time. Same once-per-session
        guarantee (``_warned_advisory``) still holds.
        """
        if self._setup_done:
            return
        self._setup_done = True

        git_bin = shutil.which("git")
        if git_bin is None:
            self._enter_advisory_mode("git binary missing")
            return

        # auto_commit=False short-circuits everything below: we treat the
        # working tree as read-write-through and never call `git commit`.
        # Managed paths fall into the existing advisory-mode write-through.
        if not self._auto_commit:
            self._enter_advisory_mode("auto_commit disabled")
            return

        if (self._cwd / ".git").exists():
            self._initial_snapshot_needed = not self._git.has_head()
            return

        self._shadow_git_dir.parent.mkdir(parents=True, exist_ok=True)
        if self._shadow_git_dir.exists():
            self._git.prefix = (
                f"--git-dir={self._shadow_git_dir}",
                f"--work-tree={self._cwd}",
            )
            self._initial_snapshot_needed = not self._git.has_head()
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
        self._git.prefix = (
            f"--git-dir={self._shadow_git_dir}",
            f"--work-tree={self._cwd}",
        )
        self._initial_snapshot_needed = True

    async def read(self, path: str) -> bytes:
        return await asyncio.to_thread(_resolve_path(self._cwd, path).read_bytes)

    async def exists(self, path: str) -> bool:
        resolved = _resolve_path(self._cwd, path)
        return await asyncio.to_thread(
            lambda: resolved.exists() and os.access(resolved, os.R_OK)
        )

    async def list_dir(self, path: str) -> list[str]:
        def _list() -> list[str]:
            return sorted(e.name for e in _resolve_path(self._cwd, path).iterdir())

        return await asyncio.to_thread(_list)

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        return await self._classified_write(
            path=path,
            rationale=rationale,
            author=author,
            raw_op=lambda r: _write_bytes(r, content),
            commit_op=lambda r: _write_bytes(r, content),
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
        await asyncio.to_thread(self._lazy_setup)
        path_class = self.classify(path)
        resolved = _resolve_path(self._cwd, path)
        if path_class == "constitution":
            return WriteResult._error(
                path, path_class, f"Refusing to modify constitution path {path!r}"
            )
        try:
            current = await asyncio.to_thread(resolved.read_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.debug("resource_writer: read_bytes failed for {}: {}", path, exc)
            return WriteResult._error(path, path_class, str(exc))
        if current != old:
            return WriteResult._error(
                path,
                path_class,
                f"Current bytes for {path!r} no longer match expected content",
            )
        if path_class == "unmanaged" or self._advisory_mode:
            try:
                await asyncio.to_thread(_write_bytes, resolved, new)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "resource_writer: write_bytes failed for {}: {}", path, exc
                )
                return WriteResult._error(path, path_class, str(exc))
            return WriteResult._uncommitted(path, path_class)

        return await self._commit_single_path(
            path=path,
            rationale=rationale,
            author=author,
            write_op=lambda _resolved: _write_bytes(resolved, new),
        )

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        return await self._classified_write(
            path=path,
            rationale=rationale,
            author=author,
            raw_op=lambda r: _delete_path(r),
            commit_op=lambda r: _delete_path(r),
        )

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

    def current_version_for_path(self, path: str) -> str | None:
        # Read-only by contract: must never trigger _lazy_setup() or touch
        # disk. With no writes yet, there is no commit and therefore no
        # version to report — return None. Observability calls this on
        # every session entry; touching disk here would defeat lazy-init.
        if not self._setup_done:
            return None
        path_class = self.classify(path)
        if path_class != "managed" or self._advisory_mode:
            return None

        resolved = _resolve_path(self._cwd, path)
        try:
            relative = resolved.relative_to(self._cwd)
        except ValueError:
            return None

        try:
            self._ensure_git_ready()
            result = self._git.run(
                (
                    "log",
                    "-n",
                    "1",
                    "--format=%H",
                    "--",
                    PurePosixPath(relative).as_posix(),
                )
            )
        except GitOperationError:
            return None

        sha = result.stdout.strip()
        return sha or None

    def restore(self, path: Path, version: str) -> None:
        self._lazy_setup()
        if self._advisory_mode or self.classify(str(path)) != "managed":
            raise NotImplementedError(
                "git rollback requires a versioned ResourceWriter"
            )

        resolved = _resolve_path(self._cwd, str(path))
        relative = resolved.relative_to(self._cwd)
        relative_posix = PurePosixPath(relative).as_posix()
        self._ensure_git_ready()
        self._git.run(("restore", "--source", version, "--", relative_posix))
        self._git.run(("reset", "--hard", version))

    async def _apply_batch(
        self,
        pending: list[_PendingBatchOp],
        *,
        rationale: str,
        author: WriterAuthor,
    ) -> None:
        if not pending:
            return

        await asyncio.to_thread(self._lazy_setup)

        managed_ops: list[tuple[_PendingBatchOp, Path, str]] = []
        unmanaged_ops: list[tuple[_PendingBatchOp, Path]] = []

        for op in pending:
            path_class = self.classify(op.path)
            if path_class == "constitution":
                raise RuntimeError(f"Refusing to modify constitution path {op.path!r}")

            resolved = _resolve_path(self._cwd, op.path)
            if path_class == "managed" and not self._advisory_mode:
                relative = resolved.relative_to(self._cwd)
                relative_posix = PurePosixPath(relative).as_posix()
                managed_ops.append((op, resolved, relative_posix))
            else:
                unmanaged_ops.append((op, resolved))

        # Refuse the entire batch up-front if any managed op would commit to
        # a protected branch — surfacing the failure before mutating any
        # unmanaged paths keeps the batch atomic from the caller's view.
        if managed_ops:
            await asyncio.to_thread(self._git.check_protected_branch)

        unmanaged_restore: list[_RestoreState] = []
        try:
            for op, resolved in unmanaged_ops:
                unmanaged_restore.append(
                    await asyncio.to_thread(_capture_restore_state, resolved)
                )
                await asyncio.to_thread(_apply_pending_op, op, resolved)
        except Exception:  # noqa: BLE001
            await asyncio.to_thread(_restore_paths, unmanaged_restore)
            raise

        if not managed_ops:
            return

        await asyncio.to_thread(self._ensure_git_ready)
        pre_sha = await asyncio.to_thread(self._git.head_sha)
        managed_restore = await asyncio.to_thread(
            _capture_restore_states,
            [resolved for _, resolved, _ in managed_ops],
        )

        try:
            await asyncio.to_thread(
                self._git.snapshot_human_for_dirty_paths,
                [relative_posix for _, _, relative_posix in managed_ops],
            )
            for op, resolved, _ in managed_ops:
                await asyncio.to_thread(_apply_pending_op, op, resolved)

            relative_paths = [relative_posix for _, _, relative_posix in managed_ops]
            await asyncio.to_thread(self._git.stage_paths, relative_paths)
            if await asyncio.to_thread(
                self._git.is_index_clean_for_paths, relative_paths
            ):
                return

            await asyncio.to_thread(self._git.commit, rationale, author)
            post_sha = await asyncio.to_thread(self._git.head_sha)
        except Exception as exc:
            logger.warning("resource_writer: batch commit failed, restoring: {}", exc)
            await asyncio.to_thread(
                _restore_after_failure_batch,
                self._git,
                managed_restore,
                pre_sha,
                [relative_posix for _, _, relative_posix in managed_ops],
            )
            await asyncio.to_thread(_restore_paths, unmanaged_restore)
            raise

        for _, _, relative_posix in managed_ops:
            await self._bus.emit(
                ResourceWriteEvent.CHANNEL,
                ResourceWriteEvent(
                    path=relative_posix,
                    pre_sha=pre_sha,
                    post_sha=post_sha,
                    rationale=rationale,
                    author=author,
                ),
            )

    async def _classified_write(
        self,
        *,
        path: str,
        rationale: str,
        author: WriterAuthor,
        raw_op: Callable[[Path], None],
        commit_op: Callable[[Path], None],
    ) -> WriteResult:
        """Common classify-then-dispatch for write/delete.

        ``raw_op`` runs for unmanaged/advisory paths, ``commit_op`` runs
        inside a managed git commit.  ``replace`` has extra pre-validation
        (read + compare) that doesn't fit this shape, so it calls the
        individual pieces directly.
        """
        await asyncio.to_thread(self._lazy_setup)
        path_class = self.classify(path)
        resolved = _resolve_path(self._cwd, path)
        if path_class == "constitution":
            return WriteResult._error(
                path, path_class, f"Refusing to modify constitution path {path!r}"
            )
        if path_class == "unmanaged" or self._advisory_mode:
            try:
                await asyncio.to_thread(raw_op, resolved)
            except Exception as exc:  # noqa: BLE001
                logger.debug("resource_writer: raw_op failed for {}: {}", path, exc)
                return WriteResult._error(path, path_class, str(exc))
            return WriteResult._uncommitted(path, path_class)

        return await self._commit_single_path(
            path=path,
            rationale=rationale,
            author=author,
            write_op=commit_op,
        )

    async def _commit_single_path(
        self,
        *,
        path: str,
        rationale: str,
        author: WriterAuthor,
        write_op: Callable[[Path], None],
    ) -> WriteResult:
        resolved = _resolve_path(self._cwd, path)
        relative = resolved.relative_to(self._cwd)
        relative_posix = PurePosixPath(relative).as_posix()
        try:
            await asyncio.to_thread(self._git.check_protected_branch)
        except ProtectedBranchError as exc:
            return WriteResult._error(path, "managed", str(exc))
        await asyncio.to_thread(self._ensure_git_ready)
        pre_sha = await asyncio.to_thread(self._git.head_sha)
        restore_exists = resolved.exists()
        restore_bytes = (
            await asyncio.to_thread(resolved.read_bytes) if restore_exists else None
        )
        try:
            await asyncio.to_thread(self._git.snapshot_human_if_dirty, relative_posix)
            await asyncio.to_thread(write_op, resolved)
            await asyncio.to_thread(self._git.stage_paths, [relative_posix])
            if await asyncio.to_thread(
                self._git.is_index_clean_for_paths, [relative_posix]
            ):
                current_sha = await asyncio.to_thread(self._git.head_sha)
                return WriteResult._uncommitted(
                    path,
                    "managed",
                    commit_sha_before=pre_sha,
                    commit_sha_after=current_sha,
                )
            await asyncio.to_thread(self._git.commit, rationale, author)
            post_sha = await asyncio.to_thread(self._git.head_sha)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "resource_writer: commit failed for {}, restoring: {}", path, exc
            )
            await asyncio.to_thread(
                _restore_after_failure,
                self._git,
                resolved,
                restore_exists,
                restore_bytes,
                pre_sha,
                relative_posix,
            )
            return WriteResult._error(
                path, "managed", str(exc), commit_sha_before=pre_sha
            )

        await self._bus.emit(
            ResourceWriteEvent.CHANNEL,
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
            self._git.check_protected_branch()
            self._git.run(("add", "-A"))
            if self._git.is_index_clean():
                self._git.commit(
                    "agentm: initial snapshot", "agent", allow_empty=True
                )
            else:
                self._git.commit("agentm: initial snapshot", "agent")
            self._initial_snapshot_needed = False


__all__ = [
    "BatchHandle",
    "DEFAULT_PROTECTED_BRANCHES",
    "GitBackedResourceWriter",
    "GitOperationError",
    "PathClass",
    "ProtectedBranchError",
    "ResourceWriter",
    "WriteResult",
    "WriterAuthor",
]
