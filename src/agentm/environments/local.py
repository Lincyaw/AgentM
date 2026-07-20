"""Local environment and snapshot lifecycle helpers."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from signal import SIGKILL
from typing import IO, Iterator

from loguru import logger

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.lifecycle import (
    EffectScope,
    EffectTxn,
    EnvironmentCheckpoint,
    EnvironmentFork,
    EnvironmentSnapshot,
    EnvironmentSnapshotter,
    LifecycleMeta,
)
from agentm.core.abi.operations import EnvironmentRef, ExecResult
from agentm.core.abi.trajectory import Turn, TurnRef


_DEFAULT_CONTROL_PLANE_EXCLUSIONS = (".agentm",)


class LocalBashOperations:
    """Local shell implementation backed by asyncio subprocesses."""

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        on_data: Callable[[bytes], None] | None = None,
        signal: CancelSignal | None = None,
        log_path: str | None = None,
    ) -> ExecResult:
        if not isinstance(cmd, str):
            raise TypeError("bash command must be a string")
        if not isinstance(cwd, str) or not cwd:
            raise TypeError("bash cwd must be a non-empty string")
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("bash timeout must be a finite positive number")
        if signal is not None and signal.is_set():
            raise asyncio.CancelledError("bash command interrupted before start")
        process = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        log_file: IO[bytes] | None = None
        if log_path is not None:
            try:
                resolved = Path(log_path)
                if not resolved.is_absolute():
                    resolved = Path(cwd) / resolved
                resolved.parent.mkdir(parents=True, exist_ok=True)
                log_file = resolved.open("ab")
            except OSError as exc:
                logger.debug("local bash: cannot open log {}: {}", log_path, exc)

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        timed_out = False

        async def read_stream(
            stream: asyncio.StreamReader,
            sink: list[bytes],
            callback: Callable[[bytes], None] | None = None,
        ) -> None:
            while chunk := await stream.read(65536):
                sink.append(chunk)
                if log_file is not None:
                    try:
                        log_file.write(chunk)
                        log_file.flush()
                    except OSError as exc:
                        logger.debug(
                            "local bash: cannot append log {}: {}",
                            log_path,
                            exc,
                        )
                if callback is not None:
                    callback(chunk)

        stdout_task = asyncio.create_task(
            read_stream(process.stdout, stdout_chunks, on_data)
        )
        stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_chunks))
        stdin_task = (
            asyncio.create_task(_write_stdin(process.stdin, stdin))
            if stdin is not None and process.stdin is not None
            else None
        )
        signal_task = asyncio.create_task(signal.wait()) if signal is not None else None
        wait_task = asyncio.create_task(process.wait())
        interrupted = False
        try:
            done, _ = await asyncio.wait(
                [task for task in (wait_task, signal_task) if task is not None],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task not in done:
                interrupted = signal_task is not None and signal_task in done
                timed_out = not interrupted
                await self._terminate_process_group(process)
                await _await_process_exit(wait_task)
            else:
                await wait_task
        except asyncio.CancelledError:
            await self._terminate_process_group(process)
            await _await_process_exit(wait_task)
            raise
        finally:
            try:
                if signal_task is not None and not signal_task.done():
                    signal_task.cancel()
                    await asyncio.gather(signal_task, return_exceptions=True)
                if stdin_task is not None:
                    await asyncio.gather(stdin_task, return_exceptions=True)
                await asyncio.gather(stdout_task, stderr_task)
            finally:
                if log_file is not None:
                    try:
                        log_file.close()
                    except OSError as exc:
                        logger.debug(
                            "local bash: cannot close log {}: {}",
                            log_path,
                            exc,
                        )

        if interrupted:
            raise asyncio.CancelledError("bash command interrupted")

        return ExecResult(
            stdout=b"".join(stdout_chunks),
            stderr=b"".join(stderr_chunks),
            exit_code=(
                process.returncode if process.returncode is not None else -SIGKILL
            ),
            timed_out=timed_out,
        )

    async def _terminate_process_group(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode is not None:
            return
        if process.pid is None:
            process.kill()
            return
        try:
            os.killpg(process.pid, SIGKILL)
        except ProcessLookupError:
            return


async def _write_stdin(
    writer: asyncio.StreamWriter,
    payload: bytes,
) -> None:
    try:
        writer.write(payload)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            pass


async def _await_process_exit(task: asyncio.Task[int]) -> int:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class LocalEnvironmentOperations:
    """Local ``EnvironmentOperations`` implementation with snapshot identity."""

    def __init__(
        self,
        *,
        cwd: str | Path,
        environment_id: str | None = None,
        bash: LocalBashOperations | None = None,
        snapshotter: EnvironmentSnapshotter | None = None,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self._cwd = Path(cwd)
        self._bash = bash or LocalBashOperations()
        self._snapshotter = snapshotter
        self._close_callback = close_callback
        self._closed = False
        self._ref = EnvironmentRef(
            id=environment_id or f"local:{_real_path(self._cwd)}",
            kind="local",
            metadata={"cwd": str(_real_path(self._cwd))},
        )

    @property
    def ref(self) -> EnvironmentRef:
        return self._ref

    @property
    def bash(self) -> LocalBashOperations:
        return self._bash

    async def snapshot(self) -> str | None:
        if self._snapshotter is None:
            return None
        snapshot = await self._snapshotter.snapshot(
            session_id=self._ref.id,
            ref=str(time.time()),
            metadata={"checkpoint": "fork"},
        )
        return snapshot.id

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            await asyncio.to_thread(self._close_callback)


class LocalSnapshotStore(EnvironmentSnapshotter):
    """Persist copy-based external-world snapshots under a local directory.

    ``.agentm`` is excluded by default because it is SDK control-plane state,
    not part of the environment world. Hosts that place trajectory, resource
    transaction, catalog, or snapshot state elsewhere inside the workspace
    must add those top-level names to ``excluded_names``.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        snapshot_root: str | Path,
        excluded_names: Sequence[str] = _DEFAULT_CONTROL_PLANE_EXCLUSIONS,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._snapshot_root = Path(snapshot_root)
        self._excluded_names = _validate_excluded_names(excluded_names)
        self._copy_policy_id = _copy_policy_id(self._excluded_names)
        self._lock_path = self._snapshot_root / "snapshot.lock"
        workspace_real = _real_path(self._workspace_root)
        snapshot_real = _real_path(self._snapshot_root)
        if snapshot_real == workspace_real or workspace_real in snapshot_real.parents:
            raise ValueError(
                "snapshot_root must be outside workspace_root so restore cannot "
                "delete its own checkpoint data"
            )
        self._snapshot_root.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot_root(self) -> Path:
        return self._snapshot_root

    async def snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef,
        metadata: LifecycleMeta | None = None,
    ) -> EnvironmentSnapshot:
        return await asyncio.to_thread(
            self._snapshot,
            session_id,
            ref,
            metadata,
        )

    def _snapshot(
        self,
        session_id: str,
        ref: TurnRef,
        metadata: LifecycleMeta | None,
    ) -> EnvironmentSnapshot:
        with self._locked():
            snapshot_id = (
                f"{_ref_token(session_id)}-{_ref_token(ref)}-{uuid.uuid4().hex}"
            )
            target = self._snapshot_dir(snapshot_id) / "workspace"
            _copytree(
                self._workspace_root,
                target,
                excluded_names=self._excluded_names,
            )
            snapshot = EnvironmentSnapshot(
                id=snapshot_id,
                session_id=session_id,
                ref=ref,
                metadata={
                    "kind": "local_copy",
                    "path": str(target),
                    "created_at": time.time(),
                    "copy_policy_id": self._copy_policy_id,
                    **dict(metadata or {}),
                },
            )
            _write_manifest(
                self._snapshot_dir(snapshot_id) / "snapshot.json",
                snapshot,
            )
            return snapshot

    async def fork_from(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        child_session_id: str,
    ) -> EnvironmentFork | None:
        fork_data = await asyncio.to_thread(
            self._prepare_fork,
            snapshot,
            child_session_id,
        )
        if fork_data is None:
            return None
        forked_snapshot, child_workspace = fork_data
        child_snapshotter = LocalSnapshotStore(
            workspace_root=child_workspace,
            snapshot_root=self._snapshot_root,
            excluded_names=tuple(self._excluded_names),
        )
        child_scope = LocalSnapshotEffectScope(
            snapshotter=child_snapshotter,
            session_id=child_session_id,
            snapshots={forked_snapshot.ref: forked_snapshot},
        )
        bash = LocalBashOperations()
        return EnvironmentFork(
            effect_scope=child_scope,
            cwd=str(child_workspace),
            lease=_LocalEnvironmentForkLease(
                snapshotter=self,
                snapshot=forked_snapshot,
                workspace=child_workspace,
            ),
            operations=LocalEnvironmentOperations(
                cwd=child_workspace,
                bash=bash,
                snapshotter=child_snapshotter,
                close_callback=lambda: _rmtree_and_fsync(child_workspace),
            ),
        )

    def _prepare_fork(
        self,
        snapshot: EnvironmentSnapshot,
        child_session_id: str,
    ) -> tuple[EnvironmentSnapshot, Path] | None:
        with self._locked():
            source = self._snapshot_source(snapshot)
            if not source.exists():
                return None
            turn_id = snapshot.metadata.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                raise RuntimeError(f"snapshot {snapshot.id!r} has no committed turn id")
            snapshot_id = f"{_ref_token(child_session_id)}-fork-{uuid.uuid4().hex}"
            target = self._snapshot_dir(snapshot_id) / "workspace"
            _copytree(
                source,
                target,
                excluded_names=self._excluded_names,
            )
            forked = EnvironmentSnapshot(
                id=snapshot_id,
                session_id=child_session_id,
                ref=snapshot.ref,
                metadata={
                    "kind": "local_copy",
                    "path": str(target),
                    "source_snapshot_id": snapshot.id,
                    "created_at": time.time(),
                    "checkpoint": "fork",
                    "turn_id": turn_id,
                    "copy_policy_id": self._copy_policy_id,
                },
            )
            _write_manifest(
                self._snapshot_dir(snapshot_id) / "snapshot.json",
                forked,
            )
            workspace = self._materialize_workspace(forked, child_session_id)
            return forked, workspace

    async def restore_to(self, snapshot: EnvironmentSnapshot) -> None:
        await asyncio.to_thread(self._restore_to, snapshot)

    def _restore_to(self, snapshot: EnvironmentSnapshot) -> None:
        with self._locked():
            source = self._snapshot_source(snapshot)
            if not source.exists():
                raise FileNotFoundError(source)
            _restore_tree(
                source,
                self._workspace_root,
                excluded_names=self._excluded_names,
            )

    async def find_snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef | None = None,
        checkpoint: EnvironmentCheckpoint,
    ) -> EnvironmentSnapshot | None:
        return await asyncio.to_thread(
            self._find_snapshot,
            session_id,
            ref,
            checkpoint,
        )

    def _find_snapshot(
        self,
        session_id: str,
        ref: TurnRef | None,
        checkpoint: EnvironmentCheckpoint,
    ) -> EnvironmentSnapshot | None:
        with self._locked():
            matches: list[EnvironmentSnapshot] = []
            for manifest_path in self._snapshot_root.glob("*/snapshot.json"):
                snapshot = _read_snapshot_manifest(manifest_path)
                if snapshot.id != manifest_path.parent.name:
                    raise ValueError(
                        f"snapshot identity does not match directory: {manifest_path}"
                    )
                self._snapshot_source(snapshot)
                if snapshot.session_id != session_id:
                    continue
                if snapshot.metadata.get("checkpoint") != checkpoint:
                    continue
                turn_id = snapshot.metadata.get("turn_id")
                if ref is not None and snapshot.ref != ref and turn_id != ref:
                    continue
                matches.append(snapshot)
            return max(
                matches,
                key=_snapshot_created_at,
                default=None,
            )

    async def discard(self, snapshot: EnvironmentSnapshot) -> None:
        await asyncio.to_thread(self._discard, snapshot)

    def _discard(self, snapshot: EnvironmentSnapshot) -> None:
        with self._locked():
            snapshot_dir = self._snapshot_dir(snapshot.id)
            if not snapshot_dir.exists():
                raise FileNotFoundError(snapshot_dir)
            _rmtree_and_fsync(snapshot_dir)

    def _materialize_workspace(
        self,
        snapshot: EnvironmentSnapshot,
        child_session_id: str,
    ) -> Path:
        source = self._snapshot_source(snapshot)
        if not source.exists():
            raise RuntimeError(f"snapshot {snapshot.id!r} has no materializable world")
        workspace = (
            self._snapshot_root
            / "forked_workspaces"
            / f"{_ref_token(child_session_id)}-{uuid.uuid4().hex}"
        )
        _copytree(
            source,
            workspace,
            excluded_names=self._excluded_names,
        )
        return workspace

    def _snapshot_source(self, snapshot: EnvironmentSnapshot) -> Path:
        policy_id = snapshot.metadata.get("copy_policy_id")
        if policy_id != self._copy_policy_id:
            raise ValueError(f"snapshot {snapshot.id!r} uses a different copy policy")
        path = snapshot.metadata.get("path")
        if not isinstance(path, str):
            raise ValueError(f"snapshot {snapshot.id!r} has no local path")
        source = _real_path(Path(path))
        expected = _real_path(self._snapshot_dir(snapshot.id) / "workspace")
        if source != expected:
            raise ValueError(
                f"snapshot {snapshot.id!r} points outside its snapshot directory"
            )
        return source

    def _snapshot_dir(self, snapshot_id: str) -> Path:
        if (
            not snapshot_id
            or snapshot_id in {".", ".."}
            or "/" in snapshot_id
            or "\0" in snapshot_id
        ):
            raise ValueError(f"invalid snapshot id: {snapshot_id!r}")
        return self._snapshot_root / snapshot_id

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as exc:
                raise RuntimeError("snapshot store file locking failed") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _LocalEnvironmentForkLease:
    """Remove every local artifact owned only by an unreturned child."""

    def __init__(
        self,
        *,
        snapshotter: LocalSnapshotStore,
        snapshot: EnvironmentSnapshot,
        workspace: Path,
    ) -> None:
        self._snapshotter = snapshotter
        self._snapshot = snapshot
        self._workspace = workspace

    async def commit(self) -> None:
        await asyncio.to_thread(self._commit)

    def _commit(self) -> None:
        with self._snapshotter._locked():
            manifest_path = (
                self._snapshotter._snapshot_dir(self._snapshot.id) / "snapshot.json"
            )
            current = _read_snapshot_manifest(manifest_path)
            checkpoint = current.metadata.get("checkpoint")
            if checkpoint == "after_turn":
                return
            if checkpoint != "fork":
                raise RuntimeError(
                    f"environment fork {current.id!r} has invalid checkpoint "
                    f"{checkpoint!r}"
                )
            turn_id = current.metadata.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                raise RuntimeError(
                    f"environment fork {current.id!r} has no committed turn id"
                )
            committed = replace(
                current,
                metadata={
                    **dict(current.metadata),
                    "checkpoint": "after_turn",
                },
            )
            _write_manifest(manifest_path, committed)

    async def abandon(self) -> None:
        await asyncio.to_thread(self._abandon)

    def _abandon(self) -> None:
        with self._snapshotter._locked():
            _rmtree_and_fsync_if_present(self._workspace)
            _rmtree_and_fsync_if_present(
                self._snapshotter._snapshot_dir(self._snapshot.id)
            )


class LocalSnapshotEffectScope(EffectScope):
    """Recoverable serial effect scope for a local workspace."""

    def __init__(
        self,
        *,
        snapshotter: EnvironmentSnapshotter,
        session_id: str = "",
        snapshots: dict[TurnRef, EnvironmentSnapshot] | None = None,
    ) -> None:
        self._snapshotter = snapshotter
        self._session_id = session_id
        self._snapshots = dict(snapshots or {})
        self._before_by_token: dict[str, EnvironmentSnapshot] = {}
        self._after_by_token: dict[str, EnvironmentSnapshot] = {}
        self._turn_lock = asyncio.Lock()
        self._active_tokens: set[str] = set()

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        turn_index: int,
    ) -> EffectTxn:
        await self._turn_lock.acquire()
        try:
            self._session_id = session_id
            token = f"{session_id}:{turn_id}"
            before = await self._snapshotter.snapshot(
                session_id=session_id,
                ref=turn_index,
                metadata={
                    "checkpoint": "before_turn",
                    "turn_id": turn_id,
                },
            )
        except BaseException:
            self._turn_lock.release()
            raise
        self._before_by_token[token] = before
        self._active_tokens.add(token)
        return EffectTxn(
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
            token=token,
        )

    async def prepare_turn(self, txn: EffectTxn, turn: Turn) -> None:
        self._require_active(txn)
        snapshot = await self._snapshotter.snapshot(
            session_id=txn.session_id,
            ref=turn.index,
            metadata={
                "checkpoint": "after_turn",
                "turn_id": turn.id,
            },
        )
        self._after_by_token[txn.token] = snapshot

    async def commit_turn(self, txn: EffectTxn, turn: Turn) -> None:
        self._require_active(txn)
        try:
            snapshot = self._after_by_token.pop(txn.token, None)
            if snapshot is None:
                snapshot = await self._snapshotter.find_snapshot(
                    session_id=txn.session_id,
                    ref=turn.id,
                    checkpoint="after_turn",
                )
            if snapshot is None:
                raise RuntimeError(
                    f"effect transaction {txn.token} has no prepared snapshot"
                )
            self._snapshots[turn.index] = snapshot
            self._snapshots[turn.id] = snapshot
            before = self._before_by_token.pop(txn.token, None)
            if before is not None:
                await self._snapshotter.discard(before)
        finally:
            self._release_turn(txn.token)

    async def abandon_turn(self, txn: EffectTxn) -> None:
        self._require_active(txn)
        try:
            before = self._before_by_token.pop(txn.token, None)
            if before is None:
                before = await self._snapshotter.find_snapshot(
                    session_id=txn.session_id,
                    ref=txn.turn_id,
                    checkpoint="before_turn",
                )
            if before is None:
                raise RuntimeError(
                    f"effect transaction {txn.token} has no rollback snapshot"
                )
            await self._snapshotter.restore_to(before)
            after = self._after_by_token.pop(txn.token, None)
            if after is not None:
                await self._snapshotter.discard(after)
            await self._snapshotter.discard(before)
        finally:
            self._release_turn(txn.token)

    async def fork_at(
        self,
        ref: TurnRef,
        *,
        source_session_id: str,
        child_session_id: str,
    ) -> EnvironmentFork:
        async with self._turn_lock:
            snapshot = self._snapshots.get(ref)
            if snapshot is None:
                snapshot = await self._snapshotter.find_snapshot(
                    session_id=source_session_id,
                    ref=ref,
                    checkpoint="after_turn",
                )
            if snapshot is None:
                raise RuntimeError(
                    f"no committed environment snapshot for {source_session_id}:{ref}"
                )
            environment_fork = await self._snapshotter.fork_from(
                snapshot,
                child_session_id=child_session_id,
            )
        if environment_fork is None:
            raise RuntimeError(
                f"environment backend cannot fork snapshot {snapshot.id}"
            )
        return environment_fork

    async def restore(
        self,
        *,
        session_id: str,
        turns: Sequence[Turn],
    ) -> None:
        async with self._turn_lock:
            if not turns:
                rollback = await self._snapshotter.find_snapshot(
                    session_id=session_id,
                    checkpoint="before_turn",
                )
                if rollback is not None:
                    await self._snapshotter.restore_to(rollback)
                return
            last = turns[-1]
            snapshot = self._snapshots.get(last.index) or self._snapshots.get(last.id)
            if snapshot is None:
                snapshot = await self._snapshotter.find_snapshot(
                    session_id=session_id,
                    ref=last.id,
                    checkpoint="after_turn",
                )
            if snapshot is None:
                raise RuntimeError(
                    f"no environment snapshot for committed turn {last.id}"
                )
            await self._snapshotter.restore_to(snapshot)

    def _require_active(self, txn: EffectTxn) -> None:
        if txn.token not in self._active_tokens:
            raise RuntimeError(f"effect transaction {txn.token!r} is not active")

    def _release_turn(self, token: str) -> None:
        self._active_tokens.discard(token)
        if self._turn_lock.locked():
            self._turn_lock.release()


def _copytree(
    source: Path,
    target: Path,
    *,
    excluded_names: frozenset[str],
) -> None:
    if target.exists():
        raise FileExistsError(target)
    ignore = shutil.ignore_patterns(*excluded_names) if excluded_names else None
    shutil.copytree(source, target, ignore=ignore)


def _restore_tree(
    source: Path,
    target: Path,
    *,
    excluded_names: frozenset[str],
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    source_names = {child.name for child in source.iterdir()}
    for child in target.iterdir():
        if child.name in excluded_names or child.name in source_names:
            continue
        _remove_path(child)
    for source_child in source.iterdir():
        if source_child.name in excluded_names:
            continue
        target_child = target / source_child.name
        if source_child.is_dir():
            if target_child.exists() and not target_child.is_dir():
                _remove_path(target_child)
            _restore_tree(
                source_child,
                target_child,
                excluded_names=excluded_names,
            )
        else:
            if target_child.exists() and target_child.is_dir():
                _remove_path(target_child)
            target_child.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_child, target_child)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _snapshot_created_at(snapshot: EnvironmentSnapshot) -> float:
    value = snapshot.metadata.get("created_at")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"snapshot {snapshot.id!r} has invalid created_at metadata")
    return float(value)


def _write_manifest(path: Path, snapshot: EnvironmentSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": snapshot.id,
        "session_id": snapshot.session_id,
        "ref": snapshot.ref,
        "metadata": dict(snapshot.metadata),
    }
    descriptor, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_snapshot_manifest(path: Path) -> EnvironmentSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot manifest must be an object: {path}")
    snapshot_id = payload.get("id")
    session_id = payload.get("session_id")
    ref = payload.get("ref")
    metadata = payload.get("metadata")
    if not isinstance(snapshot_id, str) or not isinstance(session_id, str):
        raise ValueError(f"snapshot manifest has invalid identity: {path}")
    if isinstance(ref, bool) or not isinstance(ref, (str, int)):
        raise ValueError(f"snapshot manifest has invalid turn ref: {path}")
    if not isinstance(metadata, dict):
        raise ValueError(f"snapshot manifest has invalid metadata: {path}")
    typed_metadata: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not (
            value is None or isinstance(value, (str, int, float, bool))
        ):
            raise ValueError(f"snapshot manifest has invalid metadata: {path}")
        typed_metadata[key] = value
    return EnvironmentSnapshot(
        id=snapshot_id,
        session_id=session_id,
        ref=ref,
        metadata=typed_metadata,
    )


def _ref_token(ref: TurnRef) -> str:
    return hashlib.sha256(str(ref).encode("utf-8")).hexdigest()[:24]


def _validate_excluded_names(value: Sequence[str]) -> frozenset[str]:
    names: set[str] = set()
    for name in value:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\0" in name
        ):
            raise ValueError(f"invalid snapshot exclusion name: {name!r}")
        names.add(name)
    return frozenset(names)


def _copy_policy_id(excluded_names: frozenset[str]) -> str:
    payload = json.dumps(sorted(excluded_names), separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _rmtree_and_fsync(path: Path) -> None:
    parent = path.parent
    shutil.rmtree(path)
    _fsync_directory(parent)


def _rmtree_and_fsync_if_present(path: Path) -> None:
    try:
        _rmtree_and_fsync(path)
    except FileNotFoundError:
        return


def _real_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "LocalBashOperations",
    "LocalEnvironmentOperations",
    "LocalSnapshotEffectScope",
    "LocalSnapshotStore",
]
