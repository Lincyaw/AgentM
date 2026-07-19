"""Local environment and snapshot lifecycle helpers."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from collections.abc import Sequence

from agentm.core.abi.lifecycle import (
    EffectScope,
    EffectTxn,
    EnvironmentSnapshot,
    EnvironmentSnapshotter,
)
from agentm.core.abi.operations import EnvironmentRef
from agentm.core.abi.trajectory import Turn, TurnRef
from agentm.extensions.builtin.operations import LocalBashOperations


class LocalEnvironmentOperations:
    """Local ``EnvironmentOperations`` implementation with snapshot identity."""

    def __init__(
        self,
        *,
        cwd: str | Path,
        environment_id: str | None = None,
        bash: LocalBashOperations | None = None,
        snapshotter: EnvironmentSnapshotter | None = None,
    ) -> None:
        self._cwd = Path(cwd)
        self._bash = bash or LocalBashOperations()
        self._snapshotter = snapshotter
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
        )
        return snapshot.id

    async def close(self) -> None:
        return None


class LocalSnapshotStore(EnvironmentSnapshotter):
    """Persist copy-based filesystem snapshots under a local directory."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        snapshot_root: str | Path,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._snapshot_root = Path(snapshot_root)
        self._snapshot_root.mkdir(parents=True, exist_ok=True)

    async def snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef,
    ) -> EnvironmentSnapshot:
        snapshot_id = f"{session_id}-{_ref_token(ref)}-{int(time.time() * 1000)}"
        target = self._snapshot_root / snapshot_id / "workspace"
        _copytree(self._workspace_root, target)
        snapshot = EnvironmentSnapshot(
            id=snapshot_id,
            session_id=session_id,
            ref=ref,
            metadata={
                "kind": "local_copy",
                "path": str(target),
                "created_at": time.time(),
            },
        )
        _write_manifest(self._snapshot_root / snapshot_id / "snapshot.json", snapshot)
        return snapshot

    async def fork_from(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        child_session_id: str,
    ) -> EnvironmentSnapshot | None:
        source = _snapshot_path(snapshot)
        if source is None or not source.exists():
            return None
        snapshot_id = f"{child_session_id}-fork-{int(time.time() * 1000)}"
        target = self._snapshot_root / snapshot_id / "workspace"
        _copytree(source, target)
        forked = EnvironmentSnapshot(
            id=snapshot_id,
            session_id=child_session_id,
            ref=snapshot.ref,
            metadata={
                "kind": "local_copy",
                "path": str(target),
                "source_snapshot_id": snapshot.id,
                "created_at": time.time(),
            },
        )
        _write_manifest(self._snapshot_root / snapshot_id / "snapshot.json", forked)
        return forked

    async def restore_to(self, snapshot: EnvironmentSnapshot) -> None:
        source = _snapshot_path(snapshot)
        if source is None:
            raise ValueError("snapshot does not contain a local path")
        if not source.exists():
            raise FileNotFoundError(source)
        _copytree(source, self._workspace_root)


class LocalSnapshotEffectScope(EffectScope):
    """Effect scope that snapshots local workspace state after committed turns."""

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

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        turn_index: int,
    ) -> EffectTxn:
        self._session_id = session_id
        return EffectTxn(
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
            token=f"{session_id}:{turn_id}",
        )

    async def commit_turn(self, txn: EffectTxn, turn: Turn) -> None:
        snapshot = await self._snapshotter.snapshot(
            session_id=txn.session_id,
            ref=turn.index,
        )
        self._snapshots[turn.index] = snapshot
        self._snapshots[turn.id] = snapshot

    async def abandon_turn(self, txn: EffectTxn) -> None:
        del txn

    async def fork_at(
        self,
        ref: TurnRef,
        *,
        source_session_id: str,
        child_session_id: str,
    ) -> "LocalSnapshotEffectScope":
        del source_session_id
        snapshot = self._snapshots.get(ref)
        forked_snapshot = (
            await self._snapshotter.fork_from(
                snapshot,
                child_session_id=child_session_id,
            )
            if snapshot is not None
            else None
        )
        return LocalSnapshotEffectScope(
            snapshotter=self._snapshotter,
            session_id=child_session_id,
            snapshots={ref: forked_snapshot} if forked_snapshot is not None else {},
        )

    async def restore(
        self,
        *,
        session_id: str,
        turns: Sequence[Turn],
    ) -> None:
        del session_id
        if not turns:
            return
        snapshot = self._snapshots.get(turns[-1].index) or self._snapshots.get(turns[-1].id)
        if snapshot is not None:
            await self._snapshotter.restore_to(snapshot)


def _copytree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".venv", "node_modules")
    shutil.copytree(source, target, ignore=ignore)


def _snapshot_path(snapshot: EnvironmentSnapshot) -> Path | None:
    path = snapshot.metadata.get("path")
    return Path(path) if isinstance(path, str) else None


def _write_manifest(path: Path, snapshot: EnvironmentSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": snapshot.id,
        "session_id": snapshot.session_id,
        "ref": snapshot.ref,
        "metadata": dict(snapshot.metadata),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _ref_token(ref: TurnRef) -> str:
    return str(ref).replace("/", "_").replace(":", "_")


def _real_path(path: Path) -> Path:
    return path.expanduser().resolve()


__all__ = [
    "LocalEnvironmentOperations",
    "LocalSnapshotEffectScope",
    "LocalSnapshotStore",
]
