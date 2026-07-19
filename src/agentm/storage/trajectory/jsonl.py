"""Local JSONL sidecar implementation of ``TrajectoryNodeStore``."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentm.core.abi.store import TrajectoryNodeQuery
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_HEAD_ID,
    ContentReplacementState,
    PromptCacheState,
    TRAJECTORY_HEAD_INDEXES,
    TRAJECTORY_NODE_INDEXES,
    TrajectoryBranchId,
    TrajectoryHead,
    TrajectoryHeadAdvance,
    TrajectoryHeadId,
    TrajectoryIndexSpec,
    TrajectoryLeaf,
    TrajectoryNode,
    TrajectoryProjectionStatus,
)
from agentm.core.lib.trajectory_nodes import InMemoryTrajectoryNodeStore
from agentm.storage.serialization import (
    deserialize_content_state,
    deserialize_head,
    deserialize_node,
    deserialize_projection_status,
    deserialize_prompt_cache_state,
    serialize_content_state,
    serialize_head,
    serialize_node,
    serialize_projection_status,
    serialize_prompt_cache_state,
)


class JsonlTrajectoryNodeStore:
    """Durable local sidecar store for trajectory nodes and append heads.

    Every line in ``projection-journal.jsonl`` is an atomic snapshot of one
    session's nodes, heads, projection status, and cache/compaction state.
    Reload replays the latest complete record per session. A torn final line
    is ignored; corruption in any complete record fails startup.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._journal_path = self._root / "projection-journal.jsonl"
        self._lock_path = self._root / ".lock"
        self._process_lock = threading.RLock()
        self._store = InMemoryTrajectoryNodeStore()
        self._known_sessions: set[str] = set()
        self._content_states: dict[tuple[str, str], ContentReplacementState] = {}
        self._prompt_cache_states: dict[tuple[str, str], PromptCacheState] = {}
        self._root.mkdir(parents=True, exist_ok=True)
        with self._guard():
            self._reload_unlocked()

    @property
    def indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_NODE_INDEXES

    @property
    def head_indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_HEAD_INDEXES

    def append_nodes(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        advance_head: TrajectoryHeadAdvance | None = None,
    ) -> None:
        if not nodes:
            return
        with self._guard():
            self._reload_unlocked()
            self._store.append_nodes(
                session_id,
                nodes,
                advance_head=advance_head,
            )
            self._known_sessions.add(session_id)
            self._persist_unlocked(session_id)

    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        with self._guard():
            self._reload_unlocked()
            return self._store.query_nodes(query)

    def get_head(
        self,
        session_id: str,
        *,
        head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> TrajectoryHead | None:
        with self._guard():
            self._reload_unlocked()
            return self._store.get_head(
                session_id,
                head_id=head_id,
                branch_id=branch_id,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
            )

    def list_heads(
        self,
        session_id: str,
        *,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
        include_inactive: bool = False,
    ) -> list[TrajectoryHead]:
        with self._guard():
            self._reload_unlocked()
            return self._store.list_heads(
                session_id,
                branch_id=branch_id,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
                include_inactive=include_inactive,
            )

    def load_chain(
        self,
        session_id: str,
        leaf_node_id: str,
        *,
        include_logical_parent: bool = False,
    ) -> list[TrajectoryNode]:
        with self._guard():
            self._reload_unlocked()
            return self._store.load_chain(
                session_id,
                leaf_node_id,
                include_logical_parent=include_logical_parent,
            )

    def leaves(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> list[TrajectoryLeaf]:
        with self._guard():
            self._reload_unlocked()
            return self._store.leaves(
                session_id,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
            )

    def replace_session_projection(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        heads: Sequence[TrajectoryHead] = (),
        status: TrajectoryProjectionStatus | None = None,
    ) -> None:
        with self._guard():
            self._reload_unlocked()
            self._store.replace_session_projection(
                session_id,
                nodes,
                heads=heads,
                status=status,
            )
            self._known_sessions.add(session_id)
            self._persist_unlocked(session_id)

    def projection_status(
        self,
        session_id: str,
    ) -> TrajectoryProjectionStatus | None:
        with self._guard():
            self._reload_unlocked()
            return self._store.projection_status(session_id)

    def save_content_replacement_state(
        self,
        session_id: str,
        state: ContentReplacementState,
    ) -> None:
        with self._guard():
            self._reload_unlocked()
            self._store.save_content_replacement_state(session_id, state)
            self._content_states[(session_id, state.state_key)] = state
            self._known_sessions.add(session_id)
            self._persist_unlocked(session_id)

    def load_content_replacement_state(
        self,
        session_id: str,
        state_key: str,
    ) -> ContentReplacementState | None:
        with self._guard():
            self._reload_unlocked()
            return self._store.load_content_replacement_state(session_id, state_key)

    def clone_content_replacement_state(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        state_key: str,
        target_leaf_id: str | None = None,
    ) -> ContentReplacementState | None:
        with self._guard():
            self._reload_unlocked()
            state = self._store.load_content_replacement_state(
                source_session_id,
                state_key,
            )
            if state is None:
                return None
            cloned = replace(
                state,
                source_session_id=source_session_id,
                source_leaf_id=state.leaf_node_id or state.source_leaf_id,
                leaf_node_id=target_leaf_id,
            )
            self._store.save_content_replacement_state(target_session_id, cloned)
            self._content_states[(target_session_id, cloned.state_key)] = cloned
            self._known_sessions.add(target_session_id)
            self._persist_unlocked(target_session_id)
            return cloned

    def save_prompt_cache_state(
        self,
        session_id: str,
        state: PromptCacheState,
    ) -> None:
        with self._guard():
            self._reload_unlocked()
            self._store.save_prompt_cache_state(session_id, state)
            self._prompt_cache_states[(session_id, state.cache_key)] = state
            self._known_sessions.add(session_id)
            self._persist_unlocked(session_id)

    def load_prompt_cache_state(
        self,
        session_id: str,
        cache_key: str,
    ) -> PromptCacheState | None:
        with self._guard():
            self._reload_unlocked()
            return self._store.load_prompt_cache_state(session_id, cache_key)

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._process_lock:
            self._root.mkdir(parents=True, exist_ok=True)
            with _FileLock(self._lock_path):
                yield

    def _reload_unlocked(self) -> None:
        store = InMemoryTrajectoryNodeStore()
        known_sessions: set[str] = set()
        latest = _read_latest_session_records(self._journal_path)
        content_states: dict[tuple[str, str], ContentReplacementState] = {}
        prompt_cache_states: dict[tuple[str, str], PromptCacheState] = {}
        for session_id, record in sorted(latest.items()):
            nodes = [
                deserialize_node(item)
                for item in _mapping_list(record, "nodes")
            ]
            heads = [
                deserialize_head(item)
                for item in _mapping_list(record, "heads")
            ]
            raw_status = record.get("status")
            status = (
                deserialize_projection_status(raw_status)
                if isinstance(raw_status, Mapping)
                else None
            )
            store.replace_session_projection(
                session_id,
                nodes,
                heads=heads,
                status=status,
            )
            known_sessions.add(session_id)
            for item in _mapping_list(record, "content_states"):
                content_state = deserialize_content_state(item)
                store.save_content_replacement_state(session_id, content_state)
                content_states[(session_id, content_state.state_key)] = (
                    content_state
                )
            for item in _mapping_list(record, "prompt_cache_states"):
                cache_state = deserialize_prompt_cache_state(item)
                store.save_prompt_cache_state(session_id, cache_state)
                prompt_cache_states[(session_id, cache_state.cache_key)] = (
                    cache_state
                )

        self._store = store
        self._known_sessions = known_sessions
        self._content_states = content_states
        self._prompt_cache_states = prompt_cache_states

    def _persist_unlocked(self, session_id: str) -> None:
        status = self._store.projection_status(session_id)
        record: dict[str, object] = {
            "version": 1,
            "session_id": session_id,
            "nodes": [
                serialize_node(node)
                for node in self._store.query_nodes(
                    TrajectoryNodeQuery(session_id=session_id)
                )
            ],
            "heads": [
                serialize_head(head)
                for head in self._store.list_heads(
                    session_id,
                    include_inactive=True,
                )
            ],
            "status": (
                serialize_projection_status(status)
                if status is not None
                else None
            ),
            "content_states": [
                serialize_content_state(state)
                for (state_session_id, _), state in sorted(
                    self._content_states.items()
                )
                if state_session_id == session_id
            ],
            "prompt_cache_states": [
                serialize_prompt_cache_state(state)
                for (state_session_id, _), state in sorted(
                    self._prompt_cache_states.items()
                )
                if state_session_id == session_id
            ],
        }
        _append_jsonl_record(self._journal_path, record)


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any | None = None

    def __enter__(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+b")
        try:
            import fcntl
        except ImportError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                "JsonlTrajectoryNodeStore requires OS file locking"
            ) from exc
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *_exc: object) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            import fcntl
        except ImportError as exc:
            handle.close()
            self._handle = None
            raise RuntimeError(
                "JsonlTrajectoryNodeStore lost OS file locking support"
            ) from exc
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


def _read_latest_session_records(
    path: Path,
) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    latest: dict[str, Mapping[str, Any]] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        complete = raw_line.endswith((b"\n", b"\r"))
        if line_no == len(lines) and not complete:
            break
        try:
            item = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError(f"{path}:{line_no} is not valid JSON") from None
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        if item.get("version") != 1:
            raise ValueError(f"{path}:{line_no} has unsupported journal version")
        session_id = item.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{path}:{line_no} has no session_id")
        latest[session_id] = item
    return latest


def _mapping_list(
    record: Mapping[str, Any],
    key: str,
) -> list[Mapping[str, Any]]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"trajectory journal record has invalid {key!r}")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"trajectory journal record has invalid {key!r} item")
    return list(value)


def _append_jsonl_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = ["JsonlTrajectoryNodeStore"]
