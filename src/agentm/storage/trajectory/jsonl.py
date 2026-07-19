"""Local JSONL sidecar implementation of ``TrajectoryNodeStore``."""

from __future__ import annotations

import json
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
from agentm.core.runtime.trajectory_nodes import InMemoryTrajectoryNodeStore
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

    The physical storage is intentionally simple:

    - ``nodes.jsonl`` stores one serialized node per line.
    - ``heads.json`` stores explicit append heads.
    - ``projection_status.json`` stores projection health.
    - ``content_replacement_states.json`` and ``prompt_cache_states.json`` store
      deterministic fork/resume/cache policy state.

    Mutations reload under a file lock, apply the same in-memory reference
    consistency rules, then atomically rewrite the sidecar. This keeps the
    local backend behavior aligned with the SDK-level protocol before SQL
    backends optimize the same schema with indexes.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._nodes_path = self._root / "nodes.jsonl"
        self._heads_path = self._root / "heads.json"
        self._projection_status_path = self._root / "projection_status.json"
        self._content_states_path = self._root / "content_replacement_states.json"
        self._prompt_cache_states_path = self._root / "prompt_cache_states.json"
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
            self._persist_unlocked()

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
            self._persist_unlocked()

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
            self._persist_unlocked()

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
            self._persist_unlocked()
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
            self._persist_unlocked()

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
        nodes_by_session: dict[str, list[TrajectoryNode]] = {}
        for item in _read_jsonl(self._nodes_path):
            node = deserialize_node(item)
            nodes_by_session.setdefault(node.session_id, []).append(node)
            known_sessions.add(node.session_id)

        heads_by_session: dict[str, list[TrajectoryHead]] = {}
        for item in _read_json_array(self._heads_path):
            head = deserialize_head(item)
            heads_by_session.setdefault(head.session_id, []).append(head)
            known_sessions.add(head.session_id)

        statuses: dict[str, TrajectoryProjectionStatus] = {}
        for item in _read_json_array(self._projection_status_path):
            status = deserialize_projection_status(item)
            statuses[status.session_id] = status
            known_sessions.add(status.session_id)

        for session_id in sorted(known_sessions):
            nodes = sorted(nodes_by_session.get(session_id, ()), key=lambda node: node.seq)
            store.replace_session_projection(
                session_id,
                nodes,
                heads=heads_by_session.get(session_id, ()),
                status=statuses.get(session_id),
            )

        content_states: dict[tuple[str, str], ContentReplacementState] = {}
        for item in _read_json_array(self._content_states_path):
            content_session_id = item.get("session_id")
            state_data = item.get("state")
            if not isinstance(content_session_id, str) or not isinstance(state_data, Mapping):
                continue
            content_state = deserialize_content_state(state_data)
            store.save_content_replacement_state(content_session_id, content_state)
            content_states[(content_session_id, content_state.state_key)] = content_state
            known_sessions.add(content_session_id)

        prompt_cache_states: dict[tuple[str, str], PromptCacheState] = {}
        for item in _read_json_array(self._prompt_cache_states_path):
            cache_session_id = item.get("session_id")
            cache_state_data = item.get("state")
            if not isinstance(cache_session_id, str) or not isinstance(cache_state_data, Mapping):
                continue
            cache_state = deserialize_prompt_cache_state(cache_state_data)
            store.save_prompt_cache_state(cache_session_id, cache_state)
            prompt_cache_states[(cache_session_id, cache_state.cache_key)] = cache_state
            known_sessions.add(cache_session_id)

        self._store = store
        self._known_sessions = known_sessions
        self._content_states = content_states
        self._prompt_cache_states = prompt_cache_states

    def _persist_unlocked(self) -> None:
        nodes = sorted(
            self._store.query_nodes(TrajectoryNodeQuery()),
            key=lambda node: (node.session_id, node.seq),
        )
        _atomic_write_text(
            self._nodes_path,
            "".join(
                json.dumps(serialize_node(node), sort_keys=True) + "\n"
                for node in nodes
            ),
        )
        heads = [
            head
            for session_id in sorted(self._known_sessions)
            for head in self._store.list_heads(session_id, include_inactive=True)
        ]
        _atomic_write_json_array(
            self._heads_path,
            [serialize_head(head) for head in heads],
        )
        statuses = [
            status
            for session_id in sorted(self._known_sessions)
            if (status := self._store.projection_status(session_id)) is not None
        ]
        _atomic_write_json_array(
            self._projection_status_path,
            [serialize_projection_status(status) for status in statuses],
        )
        _atomic_write_json_array(
            self._content_states_path,
            [
                {
                    "session_id": session_id,
                    "state": serialize_content_state(state),
                }
                for (session_id, _), state in sorted(self._content_states.items())
            ],
        )
        _atomic_write_json_array(
            self._prompt_cache_states_path,
            [
                {
                    "session_id": session_id,
                    "state": serialize_prompt_cache_state(state),
                }
                for (session_id, _), state in sorted(self._prompt_cache_states.items())
            ],
        )


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any | None = None

    def __enter__(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+b")
        try:
            import fcntl
        except ImportError:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *_exc: object) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            import fcntl
        except ImportError:
            handle.close()
            self._handle = None
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(item)
    return rows


def _read_json_array(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in value if isinstance(item, Mapping)]


def _atomic_write_json_array(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        json.dumps(list(rows), sort_keys=True, indent=2) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


__all__ = ["JsonlTrajectoryNodeStore"]
