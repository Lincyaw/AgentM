# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""Crash-aware JSONL implementation of the unified trajectory store."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.store import (
    SessionMeta,
    TrajectoryCommit,
    TrajectoryCompactionCommit,
    TrajectoryNodeQuery,
)
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
    Turn,
    TurnCheckpoint,
    TurnRef,
)
from agentm.storage.trajectory.memory import InMemoryTrajectoryStore
from agentm.storage.serialization import (
    deserialize_content_state,
    deserialize_head,
    deserialize_node,
    deserialize_prompt_cache_state,
    serialize_content_state,
    serialize_head,
    serialize_node,
    serialize_prompt_cache_state,
)

_VERSION = 2
_SESSION = "session"
_CHECKPOINT = "turn_checkpoint"
_CHECKPOINT_DISCARD = "turn_checkpoint_discard"
_COMMIT = "turn_commit"
_COMPACTION_COMMIT = "compaction_commit"
_CONTENT_STATE = "content_replacement_state"
_PROMPT_CACHE_STATE = "prompt_cache_state"


class JsonlTrajectoryStore:  # code-health: ignore[AM009] -- complete store port
    """One append-only, fsync-backed journal per session.

    A turn commit is one JSONL record containing the immutable turn, committed
    message nodes, and compare-and-advance head mutation. The record is either
    visible in full or ignored as a torn tail.
    """

    def __init__(
        self,
        directory: str | Path,
        codec: CodecRegistry | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._directory / ".lock"
        self._process_lock = threading.RLock()
        self._codec = codec if codec is not None else CodecRegistry()

    @property
    def codec(self) -> CodecRegistry:
        return self._codec

    @property
    def indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_NODE_INDEXES

    @property
    def head_indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_HEAD_INDEXES

    def create_session(
        self,
        meta: SessionMeta,
        *,
        turns: Sequence[Turn] = (),
        nodes: Sequence[TrajectoryNode] = (),
        head: TrajectoryHead,
    ) -> None:
        copied_turns = tuple(turns)
        copied_nodes = tuple(nodes)
        with self._guard():
            state = InMemoryTrajectoryStore()
            state.create_session(
                meta,
                turns=copied_turns,
                nodes=copied_nodes,
                head=head,
            )
            record: dict[str, object] = {
                "version": _VERSION,
                "record_type": _SESSION,
                "meta": self._codec.serialize_session_meta(meta),
                "turns": [self._codec.serialize_turn(turn) for turn in copied_turns],
                "nodes": [serialize_node(node) for node in copied_nodes],
                "head": serialize_head(head),
            }
            self._create_file_unlocked(meta.id, record)

    def save_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        with self._guard():
            state = self._load_session_unlocked(session_id)
            state.save_checkpoint(session_id, checkpoint)
            self._normalize_tail_for_append_unlocked(session_id)
            self._append_record_unlocked(
                session_id,
                {
                    "version": _VERSION,
                    "record_type": _CHECKPOINT,
                    "checkpoint": self._codec.serialize_turn_checkpoint(checkpoint),
                },
            )

    def load_checkpoint(self, session_id: str) -> TurnCheckpoint | None:
        with self._guard():
            return self._load_session_unlocked(session_id).load_checkpoint(session_id)

    def discard_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        with self._guard():
            state = self._load_session_unlocked(session_id)
            current = state.load_checkpoint(session_id)
            state.discard_checkpoint(session_id, checkpoint)
            if current is None:
                return
            self._normalize_tail_for_append_unlocked(session_id)
            self._append_record_unlocked(
                session_id,
                {
                    "version": _VERSION,
                    "record_type": _CHECKPOINT_DISCARD,
                    "checkpoint": self._codec.serialize_turn_checkpoint(checkpoint),
                },
            )

    def commit_turn(self, session_id: str, commit: TrajectoryCommit) -> None:
        with self._guard():
            state = self._load_session_unlocked(session_id)
            state.commit_turn(session_id, commit)
            self._normalize_tail_for_append_unlocked(session_id)
            self._append_record_unlocked(
                session_id,
                {
                    "version": _VERSION,
                    "record_type": _COMMIT,
                    "turn": self._codec.serialize_turn(commit.turn),
                    "nodes": [serialize_node(node) for node in commit.nodes],
                    "advance_head": (
                        _serialize_head_advance(commit.advance_head)
                        if commit.advance_head is not None
                        else None
                    ),
                },
            )

    def commit_compaction(
        self,
        session_id: str,
        commit: TrajectoryCompactionCommit,
    ) -> None:
        with self._guard():
            state = self._load_session_unlocked(session_id)
            state.commit_compaction(session_id, commit)
            self._normalize_tail_for_append_unlocked(session_id)
            self._append_record_unlocked(
                session_id,
                {
                    "version": _VERSION,
                    "record_type": _COMPACTION_COMMIT,
                    "boundary": serialize_node(commit.boundary),
                    "advance_head": _serialize_head_advance(commit.advance_head),
                    "content_replacement_state": serialize_content_state(
                        commit.content_replacement_state
                    ),
                },
            )

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        with self._guard():
            return self._load_session_unlocked(session_id).load(session_id)

    def load_prefix(
        self,
        session_id: str,
        up_to: TurnRef,
    ) -> tuple[SessionMeta, list[Turn]]:
        with self._guard():
            return self._load_session_unlocked(session_id).load_prefix(
                session_id,
                up_to,
            )

    def session_children(self, session_id: str) -> list[str]:
        with self._guard():
            return self._load_all_unlocked().session_children(session_id)

    def session_exists(self, session_id: str) -> bool:
        with self._guard():
            return self._path(session_id).exists()

    def list_sessions(self) -> list[SessionMeta]:
        with self._guard():
            return self._load_all_unlocked().list_sessions()

    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        with self._guard():
            state = (
                self._load_session_unlocked(query.session_id)
                if query.session_id
                else self._load_all_unlocked()
            )
            return state.query_nodes(query)

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
            return self._load_session_unlocked(session_id).get_head(
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
            return self._load_session_unlocked(session_id).list_heads(
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
            state = (
                self._load_lineage_unlocked(session_id)
                if include_logical_parent
                else self._load_session_unlocked(session_id)
            )
            return state.load_chain(
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
            return self._load_session_unlocked(session_id).leaves(
                session_id,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
            )

    def save_content_replacement_state(
        self,
        session_id: str,
        state_value: ContentReplacementState,
    ) -> None:
        with self._guard():
            state = self._load_session_unlocked(session_id)
            state.save_content_replacement_state(session_id, state_value)
            self._normalize_tail_for_append_unlocked(session_id)
            self._append_record_unlocked(
                session_id,
                {
                    "version": _VERSION,
                    "record_type": _CONTENT_STATE,
                    "state": serialize_content_state(state_value),
                },
            )

    def load_content_replacement_state(
        self,
        session_id: str,
        state_key: str,
    ) -> ContentReplacementState | None:
        with self._guard():
            return self._load_session_unlocked(
                session_id
            ).load_content_replacement_state(
                session_id,
                state_key,
            )

    def clone_content_replacement_state(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        state_key: str,
        target_leaf_id: str | None = None,
    ) -> ContentReplacementState | None:
        with self._guard():
            source_state = self._load_session_unlocked(source_session_id)
            target_state = self._load_session_unlocked(target_session_id)
            source = source_state.load_content_replacement_state(
                source_session_id,
                state_key,
            )
            if source is None:
                return None
            cloned = replace(
                source,
                source_session_id=source_session_id,
                source_leaf_id=source.leaf_node_id or source.source_leaf_id,
                leaf_node_id=target_leaf_id,
            )
            target_state.save_content_replacement_state(target_session_id, cloned)
            self._normalize_tail_for_append_unlocked(target_session_id)
            self._append_record_unlocked(
                target_session_id,
                {
                    "version": _VERSION,
                    "record_type": _CONTENT_STATE,
                    "state": serialize_content_state(cloned),
                },
            )
            return cloned

    def save_prompt_cache_state(
        self,
        session_id: str,
        state_value: PromptCacheState,
    ) -> None:
        with self._guard():
            state = self._load_session_unlocked(session_id)
            state.save_prompt_cache_state(session_id, state_value)
            self._normalize_tail_for_append_unlocked(session_id)
            self._append_record_unlocked(
                session_id,
                {
                    "version": _VERSION,
                    "record_type": _PROMPT_CACHE_STATE,
                    "state": serialize_prompt_cache_state(state_value),
                },
            )

    def load_prompt_cache_state(
        self,
        session_id: str,
        cache_key: str,
    ) -> PromptCacheState | None:
        with self._guard():
            return self._load_session_unlocked(session_id).load_prompt_cache_state(
                session_id,
                cache_key,
            )

    def file_path(self, session_id: str) -> Path:
        return self._path(session_id)

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._process_lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _load_all_unlocked(self) -> InMemoryTrajectoryStore:
        state = InMemoryTrajectoryStore()
        for path in sorted(self._directory.glob("*.jsonl")):
            try:
                self._replay_file(path, state)
            except (ValueError, KeyError, TypeError):
                continue
        return state

    def _load_session_unlocked(
        self,
        session_id: str,
    ) -> InMemoryTrajectoryStore:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        state = InMemoryTrajectoryStore()
        self._replay_file(path, state)
        return state

    def _load_lineage_unlocked(
        self,
        session_id: str,
    ) -> InMemoryTrajectoryStore:
        state = InMemoryTrajectoryStore()
        visiting: set[str] = set()
        loaded: set[str] = set()

        def load(current_session_id: str) -> None:
            if current_session_id in loaded:
                return
            if current_session_id in visiting:
                raise ValueError(
                    f"trajectory session parent cycle includes {current_session_id}"
                )
            path = self._path(current_session_id)
            if not path.exists():
                raise KeyError(current_session_id)
            visiting.add(current_session_id)
            probe = InMemoryTrajectoryStore()
            self._replay_file(path, probe)
            meta, _turns = probe.load(current_session_id)
            if meta.parent_id is not None:
                load(meta.parent_id)
            self._replay_file(path, state)
            visiting.remove(current_session_id)
            loaded.add(current_session_id)

        load(session_id)
        return state

    def _replay_file(
        self,
        path: Path,
        state: InMemoryTrajectoryStore,
    ) -> None:
        records = _read_records(path)
        if not records:
            raise ValueError(f"corrupt empty trajectory journal: {path}")
        session_record = records[0]
        if session_record.get("record_type") != _SESSION:
            raise ValueError(f"trajectory journal does not start with session: {path}")
        meta = self._codec.deserialize_session_meta(  # type: ignore[arg-type]
            dict(_required_mapping(session_record, "meta"))
        )
        if not isinstance(meta, SessionMeta):
            raise TypeError("trajectory session metadata codec returned invalid data")
        turns = [
            self._codec.deserialize_turn(dict(item))  # type: ignore[arg-type]
            for item in _required_mapping_list(session_record, "turns")
        ]
        nodes = [
            deserialize_node(item)
            for item in _required_mapping_list(session_record, "nodes")
        ]
        head = deserialize_head(_required_mapping(session_record, "head"))
        state.create_session(
            meta,
            turns=turns,
            nodes=nodes,
            head=head,
        )
        for record in records[1:]:
            record_type = record.get("record_type")
            if record_type == _CHECKPOINT:
                state.save_checkpoint(
                    meta.id,
                    self._codec.deserialize_turn_checkpoint(  # type: ignore[arg-type]
                        dict(_required_mapping(record, "checkpoint"))
                    ),
                )
            elif record_type == _CHECKPOINT_DISCARD:
                checkpoint = self._codec.deserialize_turn_checkpoint(  # type: ignore[arg-type]
                    dict(_required_mapping(record, "checkpoint"))
                )
                state.discard_checkpoint(meta.id, checkpoint)
            elif record_type == _COMMIT:
                raw_advance = record.get("advance_head")
                state.commit_turn(
                    meta.id,
                    TrajectoryCommit(
                        turn=self._codec.deserialize_turn(  # type: ignore[arg-type]
                            dict(_required_mapping(record, "turn"))
                        ),
                        nodes=tuple(
                            deserialize_node(item)
                            for item in _required_mapping_list(record, "nodes")
                        ),
                        advance_head=(
                            _deserialize_head_advance(raw_advance)
                            if isinstance(raw_advance, Mapping)
                            else None
                        ),
                    ),
                )
            elif record_type == _COMPACTION_COMMIT:
                state.commit_compaction(
                    meta.id,
                    TrajectoryCompactionCommit(
                        boundary=deserialize_node(
                            _required_mapping(record, "boundary")
                        ),
                        advance_head=_deserialize_head_advance(
                            _required_mapping(record, "advance_head")
                        ),
                        content_replacement_state=deserialize_content_state(
                            _required_mapping(
                                record,
                                "content_replacement_state",
                            )
                        ),
                    ),
                )
            elif record_type == _CONTENT_STATE:
                state.save_content_replacement_state(
                    meta.id,
                    deserialize_content_state(_required_mapping(record, "state")),
                )
            elif record_type == _PROMPT_CACHE_STATE:
                state.save_prompt_cache_state(
                    meta.id,
                    deserialize_prompt_cache_state(_required_mapping(record, "state")),
                )
            else:
                raise ValueError(
                    f"unsupported trajectory record type {record_type!r} in {path}"
                )

    def _path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self._directory / f"{session_id}.jsonl"

    def _create_file_unlocked(
        self,
        session_id: str,
        record: Mapping[str, object],
    ) -> None:
        path = self._path(session_id)
        payload = _encode_record(record)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{session_id}.",
            dir=self._directory,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                raise ValueError(f"session already exists: {session_id}") from None
            _fsync_directory(self._directory)
        finally:
            temp_path.unlink(missing_ok=True)

    def _append_record_unlocked(
        self,
        session_id: str,
        record: Mapping[str, object],
    ) -> None:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        with path.open("ab") as handle:
            handle.write(_encode_record(record))
            handle.flush()
            os.fsync(handle.fileno())

    def _normalize_tail_for_append_unlocked(self, session_id: str) -> None:
        path = self._path(session_id)
        raw = path.read_bytes()
        if not raw or raw.endswith((b"\n", b"\r")):
            return
        last_newline = raw.rfind(b"\n")
        tail = raw[last_newline + 1 :]
        try:
            value: object = json.loads(tail)
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        if isinstance(value, Mapping) and value.get("version") == _VERSION:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return
        with path.open("r+b") as handle:
            handle.truncate(last_newline + 1)
            handle.flush()
            os.fsync(handle.fileno())


def _serialize_head_advance(
    advance: TrajectoryHeadAdvance,
) -> dict[str, object]:
    return {
        "head": serialize_head(advance.to_head()),
        "previous_node_id": advance.previous_node_id,
    }


def _deserialize_head_advance(
    value: Mapping[str, object],
) -> TrajectoryHeadAdvance:
    head = deserialize_head(_required_mapping(value, "head"))
    previous = value.get("previous_node_id")
    if previous is not None and not isinstance(previous, str):
        raise ValueError("trajectory head previous_node_id must be a string or null")
    if head.node_id is None or head.seq is None:
        raise ValueError("trajectory head advance must identify a node and sequence")
    return TrajectoryHeadAdvance(
        session_id=head.session_id,
        node_id=head.node_id,
        seq=head.seq,
        previous_node_id=previous,
        head_id=head.head_id,
        branch_id=head.branch_id,
        root_session_id=head.root_session_id,
        parent_session_id=head.parent_session_id,
        logical_parent_id=head.logical_parent_id,
        agent_id=head.agent_id,
        is_sidechain=head.is_sidechain,
        status=head.status,
        updated_at=head.updated_at,
        metadata=head.metadata,
    )


def _read_records(path: Path) -> list[Mapping[str, object]]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        terminated = line.endswith((b"\n", b"\r"))
        try:
            value: object = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if line_number == len(lines) and not terminated:
                break
            raise ValueError(
                f"corrupt trajectory record {line_number} in {path}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError(
                f"trajectory record {line_number} in {path} is not an object"
            )
        record = {str(key): item for key, item in value.items()}
        if record.get("version") != _VERSION:
            raise ValueError(
                f"trajectory record {line_number} in {path} has unsupported version"
            )
        records.append(record)
    return records


def _required_mapping(
    value: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"trajectory record field {key!r} must be an object")
    return {str(item_key): item_value for item_key, item_value in item.items()}


def _required_mapping_list(
    value: Mapping[str, object],
    key: str,
) -> list[Mapping[str, object]]:
    item = value.get(key)
    if not isinstance(item, list) or not all(
        isinstance(element, Mapping) for element in item
    ):
        raise ValueError(f"trajectory record field {key!r} must be an object list")
    return [
        {str(item_key): item_value for item_key, item_value in element.items()}
        for element in item
        if isinstance(element, Mapping)
    ]


def _encode_record(record: Mapping[str, object]) -> bytes:
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_session_id(session_id: str) -> None:
    if (
        not session_id
        or session_id in {".", ".."}
        or Path(session_id).name != session_id
        or "\\" in session_id
        or "\x00" in session_id
    ):
        raise ValueError(f"session_id is not a valid path token: {session_id!r}")


__all__ = ["JsonlTrajectoryStore"]
