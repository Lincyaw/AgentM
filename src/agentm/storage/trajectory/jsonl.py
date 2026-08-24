# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""Crash-aware JSONL implementation of the unified trajectory store."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.store import (
    SessionMeta,
    TrajectoryCommit,
    TrajectoryCompactionCommit,
    TrajectoryDiagnostic,
    TrajectoryNodeQuery,
)
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_HEAD_ID,
    TRAJECTORY_HEAD_INDEXES,
    TRAJECTORY_NODE_INDEXES,
    ContentReplacementState,
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
from agentm.storage.serialization import (
    deserialize_content_state,
    deserialize_diagnostic,
    deserialize_head,
    deserialize_node,
    serialize_content_state,
    serialize_diagnostic,
    serialize_head,
    serialize_node,
)
from agentm.storage.trajectory.memory import InMemoryTrajectoryStore

_VERSION = 2
_SESSION = "session"
_CHECKPOINT = "turn_checkpoint"
_CHECKPOINT_DISCARD = "turn_checkpoint_discard"
_COMMIT = "turn_commit"
_COMPACTION_COMMIT = "compaction_commit"
_CONTENT_STATE = "content_replacement_state"
_PROMPT_CACHE_STATE = "prompt_cache_state"
_DIAGNOSTIC = "diagnostic"

# Checkpoint records are superseded rather than updated: the journal only
# appends, while the state they rebuild keeps one checkpoint slot per session.
# Every superseded record is therefore dead weight that still has to be read on
# every replay. Reclaiming it follows Postgres autovacuum: track dead bytes as
# they are produced, and rewrite the file once they exceed a fixed floor and a
# fraction of what is actually live. The floor keeps short sessions from ever
# rewriting; the fraction keeps the amortized cost of rewriting proportional to
# the writing that produced the garbage.
_VACUUM_MIN_DEAD_BYTES = 1 << 20
_VACUUM_SCALE_FACTOR = 0.5

# Resuming a replay trusts that the bytes before the resume point never change,
# which is true of an append-only journal but not of a file something else
# rewrote. Fingerprinting the window just before the resume point catches a
# rewrite for the cost of one small read. It is a staleness check, not an
# integrity check: a cold open still validates every record in the file, and
# that is what the durability guarantees rest on.
_RESUME_FINGERPRINT_WINDOW = 4096


@dataclass(slots=True)
class _SessionCache:
    """Replayed state for one session plus the journal position it reflects.

    ``offset`` is always a record boundary, so a later replay can resume from
    it instead of rebuilding the whole session. ``dev``/``ino`` detect a file
    another process replaced (a vacuum), and a file shorter than ``offset``
    detects a truncated torn tail; either one forces a full rebuild.
    """

    state: InMemoryTrajectoryStore
    offset: int
    dev: int
    ino: int
    fingerprint: bytes = b""
    live_bytes: int = 0
    dead_bytes: int = 0
    pending_checkpoint_bytes: int = 0

    def account(self, record_type: object, size: int) -> None:
        """Attribute one record's bytes to the live or dead set."""

        if record_type == _CHECKPOINT:
            # The previous checkpoint is now unreachable; this one is live
            # until something supersedes or commits it.
            self.dead_bytes += self.pending_checkpoint_bytes
            self.pending_checkpoint_bytes = size
        elif record_type == _CHECKPOINT_DISCARD:
            # Both the discard and what it discards stop affecting replay.
            self.dead_bytes += self.pending_checkpoint_bytes + size
            self.pending_checkpoint_bytes = 0
        elif record_type == _COMMIT:
            self.dead_bytes += self.pending_checkpoint_bytes
            self.pending_checkpoint_bytes = 0
            self.live_bytes += size
        else:
            self.live_bytes += size


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
        self._cache: dict[str, _SessionCache] = {}

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
        with self._write_guard(meta.id):
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
        with self._write_guard(session_id):
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
        with self._read_guard(session_id):
            return self._load_session_unlocked(session_id).load_checkpoint(session_id)

    def discard_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        with self._write_guard(session_id):
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
        with self._write_guard(session_id):
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
        with self._write_guard(session_id):
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
        with self._read_guard(session_id):
            return self._load_session_unlocked(session_id).load(session_id)

    def load_prefix(
        self,
        session_id: str,
        up_to: TurnRef,
    ) -> tuple[SessionMeta, list[Turn]]:
        with self._read_guard(session_id):
            return self._load_session_unlocked(session_id).load_prefix(
                session_id,
                up_to,
            )

    def session_children(self, session_id: str) -> list[str]:
        with self._read_guard(None):
            return self._load_all_unlocked().session_children(session_id)

    def session_exists(self, session_id: str) -> bool:
        # Deliberately unlocked: one stat is already atomic, and taking the
        # session lock would create a lock file for every id ever probed --
        # including the ones that do not exist, which is most of them.
        return self._path(session_id).exists()

    def list_sessions(self) -> list[SessionMeta]:
        with self._read_guard(None):
            return self._load_all_unlocked().list_sessions()

    def append_diagnostic(self, diagnostic: TrajectoryDiagnostic) -> None:
        with self._write_guard(diagnostic.session_id):
            state = self._load_session_unlocked(diagnostic.session_id)
            state.append_diagnostic(diagnostic)
            self._normalize_tail_for_append_unlocked(diagnostic.session_id)
            self._append_record_unlocked(
                diagnostic.session_id,
                {
                    "version": _VERSION,
                    "record_type": _DIAGNOSTIC,
                    "diagnostic": serialize_diagnostic(diagnostic),
                },
            )

    def list_diagnostics(self, session_id: str) -> list[TrajectoryDiagnostic]:
        with self._read_guard(session_id):
            return self._load_session_unlocked(session_id).list_diagnostics(session_id)

    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        with self._read_guard(query.session_id or None):
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
        with self._read_guard(session_id):
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
        with self._read_guard(session_id):
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
        with self._read_guard(None if include_logical_parent else session_id):
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
        with self._read_guard(session_id):
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
        with self._write_guard(session_id):
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
        with self._read_guard(session_id):
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
        with self._write_guard(source_session_id, target_session_id):
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

    def file_path(self, session_id: str) -> Path:
        return self._path(session_id)

    @contextmanager
    def _read_guard(self, session_id: str | None) -> Iterator[None]:
        """Hold one session against writers, or scan across sessions."""

        with self._locked((session_id,) if session_id else (), exclusive=False):
            yield

    @contextmanager
    def _write_guard(self, *session_ids: str) -> Iterator[None]:
        """Hold every named session exclusively for the duration."""

        with self._locked(session_ids, exclusive=True):
            yield

    @contextmanager
    def _locked(self, session_ids: Sequence[str], *, exclusive: bool) -> Iterator[None]:
        # The directory lock is only ever held shared. It exists so that a
        # process running an older build -- which took it exclusively for every
        # operation -- still excludes this one during a rolling upgrade; it is
        # not what makes concurrent sessions safe here. Session locks are, and
        # they are taken in sorted order so an operation spanning two sessions
        # cannot deadlock against one spanning them the other way.
        with self._process_lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            with ExitStack() as held:
                held.enter_context(self._flock(self._lock_path, fcntl.LOCK_SH))
                try:
                    for session_id in sorted(set(session_ids)):
                        held.enter_context(
                            self._flock(self._session_lock_path(session_id), mode)
                        )
                    yield
                except BaseException:
                    # A failed operation may have applied part of a mutation to
                    # the cached state, or advanced the replay offset past a
                    # record it could not apply. Neither is recoverable from
                    # here, and both are cheap to rebuild from the journal.
                    for session_id in session_ids:
                        self._cache.pop(session_id, None)
                    raise

    @contextmanager
    def _flock(self, path: Path, mode: int) -> Iterator[None]:
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), mode)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _session_lock_path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        # Not the journal itself: a vacuum replaces that file, and a lock held
        # on the replaced inode would stop excluding anyone.
        return self._directory / f"{session_id}.lock"

    def _load_all_unlocked(self) -> InMemoryTrajectoryStore:
        state = InMemoryTrajectoryStore()
        for path in sorted(self._directory.glob("*.jsonl")):
            try:
                self._replay_file(path, state)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("skipping corrupt session file {}: {}", path.name, exc)
                continue
        return state

    def _load_session_unlocked(
        self,
        session_id: str,
    ) -> InMemoryTrajectoryStore:
        path = self._path(session_id)
        try:
            status = path.stat()
        except FileNotFoundError:
            self._cache.pop(session_id, None)
            raise KeyError(session_id) from None

        cache = self._cache.get(session_id)
        if (
            cache is not None
            and cache.dev == status.st_dev
            and cache.ino == status.st_ino
            and status.st_size >= cache.offset
            and cache.fingerprint == _fingerprint(path, cache.offset)
        ):
            if status.st_size > cache.offset:
                self._extend_cache_unlocked(session_id, path, cache)
            return cache.state
        return self._rebuild_cache_unlocked(session_id, path, status).state

    def _rebuild_cache_unlocked(
        self,
        session_id: str,
        path: Path,
        status: os.stat_result,
    ) -> _SessionCache:
        self._cache.pop(session_id, None)
        state = InMemoryTrajectoryStore()
        cache = _SessionCache(
            state=state,
            offset=0,
            dev=status.st_dev,
            ino=status.st_ino,
        )
        records, end_offset, whole = _read_records_from(path, 0)
        if not records:
            raise ValueError(f"corrupt empty trajectory journal: {path}")
        session_record, session_size = records[0]
        meta = self._apply_session_record(state, session_record, path)
        cache.account(_SESSION, session_size)
        for record, size in records[1:]:
            self._apply_record(state, meta.id, record, path)
            cache.account(record.get("record_type"), size)
        cache.offset = end_offset
        cache.fingerprint = _fingerprint(path, end_offset)
        if whole:
            self._cache[session_id] = cache
        return cache

    def _extend_cache_unlocked(
        self,
        session_id: str,
        path: Path,
        cache: _SessionCache,
    ) -> None:
        records, end_offset, whole = _read_records_from(path, cache.offset)
        for record, size in records:
            self._apply_record(cache.state, session_id, record, path)
            cache.account(record.get("record_type"), size)
        cache.offset = end_offset
        cache.fingerprint = _fingerprint(path, end_offset)
        if not whole:
            # A torn tail was applied but is not at a boundary we can resume
            # from. Drop the cache rather than remember a position that a
            # later append will invalidate.
            self._cache.pop(session_id, None)

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
        records, _end_offset, _whole = _read_records_from(path, 0)
        if not records:
            raise ValueError(f"corrupt empty trajectory journal: {path}")
        session_record, _size = records[0]
        meta = self._apply_session_record(state, session_record, path)
        for record, _record_size in records[1:]:
            self._apply_record(state, meta.id, record, path)

    def _apply_session_record(
        self,
        state: InMemoryTrajectoryStore,
        record: Mapping[str, object],
        path: Path,
    ) -> SessionMeta:
        if record.get("record_type") != _SESSION:
            raise ValueError(f"trajectory journal does not start with session: {path}")
        meta = self._codec.deserialize_session_meta(  # type: ignore[arg-type]
            dict(_required_mapping(record, "meta"))
        )
        if not isinstance(meta, SessionMeta):
            raise TypeError("trajectory session metadata codec returned invalid data")
        turns = [
            self._codec.deserialize_turn(dict(item))  # type: ignore[arg-type]
            for item in _required_mapping_list(record, "turns")
        ]
        nodes = [
            deserialize_node(item) for item in _required_mapping_list(record, "nodes")
        ]
        head = deserialize_head(_required_mapping(record, "head"))
        state.create_session(meta, turns=turns, nodes=nodes, head=head)
        return meta

    def _apply_record(
        self,
        state: InMemoryTrajectoryStore,
        session_id: str,
        record: Mapping[str, object],
        path: Path,
    ) -> None:
        record_type = record.get("record_type")
        if record_type == _CHECKPOINT:
            state.save_checkpoint(
                session_id,
                self._codec.deserialize_turn_checkpoint(  # type: ignore[arg-type]
                    dict(_required_mapping(record, "checkpoint"))
                ),
            )
        elif record_type == _CHECKPOINT_DISCARD:
            checkpoint = self._codec.deserialize_turn_checkpoint(  # type: ignore[arg-type]
                dict(_required_mapping(record, "checkpoint"))
            )
            state.discard_checkpoint(session_id, checkpoint)
        elif record_type == _COMMIT:
            raw_advance = record.get("advance_head")
            state.commit_turn(
                session_id,
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
                session_id,
                TrajectoryCompactionCommit(
                    boundary=deserialize_node(_required_mapping(record, "boundary")),
                    advance_head=_deserialize_head_advance(
                        _required_mapping(record, "advance_head")
                    ),
                    content_replacement_state=deserialize_content_state(
                        _required_mapping(record, "content_replacement_state")
                    ),
                ),
            )
        elif record_type == _CONTENT_STATE:
            state.save_content_replacement_state(
                session_id,
                deserialize_content_state(_required_mapping(record, "state")),
            )
        elif record_type == _PROMPT_CACHE_STATE:
            # Legacy prompt-cache records are no longer applied; skip them.
            pass
        elif record_type == _DIAGNOSTIC:
            state.append_diagnostic(
                deserialize_diagnostic(_required_mapping(record, "diagnostic"))
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
        payload = _encode_record(record)
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        cache = self._cache.get(session_id)
        if cache is None:
            return
        # The caller already applied this record to the cached state, so the
        # cache is current as of the byte we just wrote.
        cache.offset += len(payload)
        cache.fingerprint = _fingerprint(path, cache.offset)
        cache.account(record.get("record_type"), len(payload))
        self._maybe_vacuum_unlocked(session_id, cache)

    def _maybe_vacuum_unlocked(self, session_id: str, cache: _SessionCache) -> None:
        threshold = max(
            _VACUUM_MIN_DEAD_BYTES,
            int(_VACUUM_SCALE_FACTOR * cache.live_bytes),
        )
        if cache.dead_bytes < threshold:
            return
        self._vacuum_unlocked(session_id, cache)

    def _vacuum_unlocked(self, session_id: str, cache: _SessionCache) -> None:
        """Rewrite one journal without the records replay would discard.

        Only superseded checkpoints and the discards that retire them are
        dropped, and only while this session is held exclusively. The result is
        byte-for-byte what a replay of the original would have produced.
        """

        path = self._path(session_id)
        before = cache.offset
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{session_id}.vacuum.",
            dir=self._directory,
        )
        temp_path = Path(temp_name)
        kept = _SessionCache(
            state=cache.state,
            offset=0,
            dev=cache.dev,
            ino=cache.ino,
        )
        try:
            with os.fdopen(fd, "wb") as out, path.open("rb") as source:
                # ``held`` is the newest checkpoint not yet known to be dead.
                # It is dropped when a later checkpoint supersedes it, when a
                # discard retires it, or when a commit ends its turn. Anything
                # else forces it out first, because replay applies records in
                # file order and reordering one past an unrelated record is not
                # a property this can assume.
                held: bytes | None = None
                for line in source:
                    if not line.endswith((b"\n", b"\r")):
                        logger.debug(
                            "vacuum dropping torn tail in session {}: {} bytes",
                            session_id,
                            len(line),
                        )
                        break
                    record_type = _peek_record_type(line)
                    if record_type == _CHECKPOINT:
                        held = line
                        continue
                    if record_type == _CHECKPOINT_DISCARD and held is not None:
                        # Nothing observed either record, so both go.
                        held = None
                        continue
                    if record_type == _COMMIT:
                        held = None
                    elif held is not None:
                        out.write(held)
                        kept.account(_CHECKPOINT, len(held))
                        held = None
                    out.write(line)
                    kept.account(record_type, len(line))
                if held is not None:
                    # The one checkpoint replay would still honor.
                    out.write(held)
                    kept.account(_CHECKPOINT, len(held))
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp_path, path)
            _fsync_directory(self._directory)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            self._cache.pop(session_id, None)
            raise

        status = path.stat()
        cache.dev = status.st_dev
        cache.ino = status.st_ino
        cache.offset = status.st_size
        cache.fingerprint = _fingerprint(path, status.st_size)
        cache.live_bytes = kept.live_bytes
        cache.dead_bytes = 0
        cache.pending_checkpoint_bytes = kept.pending_checkpoint_bytes
        logger.debug(
            "vacuumed trajectory session {}: {} -> {} bytes",
            session_id,
            before,
            status.st_size,
        )

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
            logger.warning(
                "truncating unparseable tail in session {}: {} bytes discarded",
                session_id,
                len(tail),
            )
            value = None
        if isinstance(value, Mapping) and value.get("version") == _VERSION:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return
        logger.debug(
            "truncating torn trailing record in session {}: {} bytes",
            session_id,
            len(tail),
        )
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


def _fingerprint(path: Path, offset: int) -> bytes:
    """Digest the window of bytes ending at ``offset``."""

    window = min(offset, _RESUME_FINGERPRINT_WINDOW)
    with path.open("rb") as handle:
        handle.seek(offset - window)
        raw = handle.read(window)
    return hashlib.sha256(offset.to_bytes(8, "big") + raw).digest()


def _peek_record_type(line: bytes) -> object:
    value: object = json.loads(line)
    if not isinstance(value, Mapping):
        raise ValueError("trajectory record is not an object")
    return value.get("record_type")


def _read_records_from(
    path: Path,
    offset: int,
) -> tuple[list[tuple[Mapping[str, object], int]], int, bool]:
    """Read records starting at ``offset``, with each record's byte length.

    Returns the records, the offset just past the last *terminated* record,
    and whether the read ended on a record boundary. A valid-but-unterminated
    trailing record is still returned -- it is real data a crash left behind --
    but it does not advance the resumable offset, because the next append will
    terminate it in place and change the bytes at that position.
    """

    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read()
    lines = raw.splitlines(keepends=True)
    records: list[tuple[Mapping[str, object], int]] = []
    end_offset = offset
    whole = True
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
        records.append((record, len(line)))
        if terminated:
            end_offset += len(line)
        else:
            whole = False
    return records, end_offset, whole


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
