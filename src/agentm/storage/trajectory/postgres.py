# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""Postgres implementation of the unified trajectory store."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace

from sqlalchemy.engine import Connection, CursorResult, Engine

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
from agentm.core.lib.trajectory_store import (
    turn_prefix_cut,
    validate_checkpoint_commit,
    validate_checkpoint_discard,
    validate_compaction_commit,
    validate_initial_node_state,
    validate_node_append_state,
    validate_turn_append,
    validate_turn_checkpoint,
    validate_turn_sequence,
)
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

PostgresParams = Sequence[object] | Mapping[str, object]


class PostgresCursor:
    """Driver-SQL cursor facade over one SQLAlchemy connection."""

    __slots__ = ("_connection", "_result")

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._result: CursorResult[object] | None = None

    def execute(self, query: str, params: PostgresParams | None = None) -> object:
        if params is None:
            self._result = self._connection.exec_driver_sql(query)
        else:
            self._result = self._connection.exec_driver_sql(query, params)
        return self._result

    def fetchone(self) -> Sequence[object] | None:
        if self._result is None:
            raise RuntimeError("Postgres cursor has no result")
        return self._result.fetchone()

    def fetchall(self) -> Sequence[Sequence[object]]:
        if self._result is None:
            raise RuntimeError("Postgres cursor has no result")
        return self._result.fetchall()


class PostgresTrajectoryStore:  # code-health: ignore[AM009] -- complete store port
    """Postgres-backed trajectory and message-index store.

    The tables use denormalized columns for the portable index contract and a
    JSONB payload for lossless ABI reconstruction. This adapter assumes a
    synchronous SQLAlchemy engine and keeps methods blocking, matching the SDK
    store protocol.
    """

    def __init__(
        self,
        engine: Engine | Connection,
        *,
        schema: str = "public",
        codec: CodecRegistry | None = None,
        create_schema: bool = True,
    ) -> None:
        if not isinstance(engine, (Engine, Connection)):
            raise TypeError(
                "PostgresTrajectoryStore requires a SQLAlchemy Engine or Connection"
            )
        self._handle: Engine | Connection = engine
        self._schema = _validate_identifier(schema, label="Postgres schema")
        self._codec = codec if codec is not None else CodecRegistry()
        self._lock = threading.RLock()
        if create_schema:
            self.create_schema()

    @property
    def codec(self) -> CodecRegistry:
        return self._codec

    @property
    def indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_NODE_INDEXES

    @property
    def head_indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_HEAD_INDEXES

    def create_schema(self) -> None:
        with self._transaction() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_sessions")} (
                    id text PRIMARY KEY,
                    parent_id text,
                    purpose text NOT NULL DEFAULT 'root',
                    cwd text NOT NULL DEFAULT '',
                    created_at double precision NOT NULL DEFAULT 0,
                    meta_json jsonb NOT NULL
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_turns")} (
                    session_id text NOT NULL
                        REFERENCES {self._table("trajectory_sessions")}(id),
                    turn_index integer NOT NULL,
                    turn_id text NOT NULL,
                    turn_json jsonb NOT NULL,
                    PRIMARY KEY (session_id, turn_index),
                    UNIQUE (session_id, turn_id)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_checkpoints")} (
                    session_id text PRIMARY KEY
                        REFERENCES {self._table("trajectory_sessions")}(id),
                    turn_index integer NOT NULL,
                    turn_id text NOT NULL,
                    updated_at double precision NOT NULL,
                    checkpoint_json jsonb NOT NULL,
                    UNIQUE (session_id, turn_id)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_nodes")} (
                    id text PRIMARY KEY,
                    session_id text NOT NULL
                        CONSTRAINT agentm_trajectory_nodes_session_fk
                        REFERENCES {self._table("trajectory_sessions")}(id),
                    root_session_id text,
                    parent_session_id text,
                    seq bigint NOT NULL,
                    parent_id text,
                    logical_parent_id text,
                    branch_id text NOT NULL,
                    head_id text NOT NULL,
                    agent_id text,
                    is_sidechain boolean NOT NULL DEFAULT false,
                    turn_id text,
                    turn_index bigint,
                    run_id text,
                    run_step bigint,
                    message_index bigint,
                    kind text NOT NULL,
                    role text NOT NULL,
                    visibility text NOT NULL,
                    tool_call_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
                    tool_names jsonb NOT NULL DEFAULT '[]'::jsonb,
                    cache_key text,
                    content_ref text,
                    timestamp double precision NOT NULL DEFAULT 0,
                    node_json jsonb NOT NULL,
                    UNIQUE (session_id, seq),
                    UNIQUE (session_id, id)
                )
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {self._table("trajectory_nodes")}
                ADD COLUMN IF NOT EXISTS run_id text
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {self._table("trajectory_nodes")}
                ADD COLUMN IF NOT EXISTS run_step bigint
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_heads")} (
                    session_id text NOT NULL
                        CONSTRAINT agentm_trajectory_heads_session_fk
                        REFERENCES {self._table("trajectory_sessions")}(id),
                    head_id text NOT NULL,
                    root_session_id text,
                    parent_session_id text,
                    branch_id text NOT NULL,
                    agent_id text,
                    is_sidechain boolean NOT NULL DEFAULT false,
                    node_id text,
                    seq bigint,
                    logical_parent_id text,
                    status text NOT NULL,
                    updated_at double precision NOT NULL DEFAULT 0,
                    head_json jsonb NOT NULL,
                    PRIMARY KEY (session_id, head_id)
                )
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {self._table("trajectory_heads")}
                ADD COLUMN IF NOT EXISTS logical_parent_id text
                """
            )
            cur.execute(
                f"""
                UPDATE {self._table("trajectory_heads")}
                SET logical_parent_id = NULLIF(
                    head_json ->> 'logical_parent_id',
                    ''
                )
                WHERE logical_parent_id IS NULL
                  AND head_json ->> 'logical_parent_id' IS NOT NULL
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_content_states")} (
                    session_id text NOT NULL
                        CONSTRAINT agentm_trajectory_content_states_session_fk
                        REFERENCES {self._table("trajectory_sessions")}(id),
                    state_key text NOT NULL,
                    state_json jsonb NOT NULL,
                    PRIMARY KEY (session_id, state_key)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_prompt_cache_states")} (
                    session_id text NOT NULL
                        CONSTRAINT agentm_trajectory_prompt_cache_states_session_fk
                        REFERENCES {self._table("trajectory_sessions")}(id),
                    cache_key text NOT NULL,
                    state_json jsonb NOT NULL,
                    PRIMARY KEY (session_id, cache_key)
                )
                """
            )
            for table in (
                "trajectory_nodes",
                "trajectory_heads",
                "trajectory_content_states",
                "trajectory_prompt_cache_states",
            ):
                self._ensure_session_foreign_key(cur, table)
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._index("turns_session_idx")}
                ON {self._table("trajectory_turns")} (session_id, turn_index)
                """
            )
            cur.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS
                    {self._index("turns_session_turn_id_idx")}
                ON {self._table("trajectory_turns")} (session_id, turn_id)
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._index("sessions_parent_idx")}
                ON {self._table("trajectory_sessions")} (parent_id)
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._index("sessions_created_idx")}
                ON {self._table("trajectory_sessions")} (created_at)
                """
            )
            for statement in _index_statements(self._schema):
                cur.execute(statement)

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
        validate_turn_sequence(copied_turns)
        validate_initial_node_state(meta.id, copied_nodes, head)
        turn_ids = {turn.id for turn in copied_turns}
        if any(
            node.session_id != meta.id or node.turn_id not in turn_ids
            for node in copied_nodes
        ):
            raise ValueError(
                "initial trajectory nodes must belong to the session's "
                "initial committed turns"
            )
        meta_json = _json_dumps(self._codec.serialize_session_meta(meta))
        with self._transaction() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_sessions")}
                    (id, parent_id, purpose, cwd, created_at, meta_json)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                (
                    meta.id,
                    meta.parent_id,
                    meta.purpose,
                    meta.cwd,
                    meta.created_at,
                    meta_json,
                ),
            )
            if cur.fetchone() is None:
                raise ValueError(f"session already exists: {meta.id}")
            for turn in copied_turns:
                self._insert_turn(cur, meta.id, turn)
            for node in copied_nodes:
                self._insert_node(cur, node)
            self._upsert_head(cur, head)

    def save_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        checkpoint_json = _json_dumps(self._codec.serialize_turn_checkpoint(checkpoint))
        with self._transaction() as cur:
            committed = self._lock_session_turns(cur, session_id)
            existing = self._select_checkpoint(cur, session_id)
            validate_turn_checkpoint(
                committed,
                checkpoint,
                existing=existing,
            )
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_checkpoints")}
                    (
                        session_id,
                        turn_index,
                        turn_id,
                        updated_at,
                        checkpoint_json
                    )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (session_id)
                DO UPDATE SET
                    turn_index = EXCLUDED.turn_index,
                    turn_id = EXCLUDED.turn_id,
                    updated_at = EXCLUDED.updated_at,
                    checkpoint_json = EXCLUDED.checkpoint_json
                """,
                (
                    session_id,
                    checkpoint.index,
                    checkpoint.id,
                    checkpoint.updated_at,
                    checkpoint_json,
                ),
            )

    def load_checkpoint(self, session_id: str) -> TurnCheckpoint | None:
        with self._transaction() as cur:
            cur.execute(
                f"""
                SELECT c.checkpoint_json
                FROM {self._table("trajectory_sessions")} AS s
                LEFT JOIN {self._table("trajectory_checkpoints")} AS c
                    ON c.session_id = s.id
                WHERE s.id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(session_id)
            if row[0] is None:
                return None
            return self._codec.deserialize_turn_checkpoint(  # type: ignore[arg-type]
                dict(_json_mapping(row[0]))
            )

    def discard_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        with self._transaction() as cur:
            self._lock_session_turns(cur, session_id)
            current = self._select_checkpoint(cur, session_id)
            validate_checkpoint_discard(current, checkpoint)
            if current is None:
                return
            cur.execute(
                f"""
                DELETE FROM {self._table("trajectory_checkpoints")}
                WHERE session_id = %s AND turn_id = %s
                """,
                (session_id, checkpoint.id),
            )

    def commit_turn(self, session_id: str, commit: TrajectoryCommit) -> None:
        if any(node.session_id != session_id for node in commit.nodes):
            raise ValueError("trajectory commit nodes must belong to the session")
        with self._transaction() as cur:
            committed = self._lock_session_turns(cur, session_id)
            validate_turn_append(committed, commit.turn)
            validate_checkpoint_commit(
                self._select_checkpoint(cur, session_id),
                commit.turn,
            )
            if commit.advance_head is not None:
                current_head = self._validate_head_advance(
                    cur,
                    session_id,
                    commit.advance_head,
                    commit.nodes,
                )
                validate_node_append_state(
                    session_id,
                    commit.nodes,
                    commit.advance_head,
                    current_head=current_head,
                    expected_seq=self._next_node_seq(cur, session_id),
                )
            self._insert_turn(cur, session_id, commit.turn)
            for node in commit.nodes:
                self._insert_node(cur, node)
            if commit.advance_head is not None:
                self._upsert_head(cur, commit.advance_head.to_head())
            cur.execute(
                f"""
                DELETE FROM {self._table("trajectory_checkpoints")}
                WHERE session_id = %s
                """,
                (session_id,),
            )

    def commit_compaction(
        self,
        session_id: str,
        commit: TrajectoryCompactionCommit,
    ) -> None:
        if commit.boundary.session_id != session_id:
            raise ValueError("trajectory compact boundary must belong to the session")
        with self._transaction() as cur:
            committed = self._lock_session_turns(cur, session_id)
            validate_compaction_commit(committed, commit)
            current_head = self._validate_head_advance(
                cur,
                session_id,
                commit.advance_head,
                (commit.boundary,),
            )
            validate_node_append_state(
                session_id,
                (commit.boundary,),
                commit.advance_head,
                current_head=current_head,
                expected_seq=self._next_node_seq(cur, session_id),
            )
            self._insert_node(cur, commit.boundary)
            self._upsert_head(cur, commit.advance_head.to_head())
            state = commit.content_replacement_state
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_content_states")}
                    (session_id, state_key, state_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id, state_key)
                DO UPDATE SET state_json = EXCLUDED.state_json
                """,
                (
                    session_id,
                    state.state_key,
                    _json_dumps(serialize_content_state(state)),
                ),
            )

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        with self._transaction() as cur:
            cur.execute(
                f"""
                SELECT meta_json
                FROM {self._table("trajectory_sessions")}
                WHERE id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(session_id)
            meta = self._codec.deserialize_session_meta(  # type: ignore[arg-type]
                dict(_json_mapping(row[0]))
            )
            if not isinstance(meta, SessionMeta):
                raise TypeError(
                    "trajectory session metadata codec returned invalid data"
                )
            cur.execute(
                f"""
                SELECT turn_json
                FROM {self._table("trajectory_turns")}
                WHERE session_id = %s
                ORDER BY turn_index
                """,
                (session_id,),
            )
            turns = [
                self._codec.deserialize_turn(  # type: ignore[arg-type]
                    dict(_json_mapping(record[0]))
                )
                for record in cur.fetchall()
            ]
        return meta, turns

    def load_prefix(
        self,
        session_id: str,
        up_to: TurnRef,
    ) -> tuple[SessionMeta, list[Turn]]:
        meta, turns = self.load(session_id)
        cut = turn_prefix_cut(turns, up_to)
        return meta, turns[: cut + 1]

    def session_children(self, session_id: str) -> list[str]:
        with self._transaction() as cur:
            cur.execute(
                f"""
                SELECT id
                FROM {self._table("trajectory_sessions")}
                WHERE parent_id = %s
                ORDER BY created_at
                """,
                (session_id,),
            )
            return [_required_str(record[0], column="id") for record in cur.fetchall()]

    def session_exists(self, session_id: str) -> bool:
        with self._transaction() as cur:
            cur.execute(
                f"""
                SELECT 1
                FROM {self._table("trajectory_sessions")}
                WHERE id = %s
                """,
                (session_id,),
            )
            return cur.fetchone() is not None

    def list_sessions(self) -> list[SessionMeta]:
        with self._transaction() as cur:
            cur.execute(
                f"""
                SELECT meta_json
                FROM {self._table("trajectory_sessions")}
                ORDER BY created_at
                """
            )
            metas: list[SessionMeta] = []
            for (raw,) in cur.fetchall():
                try:
                    meta = self._codec.deserialize_session_meta(  # type: ignore[arg-type]
                        dict(_json_mapping(raw))
                    )
                except (ValueError, KeyError, TypeError):
                    continue
                if not isinstance(meta, SessionMeta):
                    continue
                metas.append(meta)
        return metas

    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        where, params = _node_query_where(query)
        direction = "DESC" if query.sort == "desc" else "ASC"
        order = (
            f"seq {direction}"
            if query.session_id
            else f"session_id {direction}, seq {direction}"
        )
        limit = ""
        if query.limit is not None:
            limit = "LIMIT %s"
            params.append(query.limit)
        with self._transaction() as cur:
            if query.session_id:
                self._require_session(cur, query.session_id)
            cur.execute(
                f"""
                SELECT node_json FROM {self._table("trajectory_nodes")}
                {where}
                ORDER BY {order}
                {limit}
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        return [deserialize_node(_json_mapping(row[0])) for row in rows]

    def get_head(
        self,
        session_id: str,
        *,
        head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> TrajectoryHead | None:
        with self._transaction() as cur:
            self._require_session(cur, session_id)
            cur.execute(
                f"""
                SELECT head_json FROM {self._table("trajectory_heads")}
                WHERE session_id = %s AND head_id = %s
                """,
                (session_id, head_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        head = deserialize_head(_json_mapping(row[0]))
        if branch_id is not None and head.branch_id != branch_id:
            return None
        if agent_id is not None and head.agent_id != agent_id:
            return None
        if is_sidechain is not None and head.is_sidechain != is_sidechain:
            return None
        return head

    def list_heads(
        self,
        session_id: str,
        *,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
        include_inactive: bool = False,
    ) -> list[TrajectoryHead]:
        with self._transaction() as cur:
            self._require_session(cur, session_id)
            cur.execute(
                f"""
                SELECT head_json FROM {self._table("trajectory_heads")}
                WHERE session_id = %s
                ORDER BY updated_at, head_id
                """,
                (session_id,),
            )
            rows = cur.fetchall()
        heads = [deserialize_head(_json_mapping(row[0])) for row in rows]
        if branch_id is not None:
            heads = [head for head in heads if head.branch_id == branch_id]
        if agent_id is not None:
            heads = [head for head in heads if head.agent_id == agent_id]
        if is_sidechain is not None:
            heads = [head for head in heads if head.is_sidechain == is_sidechain]
        if not include_inactive:
            heads = [head for head in heads if head.status == "active"]
        return heads

    def load_chain(
        self,
        session_id: str,
        leaf_node_id: str,
        *,
        include_logical_parent: bool = False,
    ) -> list[TrajectoryNode]:
        parent_expression = (
            "COALESCE(chain.parent_id, chain.logical_parent_id)"
            if include_logical_parent
            else "chain.parent_id"
        )
        with self._transaction() as cur:
            self._require_session(cur, session_id)
            cur.execute(
                f"""
                WITH RECURSIVE chain AS (
                    SELECT node.id, node.parent_id, node.logical_parent_id,
                           node.kind, node.node_json, 0 AS depth
                    FROM {self._table("trajectory_nodes")} AS node
                    WHERE node.id = %s
                      AND (
                          node.session_id = %s
                          OR (
                              %s
                              AND EXISTS (
                                  SELECT 1
                                  FROM {self._table("trajectory_heads")} AS head
                                  WHERE head.session_id = %s
                                    AND head.logical_parent_id = node.id
                              )
                          )
                      )
                    UNION ALL
                    SELECT parent.id, parent.parent_id, parent.logical_parent_id,
                           parent.kind, parent.node_json, chain.depth + 1
                    FROM {self._table("trajectory_nodes")} AS parent
                    JOIN chain ON parent.id = {parent_expression}
                    WHERE (%s OR chain.kind <> 'compact_boundary')
                      AND (%s OR parent.session_id = %s)
                )
                SELECT node_json FROM chain ORDER BY depth DESC
                """,
                (
                    leaf_node_id,
                    session_id,
                    include_logical_parent,
                    session_id,
                    include_logical_parent,
                    include_logical_parent,
                    session_id,
                ),
            )
            rows = cur.fetchall()
        if not rows:
            raise ValueError(f"unknown trajectory leaf node: {leaf_node_id}")
        return [deserialize_node(_json_mapping(row[0])) for row in rows]

    def leaves(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> list[TrajectoryLeaf]:
        clauses = [
            "node.session_id = %s",
            "node.kind IN ('message', 'compact_boundary')",
        ]
        params: list[object] = [session_id]
        if agent_id is not None:
            clauses.append("node.agent_id = %s")
            params.append(agent_id)
        if is_sidechain is not None:
            clauses.append("node.is_sidechain = %s")
            params.append(is_sidechain)
        clauses.append(
            f"""
            NOT EXISTS (
                SELECT 1 FROM {self._table("trajectory_nodes")} AS child
                WHERE child.parent_id = node.id
            )
            """
        )
        with self._transaction() as cur:
            self._require_session(cur, session_id)
            cur.execute(
                f"""
                SELECT session_id, id, seq, branch_id, head_id, agent_id,
                       is_sidechain
                FROM {self._table("trajectory_nodes")} AS node
                WHERE {" AND ".join(clauses)}
                ORDER BY seq
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        return [
            TrajectoryLeaf(
                session_id=_required_str(row[0], column="session_id"),
                node_id=_required_str(row[1], column="id"),
                seq=_required_int(row[2], column="seq"),
                branch_id=_required_str(row[3], column="branch_id"),
                head_id=_required_str(row[4], column="head_id"),
                agent_id=_optional_str(row[5], column="agent_id"),
                is_sidechain=_required_bool(
                    row[6],
                    column="is_sidechain",
                ),
            )
            for row in rows
        ]

    def save_content_replacement_state(
        self,
        session_id: str,
        state: ContentReplacementState,
    ) -> None:
        with self._transaction() as cur:
            self._require_session(cur, session_id, for_update=True)
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_content_states")}
                    (session_id, state_key, state_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id, state_key)
                DO UPDATE SET state_json = EXCLUDED.state_json
                """,
                (
                    session_id,
                    state.state_key,
                    _json_dumps(serialize_content_state(state)),
                ),
            )

    def load_content_replacement_state(
        self,
        session_id: str,
        state_key: str,
    ) -> ContentReplacementState | None:
        with self._transaction() as cur:
            self._require_session(cur, session_id)
            cur.execute(
                f"""
                SELECT state_json FROM {self._table("trajectory_content_states")}
                WHERE session_id = %s AND state_key = %s
                """,
                (session_id, state_key),
            )
            row = cur.fetchone()
        return None if row is None else deserialize_content_state(_json_mapping(row[0]))

    def clone_content_replacement_state(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        state_key: str,
        target_leaf_id: str | None = None,
    ) -> ContentReplacementState | None:
        with self._transaction() as cur:
            self._lock_sessions(
                cur,
                (source_session_id, target_session_id),
            )
            cur.execute(
                f"""
                SELECT state_json
                FROM {self._table("trajectory_content_states")}
                WHERE session_id = %s AND state_key = %s
                """,
                (source_session_id, state_key),
            )
            row = cur.fetchone()
            if row is None:
                return None
            state = deserialize_content_state(_json_mapping(row[0]))
            cloned = replace(
                state,
                source_session_id=source_session_id,
                source_leaf_id=state.leaf_node_id or state.source_leaf_id,
                leaf_node_id=target_leaf_id,
            )
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_content_states")}
                    (session_id, state_key, state_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id, state_key)
                DO UPDATE SET state_json = EXCLUDED.state_json
                """,
                (
                    target_session_id,
                    cloned.state_key,
                    _json_dumps(serialize_content_state(cloned)),
                ),
            )
            return cloned

    def save_prompt_cache_state(
        self,
        session_id: str,
        state: PromptCacheState,
    ) -> None:
        with self._transaction() as cur:
            self._require_session(cur, session_id, for_update=True)
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_prompt_cache_states")}
                    (session_id, cache_key, state_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id, cache_key)
                DO UPDATE SET state_json = EXCLUDED.state_json
                """,
                (
                    session_id,
                    state.cache_key,
                    _json_dumps(serialize_prompt_cache_state(state)),
                ),
            )

    def load_prompt_cache_state(
        self,
        session_id: str,
        cache_key: str,
    ) -> PromptCacheState | None:
        with self._transaction() as cur:
            self._require_session(cur, session_id)
            cur.execute(
                f"""
                SELECT state_json FROM {self._table("trajectory_prompt_cache_states")}
                WHERE session_id = %s AND cache_key = %s
                """,
                (session_id, cache_key),
            )
            row = cur.fetchone()
        return (
            None
            if row is None
            else deserialize_prompt_cache_state(_json_mapping(row[0]))
        )

    def _insert_node(self, cur: PostgresCursor, node: TrajectoryNode) -> None:
        data = serialize_node(node)
        cur.execute(
            f"""
            INSERT INTO {self._table("trajectory_nodes")} (
                id, session_id, root_session_id, parent_session_id, seq,
                parent_id, logical_parent_id, branch_id, head_id, agent_id,
                is_sidechain, turn_id, turn_index, run_id, run_step, message_index,
                kind, role, visibility, tool_call_ids, tool_names, cache_key,
                content_ref, timestamp, node_json
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb, %s::jsonb, %s,
                %s, %s, %s::jsonb
            )
            """,
            (
                node.id,
                node.session_id,
                node.root_session_id,
                node.parent_session_id,
                node.seq,
                node.parent_id,
                node.logical_parent_id,
                node.branch_id,
                node.head_id,
                node.agent_id,
                node.is_sidechain,
                node.turn_id,
                node.turn_index,
                node.run_id,
                node.run_step,
                node.message_index,
                node.kind,
                node.role,
                node.visibility,
                _json_dumps(list(node.tool_call_ids)),
                _json_dumps(list(node.tool_names)),
                node.cache_key,
                node.content_ref,
                node.timestamp,
                _json_dumps(data),
            ),
        )

    def _upsert_head(self, cur: PostgresCursor, head: TrajectoryHead) -> None:
        cur.execute(
            f"""
            INSERT INTO {self._table("trajectory_heads")} (
                session_id, head_id, root_session_id, parent_session_id,
                branch_id, agent_id, is_sidechain, node_id, seq,
                logical_parent_id, status, updated_at, head_json
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb
            )
            ON CONFLICT (session_id, head_id)
            DO UPDATE SET
                root_session_id = EXCLUDED.root_session_id,
                parent_session_id = EXCLUDED.parent_session_id,
                branch_id = EXCLUDED.branch_id,
                agent_id = EXCLUDED.agent_id,
                is_sidechain = EXCLUDED.is_sidechain,
                node_id = EXCLUDED.node_id,
                seq = EXCLUDED.seq,
                logical_parent_id = EXCLUDED.logical_parent_id,
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at,
                head_json = EXCLUDED.head_json
            """,
            (
                head.session_id,
                head.head_id,
                head.root_session_id,
                head.parent_session_id,
                head.branch_id,
                head.agent_id,
                head.is_sidechain,
                head.node_id,
                head.seq,
                head.logical_parent_id,
                head.status,
                head.updated_at,
                _json_dumps(serialize_head(head)),
            ),
        )

    def _insert_turn(
        self,
        cur: PostgresCursor,
        session_id: str,
        turn: Turn,
    ) -> None:
        turn_json = _json_dumps(self._codec.serialize_turn(turn))
        cur.execute(
            f"""
            INSERT INTO {self._table("trajectory_turns")}
                (session_id, turn_index, turn_id, turn_json)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (session_id, turn.index, turn.id, turn_json),
        )

    def _lock_session_turns(
        self,
        cur: PostgresCursor,
        session_id: str,
    ) -> list[Turn]:
        self._require_session(cur, session_id, for_update=True)
        cur.execute(
            f"""
            SELECT turn_json
            FROM {self._table("trajectory_turns")}
            WHERE session_id = %s
            ORDER BY turn_index
            """,
            (session_id,),
        )
        return [
            self._codec.deserialize_turn(  # type: ignore[arg-type]
                dict(_json_mapping(record[0]))
            )
            for record in cur.fetchall()
        ]

    def _select_checkpoint(
        self,
        cur: PostgresCursor,
        session_id: str,
    ) -> TurnCheckpoint | None:
        cur.execute(
            f"""
            SELECT checkpoint_json
            FROM {self._table("trajectory_checkpoints")}
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._codec.deserialize_turn_checkpoint(  # type: ignore[arg-type]
            dict(_json_mapping(row[0]))
        )

    def _validate_head_advance(
        self,
        cur: PostgresCursor,
        session_id: str,
        advance: TrajectoryHeadAdvance,
        nodes: Sequence[TrajectoryNode],
    ) -> TrajectoryHead | None:
        if advance.session_id != session_id:
            raise ValueError("head advance session_id does not match append session")
        if advance.node_id not in {node.id for node in nodes}:
            raise ValueError("head advance node_id is not part of the append batch")
        cur.execute(
            f"""
            SELECT head_json FROM {self._table("trajectory_heads")}
            WHERE session_id = %s AND head_id = %s
            FOR UPDATE
            """,
            (session_id, advance.head_id),
        )
        row = cur.fetchone()
        current_head = None if row is None else deserialize_head(_json_mapping(row[0]))
        if row is None and advance.previous_node_id is not None:
            cur.execute(
                f"""
                SELECT 1 FROM {self._table("trajectory_nodes")}
                WHERE id = %s
                LIMIT 1
                """,
                (advance.previous_node_id,),
            )
            if cur.fetchone() is None:
                raise ValueError(
                    f"head advance previous_node_id is unknown: {advance.previous_node_id}"
                )
            return None
        current_node_id = current_head.node_id if current_head is not None else None
        if current_node_id != advance.previous_node_id:
            raise ValueError(
                "trajectory head changed before append: "
                f"{current_node_id!r} != {advance.previous_node_id!r}"
            )
        return current_head

    def _next_node_seq(self, cur: PostgresCursor, session_id: str) -> int:
        cur.execute(
            f"""
            SELECT MAX(seq)
            FROM {self._table("trajectory_nodes")}
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return 0
        return _required_int(row[0], column="max(seq)") + 1

    def _require_session(
        self,
        cur: PostgresCursor,
        session_id: str,
        *,
        for_update: bool = False,
    ) -> None:
        lock = "FOR UPDATE" if for_update else ""
        cur.execute(
            f"""
            SELECT 1
            FROM {self._table("trajectory_sessions")}
            WHERE id = %s
            {lock}
            """,
            (session_id,),
        )
        if cur.fetchone() is None:
            raise KeyError(session_id)

    def _lock_sessions(
        self,
        cur: PostgresCursor,
        session_ids: Sequence[str],
    ) -> None:
        for session_id in sorted(set(session_ids)):
            self._require_session(cur, session_id, for_update=True)

    def _ensure_session_foreign_key(
        self,
        cur: PostgresCursor,
        table: str,
    ) -> None:
        constraint = f"agentm_{table}_session_fk"
        cur.execute(
            """
            SELECT 1
            FROM pg_constraint
            WHERE conrelid = %s::regclass AND conname = %s
            """,
            (self._table(table), constraint),
        )
        if cur.fetchone() is not None:
            return
        cur.execute(
            f"""
            ALTER TABLE {self._table(table)}
            ADD CONSTRAINT "{constraint}"
            FOREIGN KEY (session_id)
            REFERENCES {self._table("trajectory_sessions")}(id)
            """
        )

    @contextmanager
    def _transaction(self) -> Iterator[PostgresCursor]:
        with self._lock:
            if isinstance(self._handle, Engine):
                with self._handle.begin() as connection:
                    yield PostgresCursor(connection)
                return
            with self._handle.begin():
                yield PostgresCursor(self._handle)

    def _table(self, table: str) -> str:
        return f'"{self._schema}"."agentm_{table}"'

    def _index(self, name: str) -> str:
        return f"agentm_trajectory_{name}"


def _index_statements(schema: str) -> tuple[str, ...]:
    prefix = f'"{schema}"."agentm_trajectory_nodes"'
    heads = f'"{schema}"."agentm_trajectory_heads"'
    return (
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_parent_idx ON {prefix} (session_id, parent_id)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_branch_seq_idx ON {prefix} (session_id, branch_id, head_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_logical_parent_idx ON {prefix} (logical_parent_id, session_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_agent_leaf_idx ON {prefix} (session_id, agent_id, is_sidechain, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_root_session_seq_idx ON {prefix} (root_session_id, session_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_turn_v3_idx ON {prefix} (session_id, turn_index, message_index, turn_id)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_prompt_run_idx ON {prefix} (session_id, run_id, run_step, message_index)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_tool_call_idx ON {prefix} USING gin (tool_call_ids)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_tool_name_idx ON {prefix} USING gin (tool_names)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_cache_idx ON {prefix} (root_session_id, cache_key, session_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_content_ref_idx ON {prefix} (content_ref, session_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_visibility_idx ON {prefix} (session_id, kind, role, visibility, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_session_timestamp_idx ON {prefix} (session_id, timestamp, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_heads_branch_idx ON {heads} (root_session_id, session_id, branch_id, agent_id, is_sidechain)",
    )


def _node_query_where(
    query: TrajectoryNodeQuery,
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []

    def equal(column: str, value: object | None) -> None:
        if value is not None and value != "":
            clauses.append(f"{column} = %s")
            params.append(value)

    equal("session_id", query.session_id)
    equal("id", query.node_id)
    equal("root_session_id", query.root_session_id)
    equal("parent_session_id", query.parent_session_id)
    equal("branch_id", query.branch_id)
    equal("head_id", query.head_id)
    equal("agent_id", query.agent_id)
    if query.is_sidechain is not None:
        equal("is_sidechain", query.is_sidechain)
    if query.kinds:
        clauses.append("kind = ANY(%s)")
        params.append(list(query.kinds))
    equal("role", query.role)
    equal("parent_id", query.parent_id)
    equal("logical_parent_id", query.logical_parent_id)
    equal("turn_id", query.turn_id)
    if query.turn_index is not None:
        equal("turn_index", query.turn_index)
    equal("run_id", query.run_id)
    if query.run_step is not None:
        equal("run_step", query.run_step)
    if query.message_index is not None:
        equal("message_index", query.message_index)
    if query.tool_call_id is not None:
        clauses.append("tool_call_ids ? %s")
        params.append(query.tool_call_id)
    if query.tool_name is not None:
        clauses.append("tool_names ? %s")
        params.append(query.tool_name)
    equal("cache_key", query.cache_key)
    equal("content_ref", query.content_ref)
    equal("visibility", query.visibility)
    if query.after_seq is not None:
        clauses.append("seq > %s")
        params.append(query.after_seq)
    if query.before_seq is not None:
        clauses.append("seq < %s")
        params.append(query.before_seq)
    if query.since_timestamp is not None:
        clauses.append("timestamp >= %s")
        params.append(query.since_timestamp)
    if query.until_timestamp is not None:
        clauses.append("timestamp <= %s")
        params.append(query.until_timestamp)
    return (
        "WHERE " + " AND ".join(clauses) if clauses else "",
        params,
    )


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} is not a valid SQL identifier: {value!r}")
    return value


def _json_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Postgres JSON object keys must be strings")
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        parsed: object = json.loads(value)
        if isinstance(parsed, Mapping):
            if not all(isinstance(key, str) for key in parsed):
                raise ValueError("Postgres JSON object keys must be strings")
            return {str(key): item for key, item in parsed.items()}
    raise ValueError("Postgres JSON column must contain an object")


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _required_str(value: object, *, column: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Postgres column {column!r} must contain a non-empty string")
    return value


def _optional_str(value: object, *, column: str) -> str | None:
    if value is None:
        return None
    return _required_str(value, column=column)


def _required_int(value: object, *, column: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Postgres column {column!r} must contain an integer")
    return value


def _required_bool(value: object, *, column: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Postgres column {column!r} must contain a boolean")
    return value


__all__ = [
    "PostgresCursor",
    "PostgresTrajectoryStore",
]
