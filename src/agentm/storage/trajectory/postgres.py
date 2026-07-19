"""Postgres implementation of ``TrajectoryNodeStore``."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
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
from agentm.core.runtime.trajectory_nodes import (
    InMemoryTrajectoryNodeStore,
    build_chain,
    leaf_nodes,
)
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


class PostgresTrajectoryNodeStore:
    """Postgres-backed trajectory node/head store.

    The tables use denormalized columns for the portable index contract and a
    JSONB payload for lossless ABI reconstruction. This adapter assumes a
    psycopg-style sync connection and keeps methods blocking, matching the SDK
    store protocol.
    """

    def __init__(
        self,
        connection: object,
        *,
        schema: str = "public",
        create_schema: bool = True,
    ) -> None:
        self._connection = connection
        self._schema = schema
        if create_schema:
            self.create_schema()

    @property
    def indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_NODE_INDEXES

    @property
    def head_indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_HEAD_INDEXES

    def create_schema(self) -> None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_nodes")} (
                    id text PRIMARY KEY,
                    session_id text NOT NULL,
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
                    round_index bigint,
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
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_heads")} (
                    session_id text NOT NULL,
                    head_id text NOT NULL,
                    root_session_id text,
                    parent_session_id text,
                    branch_id text NOT NULL,
                    agent_id text,
                    is_sidechain boolean NOT NULL DEFAULT false,
                    node_id text,
                    seq bigint,
                    status text NOT NULL,
                    updated_at double precision NOT NULL DEFAULT 0,
                    head_json jsonb NOT NULL,
                    PRIMARY KEY (session_id, head_id)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_projection_status")} (
                    session_id text PRIMARY KEY,
                    status_json jsonb NOT NULL
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_content_states")} (
                    session_id text NOT NULL,
                    state_key text NOT NULL,
                    state_json jsonb NOT NULL,
                    PRIMARY KEY (session_id, state_key)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("trajectory_prompt_cache_states")} (
                    session_id text NOT NULL,
                    cache_key text NOT NULL,
                    state_json jsonb NOT NULL,
                    PRIMARY KEY (session_id, cache_key)
                )
                """
            )
            for statement in _index_statements(self._schema):
                cur.execute(statement)
        _commit(self._connection)

    def append_nodes(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        advance_head: TrajectoryHeadAdvance | None = None,
    ) -> None:
        if not nodes:
            return
        try:
            with _cursor(self._connection) as cur:
                if advance_head is not None:
                    self._validate_head_advance(cur, session_id, advance_head, nodes)
                for node in nodes:
                    self._insert_node(cur, node)
                if advance_head is not None:
                    self._upsert_head(cur, advance_head.to_head())
                self._upsert_projection_status(
                    cur,
                    _projection_status(session_id, self.query_nodes(TrajectoryNodeQuery(session_id=session_id))),
                )
            _commit(self._connection)
        except Exception:
            _rollback(self._connection)
            raise

    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        nodes = self._load_nodes(session_id=query.session_id or None)
        store = InMemoryTrajectoryNodeStore()
        for session_id in sorted({node.session_id for node in nodes}):
            session_nodes = sorted(
                [node for node in nodes if node.session_id == session_id],
                key=lambda node: node.seq,
            )
            store.replace_session_projection(session_id, session_nodes)
        return store.query_nodes(query)

    def get_head(
        self,
        session_id: str,
        *,
        head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> TrajectoryHead | None:
        with _cursor(self._connection) as cur:
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
        with _cursor(self._connection) as cur:
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
        nodes = self._load_nodes(session_id=None if include_logical_parent else session_id)
        return build_chain(
            nodes,
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
        nodes = self.query_nodes(
            TrajectoryNodeQuery(
                session_id=session_id,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
            )
        )
        return leaf_nodes(nodes)

    def replace_session_projection(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        heads: Sequence[TrajectoryHead] = (),
        status: TrajectoryProjectionStatus | None = None,
    ) -> None:
        try:
            with _cursor(self._connection) as cur:
                cur.execute(
                    f"DELETE FROM {self._table('trajectory_nodes')} WHERE session_id = %s",
                    (session_id,),
                )
                cur.execute(
                    f"DELETE FROM {self._table('trajectory_heads')} WHERE session_id = %s",
                    (session_id,),
                )
                cur.execute(
                    f"DELETE FROM {self._table('trajectory_projection_status')} WHERE session_id = %s",
                    (session_id,),
                )
                for node in nodes:
                    self._insert_node(cur, node)
                for head in heads:
                    self._upsert_head(cur, head)
                self._upsert_projection_status(
                    cur,
                    status if status is not None else _projection_status(session_id, nodes),
                )
            _commit(self._connection)
        except Exception:
            _rollback(self._connection)
            raise

    def projection_status(
        self,
        session_id: str,
    ) -> TrajectoryProjectionStatus | None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                SELECT status_json FROM {self._table("trajectory_projection_status")}
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
        return None if row is None else deserialize_projection_status(_json_mapping(row[0]))

    def save_content_replacement_state(
        self,
        session_id: str,
        state: ContentReplacementState,
    ) -> None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_content_states")}
                    (session_id, state_key, state_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id, state_key)
                DO UPDATE SET state_json = EXCLUDED.state_json
                """,
                (session_id, state.state_key, _json_dumps(serialize_content_state(state))),
            )
        _commit(self._connection)

    def load_content_replacement_state(
        self,
        session_id: str,
        state_key: str,
    ) -> ContentReplacementState | None:
        with _cursor(self._connection) as cur:
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
        state = self.load_content_replacement_state(source_session_id, state_key)
        if state is None:
            return None
        from dataclasses import replace

        cloned = replace(
            state,
            source_session_id=source_session_id,
            source_leaf_id=state.leaf_node_id or state.source_leaf_id,
            leaf_node_id=target_leaf_id,
        )
        self.save_content_replacement_state(target_session_id, cloned)
        return cloned

    def save_prompt_cache_state(
        self,
        session_id: str,
        state: PromptCacheState,
    ) -> None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table("trajectory_prompt_cache_states")}
                    (session_id, cache_key, state_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id, cache_key)
                DO UPDATE SET state_json = EXCLUDED.state_json
                """,
                (session_id, state.cache_key, _json_dumps(serialize_prompt_cache_state(state))),
            )
        _commit(self._connection)

    def load_prompt_cache_state(
        self,
        session_id: str,
        cache_key: str,
    ) -> PromptCacheState | None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                SELECT state_json FROM {self._table("trajectory_prompt_cache_states")}
                WHERE session_id = %s AND cache_key = %s
                """,
                (session_id, cache_key),
            )
            row = cur.fetchone()
        return None if row is None else deserialize_prompt_cache_state(_json_mapping(row[0]))

    def _insert_node(self, cur: Any, node: TrajectoryNode) -> None:
        data = serialize_node(node)
        cur.execute(
            f"""
            INSERT INTO {self._table("trajectory_nodes")} (
                id, session_id, root_session_id, parent_session_id, seq,
                parent_id, logical_parent_id, branch_id, head_id, agent_id,
                is_sidechain, turn_id, turn_index, round_index, message_index,
                kind, role, visibility, tool_call_ids, tool_names, cache_key,
                content_ref, timestamp, node_json
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
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
                node.round_index,
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

    def _upsert_head(self, cur: Any, head: TrajectoryHead) -> None:
        cur.execute(
            f"""
            INSERT INTO {self._table("trajectory_heads")} (
                session_id, head_id, root_session_id, parent_session_id,
                branch_id, agent_id, is_sidechain, node_id, seq, status,
                updated_at, head_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (session_id, head_id)
            DO UPDATE SET
                root_session_id = EXCLUDED.root_session_id,
                parent_session_id = EXCLUDED.parent_session_id,
                branch_id = EXCLUDED.branch_id,
                agent_id = EXCLUDED.agent_id,
                is_sidechain = EXCLUDED.is_sidechain,
                node_id = EXCLUDED.node_id,
                seq = EXCLUDED.seq,
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
                head.status,
                head.updated_at,
                _json_dumps(serialize_head(head)),
            ),
        )

    def _upsert_projection_status(
        self,
        cur: Any,
        status: TrajectoryProjectionStatus,
    ) -> None:
        cur.execute(
            f"""
            INSERT INTO {self._table("trajectory_projection_status")}
                (session_id, status_json)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (session_id)
            DO UPDATE SET status_json = EXCLUDED.status_json
            """,
            (status.session_id, _json_dumps(serialize_projection_status(status))),
        )

    def _validate_head_advance(
        self,
        cur: Any,
        session_id: str,
        advance: TrajectoryHeadAdvance,
        nodes: Sequence[TrajectoryNode],
    ) -> None:
        if advance.session_id != session_id:
            raise ValueError("head advance session_id does not match append session")
        if advance.node_id not in {node.id for node in nodes}:
            raise ValueError("head advance node_id is not part of the append batch")
        cur.execute(
            f"""
            SELECT node_id FROM {self._table("trajectory_heads")}
            WHERE session_id = %s AND head_id = %s
            FOR UPDATE
            """,
            (session_id, advance.head_id),
        )
        row = cur.fetchone()
        current = row[0] if row is not None else None
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
            return
        if current != advance.previous_node_id:
            raise ValueError(
                "trajectory head changed before append: "
                f"{current!r} != {advance.previous_node_id!r}"
            )

    def _load_nodes(self, *, session_id: str | None) -> list[TrajectoryNode]:
        with _cursor(self._connection) as cur:
            if session_id is None:
                cur.execute(
                    f"""
                    SELECT node_json FROM {self._table("trajectory_nodes")}
                    ORDER BY session_id, seq
                    """
                )
            else:
                cur.execute(
                    f"""
                    SELECT node_json FROM {self._table("trajectory_nodes")}
                    WHERE session_id = %s
                    ORDER BY seq
                    """,
                    (session_id,),
                )
            rows = cur.fetchall()
        return [deserialize_node(_json_mapping(row[0])) for row in rows]

    def _table(self, table: str) -> str:
        return f"{self._schema}.agentm_{table}"


class _cursor:
    def __init__(self, connection: object) -> None:
        self._connection = connection
        self._cursor: Any | None = None

    def __enter__(self) -> Any:
        cursor = getattr(self._connection, "cursor")()
        self._cursor = cursor
        return cursor

    def __exit__(self, *_exc: object) -> None:
        if self._cursor is not None:
            close = getattr(self._cursor, "close", None)
            if callable(close):
                close()


def _projection_status(
    session_id: str,
    nodes: Sequence[TrajectoryNode],
) -> TrajectoryProjectionStatus:
    last = nodes[-1] if nodes else None
    return TrajectoryProjectionStatus(
        session_id=session_id,
        state="current",
        high_water_turn_id=last.turn_id if last is not None else None,
        high_water_turn_index=last.turn_index if last is not None else None,
        node_count=len(nodes),
        updated_at=time.time(),
    )


def _index_statements(schema: str) -> tuple[str, ...]:
    prefix = f"{schema}.agentm_trajectory_nodes"
    heads = f"{schema}.agentm_trajectory_heads"
    return (
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_parent_idx ON {prefix} (session_id, parent_id)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_branch_seq_idx ON {prefix} (session_id, branch_id, head_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_logical_parent_idx ON {prefix} (logical_parent_id, session_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_agent_leaf_idx ON {prefix} (session_id, agent_id, is_sidechain, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_root_session_seq_idx ON {prefix} (root_session_id, session_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_turn_idx ON {prefix} (session_id, turn_index, round_index, message_index, turn_id)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_tool_call_idx ON {prefix} USING gin (tool_call_ids)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_nodes_cache_idx ON {prefix} (root_session_id, cache_key, session_id, seq)",
        f"CREATE INDEX IF NOT EXISTS agentm_trajectory_heads_branch_idx ON {heads} (root_session_id, session_id, branch_id, agent_id, is_sidechain)",
    )


def _json_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _commit(connection: object) -> None:
    commit = getattr(connection, "commit", None)
    if callable(commit):
        commit()


def _rollback(connection: object) -> None:
    rollback = getattr(connection, "rollback", None)
    if callable(rollback):
        rollback()


__all__ = ["PostgresTrajectoryNodeStore"]
