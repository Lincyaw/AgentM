"""Postgres implementation of ``TrajectoryStore`` (turn-level)."""

from __future__ import annotations

import json
from collections.abc import Sequence

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.store import SessionMeta
from agentm.core.abi.trajectory import Turn, TurnCheckpoint, TurnRef
from agentm.core.lib.trajectory_store import (
    turn_prefix_cut,
    validate_checkpoint_commit,
    validate_turn_append,
    validate_turn_checkpoint,
    validate_turn_sequence,
)
from agentm.storage.trajectory.postgres import (
    PostgresConnection,
    PostgresCursor,
    _commit,
    _cursor,
    _json_mapping,
    _validate_identifier,
)


class PostgresTrajectoryStore:
    """Turn-level trajectory store backed by Postgres.

    ``sessions`` owns metadata, ``checkpoints`` owns the latest incomplete
    turn state, and ``turns`` owns committed JSONB records. Final append and
    checkpoint removal share one Postgres transaction.
    """

    def __init__(
        self,
        connection: PostgresConnection,
        *,
        schema: str = "public",
        codec: CodecRegistry | None = None,
        create_schema: bool = True,
    ) -> None:
        if not isinstance(connection, PostgresConnection):
            raise TypeError(
                "PostgresTrajectoryStore requires a PostgresConnection"
            )
        self._connection = connection
        self._schema = _validate_identifier(schema, label="Postgres schema")
        self._codec = codec if codec is not None else CodecRegistry()
        if create_schema:
            self._create_schema()

    def _create_schema(self) -> None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table("sessions")} (
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
                CREATE TABLE IF NOT EXISTS {self._table("turns")} (
                    session_id text NOT NULL
                        REFERENCES {self._table("sessions")}(id),
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
                CREATE TABLE IF NOT EXISTS {self._table("checkpoints")} (
                    session_id text PRIMARY KEY
                        REFERENCES {self._table("sessions")}(id),
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
                CREATE INDEX IF NOT EXISTS {self._index("turns_session_idx")}
                ON {self._table("turns")} (session_id, turn_index)
                """
            )
            cur.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS
                    {self._index("turns_session_turn_id_idx")}
                ON {self._table("turns")} (session_id, turn_id)
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._index("sessions_parent_idx")}
                ON {self._table("sessions")} (parent_id)
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._index("sessions_created_idx")}
                ON {self._table("sessions")} (created_at)
                """
            )
        _commit(self._connection)

    def save_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        checkpoint_json = json.dumps(
            self._codec.serialize_turn_checkpoint(checkpoint),
            sort_keys=True,
            allow_nan=False,
        )
        with _cursor(self._connection) as cur:
            committed = self._lock_session_turns(cur, session_id)
            existing = self._select_checkpoint(cur, session_id)
            validate_turn_checkpoint(
                committed,
                checkpoint,
                existing=existing,
            )
            cur.execute(
                f"""
                INSERT INTO {self._table("checkpoints")}
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
        _commit(self._connection)

    def load_checkpoint(self, session_id: str) -> TurnCheckpoint | None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                SELECT c.checkpoint_json
                FROM {self._table("sessions")} AS s
                LEFT JOIN {self._table("checkpoints")} AS c
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
            return self._codec.deserialize_turn_checkpoint(
                dict(_json_mapping(row[0]))
            )

    def create_session(self, meta: SessionMeta) -> None:
        self.create_session_with_turns(meta, ())

    def create_session_with_turns(
        self, meta: SessionMeta, turns: Sequence[Turn]
    ) -> None:
        copied_turns = list(turns)
        validate_turn_sequence(copied_turns)
        meta_json = json.dumps(
            self._codec.serialize_session_meta(meta),
            sort_keys=True,
            allow_nan=False,
        )
        with _cursor(self._connection) as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table("sessions")}
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
        _commit(self._connection)

    def append(self, session_id: str, turn: Turn) -> None:
        with _cursor(self._connection) as cur:
            committed = self._lock_session_turns(cur, session_id)
            validate_turn_append(committed, turn)
            validate_checkpoint_commit(
                self._select_checkpoint(cur, session_id),
                turn,
            )
            self._insert_turn(cur, session_id, turn)
            cur.execute(
                f"""
                DELETE FROM {self._table("checkpoints")}
                WHERE session_id = %s
                """,
                (session_id,),
            )
        _commit(self._connection)

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"SELECT meta_json FROM {self._table('sessions')} WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(session_id)
            meta = self._codec.deserialize_session_meta(dict(_json_mapping(row[0])))
            cur.execute(
                f"SELECT turn_json FROM {self._table('turns')} WHERE session_id = %s ORDER BY turn_index",
                (session_id,),
            )
            turns = [
                self._codec.deserialize_turn(dict(_json_mapping(r[0])))
                for r in cur.fetchall()
            ]
        return meta, turns

    def load_prefix(
        self, session_id: str, up_to: TurnRef
    ) -> tuple[SessionMeta, list[Turn]]:
        meta, turns = self.load(session_id)
        cut = turn_prefix_cut(turns, up_to)
        return (meta, turns[: cut + 1])

    def session_children(self, session_id: str) -> list[str]:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"SELECT id FROM {self._table('sessions')} WHERE parent_id = %s ORDER BY created_at",
                (session_id,),
            )
            return [str(r[0]) for r in cur.fetchall()]

    def session_exists(self, session_id: str) -> bool:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"SELECT 1 FROM {self._table('sessions')} WHERE id = %s",
                (session_id,),
            )
            return cur.fetchone() is not None

    def list_sessions(self) -> list[SessionMeta]:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"SELECT meta_json FROM {self._table('sessions')} ORDER BY created_at"
            )
            return [
                self._codec.deserialize_session_meta(dict(_json_mapping(r[0])))
                for r in cur.fetchall()
            ]

    def _insert_turn(
        self, cur: PostgresCursor, session_id: str, turn: Turn
    ) -> None:
        turn_json = json.dumps(
            self._codec.serialize_turn(turn),
            sort_keys=True,
            allow_nan=False,
        )
        cur.execute(
            f"""
            INSERT INTO {self._table("turns")}
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
        cur.execute(
            f"""
            SELECT 1 FROM {self._table("sessions")}
            WHERE id = %s
            FOR UPDATE
            """,
            (session_id,),
        )
        if cur.fetchone() is None:
            raise KeyError(session_id)
        cur.execute(
            f"""
            SELECT turn_json
            FROM {self._table("turns")}
            WHERE session_id = %s
            ORDER BY turn_index
            """,
            (session_id,),
        )
        return [
            self._codec.deserialize_turn(dict(_json_mapping(row[0])))
            for row in cur.fetchall()
        ]

    def _select_checkpoint(
        self,
        cur: PostgresCursor,
        session_id: str,
    ) -> TurnCheckpoint | None:
        cur.execute(
            f"""
            SELECT checkpoint_json
            FROM {self._table("checkpoints")}
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._codec.deserialize_turn_checkpoint(
            dict(_json_mapping(row[0]))
        )

    def _table(self, name: str) -> str:
        return f'"{self._schema}"."agentm_trajectory_{name}"'

    def _index(self, name: str) -> str:
        return f"agentm_trajectory_{name}"

__all__ = ["PostgresTrajectoryStore"]
