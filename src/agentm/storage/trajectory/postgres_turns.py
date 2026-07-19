"""Postgres implementation of ``TrajectoryStore`` (turn-level)."""

from __future__ import annotations

import json
from collections.abc import Sequence

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.store import SessionMeta
from agentm.core.abi.trajectory import Turn, TurnRef
from agentm.storage.trajectory.postgres import (
    PostgresConnection,
    PostgresCursor,
    _commit,
    _cursor,
    _json_mapping,
    _rollback,
    _validate_identifier,
)


class PostgresTrajectoryStore:
    """Turn-level trajectory store backed by Postgres.

    Two tables: ``sessions`` (JSONB meta) and ``turns`` (JSONB per-turn,
    TOAST-compressed by Postgres). Append is single-row INSERT with an
    index check.
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
                    PRIMARY KEY (session_id, turn_index)
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

    def create_session(self, meta: SessionMeta) -> None:
        self.create_session_with_turns(meta, ())

    def create_session_with_turns(
        self, meta: SessionMeta, turns: Sequence[Turn]
    ) -> None:
        meta_json = json.dumps(
            self._codec.serialize_session_meta(meta),
            sort_keys=True,
            allow_nan=False,
        )
        with _cursor(self._connection) as cur:
            try:
                cur.execute(
                    f"""
                    INSERT INTO {self._table("sessions")}
                        (id, parent_id, purpose, cwd, created_at, meta_json)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
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
            except Exception as exc:
                if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
                    _rollback(self._connection)
                    raise ValueError(f"session already exists: {meta.id}") from None
                raise
            for turn in turns:
                self._insert_turn(cur, meta.id, turn)
        _commit(self._connection)

    def append(self, session_id: str, turn: Turn) -> None:
        with _cursor(self._connection) as cur:
            cur.execute(
                f"SELECT 1 FROM {self._table('sessions')} WHERE id = %s",
                (session_id,),
            )
            if cur.fetchone() is None:
                raise KeyError(session_id)
            cur.execute(
                f"SELECT MAX(turn_index) FROM {self._table('turns')} WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            raw_max = row[0] if row is not None else None
            max_index: int = int(raw_max) if raw_max is not None else -1  # type: ignore[call-overload]
            expected = max_index + 1
            if turn.index != expected:
                raise ValueError(
                    f"turn index {turn.index} does not follow {max_index}"
                )
            self._insert_turn(cur, session_id, turn)
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
        cut = _prefix_cut(turns, up_to)
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

    def _table(self, name: str) -> str:
        return f'"{self._schema}"."agentm_trajectory_{name}"'

    def _index(self, name: str) -> str:
        return f"agentm_trajectory_{name}"


def _prefix_cut(turns: list[Turn], up_to: TurnRef) -> int:
    if isinstance(up_to, int):
        for i, turn in enumerate(turns):
            if turn.index == up_to:
                return i
        raise KeyError(up_to)
    for i, turn in enumerate(turns):
        if turn.id == up_to:
            return i
    raise KeyError(up_to)


__all__ = ["PostgresTrajectoryStore"]
