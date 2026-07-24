"""PG-backed QuerySource for signal evaluation.

Wraps a psycopg connection scoped to one session. Signals reference
``%(session_id)s`` in their SQL; the source binds it automatically.

Also provides write access for tagger annotations and symbol sync,
so all PG I/O shares one long-lived connection per session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import psycopg


class PgQuerySource:
    """QuerySource over PG policy views, scoped to one session."""

    def __init__(self, dsn: str, session_id: str) -> None:
        self._dsn = dsn
        self._session_id = session_id
        self._conn: psycopg.Connection | None = None  # type: ignore[assignment]

    def _connection(self) -> psycopg.Connection:  # type: ignore[name-defined]
        if self._conn is None:
            import psycopg as _psycopg  # noqa: PLC0415

            self._conn = _psycopg.connect(self._dsn)
        return self._conn

    def query(
        self,
        sql: str,
        params: tuple[object, ...] | Mapping[str, object] = (),
    ) -> list[tuple]:
        conn = self._connection()
        merged = dict(params) if isinstance(params, Mapping) else {}
        merged.setdefault("session_id", self._session_id)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, merged)
                return cur.fetchall()  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            logger.warning("pg_query: query failed: {}", exc)
            try:
                conn.rollback()
            except Exception as rollback_exc:  # noqa: BLE001
                logger.debug("pg_query: rollback failed: {}", rollback_exc)
            return []

    def execute(
        self,
        sql: str,
        params: tuple[object, ...] | Sequence[object] = (),
    ) -> None:
        """Execute a write statement (INSERT/UPDATE/DELETE)."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()

    def executemany(
        self,
        sql: str,
        params_seq: Sequence[tuple[object, ...] | Sequence[object]],
    ) -> None:
        conn = self._connection()
        with conn.cursor() as cur:
            cur.executemany(sql, params_seq)
        conn.commit()

    @property
    def session_id(self) -> str:
        return self._session_id

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as close_exc:  # noqa: BLE001
                logger.debug("pg_query: close failed: {}", close_exc)
            self._conn = None
