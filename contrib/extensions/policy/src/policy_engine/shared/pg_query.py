# code-health: ignore-file[AM025] -- query parameters arrive from callers as
# whatever they had; this is the one place that is decided.
"""PG-backed QuerySource for signal evaluation.

Uses SQLAlchemy (consistent with the IFG subpackage and the SDK's
storage layer). Session-scoped: signals reference ``%(session_id)s``
which is auto-bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loguru import logger
from sqlalchemy.engine import Connection, Engine

from agentm.storage.sql import create_sql_engine


class PgQuerySource:
    """QuerySource over PG policy views, scoped to one session."""

    def __init__(self, dsn: str, session_id: str) -> None:
        self._dsn = dsn
        self._session_id = session_id
        self._engine: Engine | None = None
        self._conn: Connection | None = None

    def _connection(self) -> Connection:
        if self._conn is None:
            self._engine = create_sql_engine(self._dsn)
            self._conn = self._engine.connect()
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
            result = conn.exec_driver_sql(sql, merged)
            return [tuple(row) for row in result]
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
        conn = self._connection()
        conn.exec_driver_sql(sql, tuple(params))
        conn.commit()

    def executemany(
        self,
        sql: str,
        params_seq: Sequence[tuple[object, ...] | Sequence[object]],
    ) -> None:
        conn = self._connection()
        for params in params_seq:
            conn.exec_driver_sql(sql, tuple(params))
        conn.commit()

    @property
    def session_id(self) -> str:
        return self._session_id

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as close_exc:  # noqa: BLE001
                logger.debug("pg_query: conn close failed: {}", close_exc)
            self._conn = None
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception as eng_exc:  # noqa: BLE001
                logger.debug("pg_query: engine dispose failed: {}", eng_exc)
            self._engine = None
