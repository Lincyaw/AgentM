# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""Typed psycopg 3 adapter for AgentM's driver-neutral Postgres ports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import psycopg
from psycopg import Connection, Cursor

from agentm.storage.trajectory.postgres import (
    PostgresCursor,
)

PsycopgRow = tuple[object, ...]


class PsycopgCursorAdapter:
    """Normalize psycopg's generic cursor to the SDK's minimal cursor port."""

    __slots__ = ("_cursor",)

    def __init__(self, cursor: Cursor[PsycopgRow]) -> None:
        self._cursor = cursor

    def execute(self, query: str, params: object | None = None) -> object:
        if params is None:
            return self._cursor.execute(query)
        if isinstance(params, Mapping):
            if not all(isinstance(key, str) for key in params):
                raise TypeError("Postgres mapping parameters require string keys")
            return self._cursor.execute(
                query,
                cast(Mapping[str, object], params),
            )
        if isinstance(params, Sequence) and not isinstance(
            params,
            (str, bytes, bytearray),
        ):
            return self._cursor.execute(query, tuple(params))
        raise TypeError("Postgres parameters must be a mapping or non-string sequence")

    def fetchone(self) -> Sequence[object] | None:
        return self._cursor.fetchone()

    def fetchall(self) -> Sequence[Sequence[object]]:
        return self._cursor.fetchall()

    def close(self) -> None:
        self._cursor.close()


class PsycopgConnectionAdapter:
    """Expose one psycopg connection through ``PostgresConnection``."""

    __slots__ = ("_connection",)

    def __init__(self, connection: Connection[PsycopgRow]) -> None:
        self._connection = connection

    def cursor(self) -> PostgresCursor:
        return PsycopgCursorAdapter(self._connection.cursor())

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def connect(dsn: str) -> PsycopgConnectionAdapter:
    """Open a psycopg connection adapted to AgentM's storage port."""
    connection = cast(
        Connection[PsycopgRow],
        psycopg.connect(dsn),
    )
    return PsycopgConnectionAdapter(connection)


__all__ = [
    "PsycopgConnectionAdapter",
    "PsycopgCursorAdapter",
    "connect",
]
