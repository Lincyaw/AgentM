"""ClickHouse read-model query adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from typing import Any

from agentm.core.abi.codec import DEFAULT_CODEC
from agentm.core.abi.query import (
    EventRecord,
    QueryMeta,
    SessionFilter,
    SessionIdentity,
    SpanRecord,
)
from agentm.core.abi.trajectory import Turn


class ClickHouseTrajectoryQueryStore:
    """Trajectory/observability query store over a ClickHouse client.

    The client is expected to expose ``query(sql, parameters=...)`` or
    ``execute(sql, params)``. This adapter is read-only: ClickHouse is a query
    mirror and never owns current trajectory heads.
    """

    def __init__(self, client: object, *, database: str = "agentm") -> None:
        self._client = client
        self._database = database

    def sessions(
        self,
        filter: SessionFilter | None = None,
    ) -> Iterable[SessionIdentity]:
        where, params = _session_where(filter)
        rows = self._query(
            f"""
            SELECT id, parent_session_id, root_session_id, purpose, cwd, created_at, config
            FROM {self._database}.sessions
            {where}
            ORDER BY created_at, id
            {f"LIMIT {filter.limit}" if filter is not None and filter.limit is not None else ""}
            """,
            params,
        )
        return [
            SessionIdentity(
                id=str(row[0]),
                parent_session_id=_optional_str(row[1]),
                root_session_id=_optional_str(row[2]),
                purpose=str(row[3] or "root"),
                cwd=str(row[4] or ""),
                created_at=float(row[5] or 0.0),
                config=_json_mapping(row[6]),
            )
            for row in rows
        ]

    def turns(self, session_id: str) -> Iterable[Turn]:
        rows = self._query(
            f"""
            SELECT turn_json
            FROM {self._database}.turns
            WHERE session_id = %(session_id)s
            ORDER BY turn_index
            """,
            {"session_id": session_id},
        )
        return [DEFAULT_CODEC.deserialize_turn(_json_mapping(row[0])) for row in rows]

    def events(self, session_id: str) -> Iterable[EventRecord]:
        rows = self._query(
            f"""
            SELECT name, timestamp, payload, attributes
            FROM {self._database}.events
            WHERE session_id = %(session_id)s
            ORDER BY timestamp
            """,
            {"session_id": session_id},
        )
        return [
            EventRecord(
                session_id=session_id,
                name=str(row[0]),
                timestamp=float(row[1] or 0.0),
                payload=_json_mapping(row[2]),
                meta=QueryMeta(attributes=_json_mapping(row[3])),
            )
            for row in rows
        ]

    def spans(self, session_id: str) -> Iterable[SpanRecord]:
        rows = self._query(
            f"""
            SELECT name, span_id, parent_span_id, start_time, end_time, attributes
            FROM {self._database}.spans
            WHERE session_id = %(session_id)s
            ORDER BY start_time
            """,
            {"session_id": session_id},
        )
        return [
            SpanRecord(
                session_id=session_id,
                name=str(row[0]),
                span_id=str(row[1] or ""),
                parent_span_id=_optional_str(row[2]),
                start_time=float(row[3] or 0.0),
                end_time=float(row[4]) if row[4] is not None else None,
                attributes=_json_mapping(row[5]),
            )
            for row in rows
        ]

    def _query(self, sql: str, params: dict[str, object]) -> Sequence[Sequence[Any]]:
        query = getattr(self._client, "query", None)
        if callable(query):
            result = query(sql, parameters=params)
            rows = getattr(result, "result_rows", result)
            return list(rows)
        execute = getattr(self._client, "execute", None)
        if callable(execute):
            return list(execute(sql, params))
        raise TypeError("ClickHouse client must expose query() or execute()")


def _session_where(filter: SessionFilter | None) -> tuple[str, dict[str, object]]:
    if filter is None:
        return "", {}
    clauses: list[str] = []
    params: dict[str, object] = {}
    for attr, column in (
        ("session_id", "id"),
        ("parent_session_id", "parent_session_id"),
        ("root_session_id", "root_session_id"),
        ("purpose", "purpose"),
    ):
        value = getattr(filter, attr)
        if value is not None:
            key = attr
            clauses.append(f"{column} = %({key})s")
            params[key] = value
    if filter.since is not None:
        clauses.append("created_at >= %(since)s")
        params["since"] = filter.since
    if filter.until is not None:
        clauses.append("created_at <= %(until)s")
        params["until"] = filter.until
    return ("WHERE " + " AND ".join(clauses), params) if clauses else ("", params)


def _json_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["ClickHouseTrajectoryQueryStore"]
