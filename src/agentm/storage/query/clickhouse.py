"""ClickHouse read-model query adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
import math
import re
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi.codec import DEFAULT_CODEC
from agentm.core.abi.query import (
    EventRecord,
    QueryMeta,
    SessionFilter,
    SessionIdentity,
    SpanRecord,
)
from agentm.core.abi.trajectory import Turn


@runtime_checkable
class ClickHouseQueryResult(Protocol):
    """Result surface returned by clickhouse-connect."""

    result_rows: Sequence[Sequence[Any]]


@runtime_checkable
class ClickHouseQueryClient(Protocol):
    """Minimal clickhouse-connect query client contract."""

    def query(
        self,
        query: str,
        *,
        parameters: dict[str, object] | None = None,
    ) -> ClickHouseQueryResult: ...


class ClickHouseTraceQueryStore:
    """Trace query store over a ClickHouse client.

    The client follows the ``clickhouse-connect`` query contract. This adapter
    is read-only: ClickHouse is a query mirror and never owns current
    trajectory heads. Other client libraries should supply a small adapter
    implementing :class:`ClickHouseQueryClient`.
    """

    def __init__(
        self,
        client: ClickHouseQueryClient,
        *,
        database: str = "agentm",
    ) -> None:
        if not isinstance(client, ClickHouseQueryClient):
            raise TypeError(
                "ClickHouseTraceQueryStore requires a ClickHouseQueryClient"
            )
        self._client = client
        self._database = _validate_identifier(
            database,
            label="ClickHouse database",
        )

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
                id=_required_str(row[0], column="id"),
                parent_session_id=_optional_str(
                    row[1],
                    column="parent_session_id",
                ),
                root_session_id=_optional_str(
                    row[2],
                    column="root_session_id",
                ),
                purpose=_nullable_str(
                    row[3],
                    column="purpose",
                    default="root",
                ),
                cwd=_nullable_str(row[4], column="cwd", default=""),
                created_at=_finite_float(
                    row[5],
                    column="created_at",
                    default=0.0,
                ),
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
                name=_required_str(row[0], column="name"),
                timestamp=_finite_float(
                    row[1],
                    column="timestamp",
                    default=0.0,
                ),
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
                name=_required_str(row[0], column="name"),
                span_id=_nullable_str(
                    row[1],
                    column="span_id",
                    default="",
                ),
                parent_span_id=_optional_str(
                    row[2],
                    column="parent_span_id",
                ),
                start_time=_finite_float(
                    row[3],
                    column="start_time",
                    default=0.0,
                ),
                end_time=(
                    _finite_float(row[4], column="end_time")
                    if row[4] is not None
                    else None
                ),
                attributes=_json_mapping(row[5]),
            )
            for row in rows
        ]

    def _query(self, sql: str, params: dict[str, object]) -> Sequence[Sequence[Any]]:
        result = self._client.query(sql, parameters=params)
        if not isinstance(result, ClickHouseQueryResult):
            raise TypeError("ClickHouseQueryClient.query() returned an invalid result")
        return list(result.result_rows)


def _session_where(filter: SessionFilter | None) -> tuple[str, dict[str, object]]:
    if filter is None:
        return "", {}
    if filter.limit is not None and filter.limit < 0:
        raise ValueError("session query limit cannot be negative")
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
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return dict(parsed)
    raise ValueError("ClickHouse JSON column must contain an object")


def _required_str(value: object, *, column: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"ClickHouse column {column!r} must contain a non-empty string")
    return value


def _optional_str(value: object, *, column: str) -> str | None:
    if value is None or value == "":
        return None
    return _required_str(value, column=column)


def _nullable_str(value: object, *, column: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"ClickHouse column {column!r} must contain a string or NULL")
    return value


def _finite_float(
    value: object,
    *,
    column: str,
    default: float | None = None,
) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"ClickHouse column {column!r} must contain a number")
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"ClickHouse column {column!r} must contain a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"ClickHouse column {column!r} must contain a finite number")
    return number


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} is not a valid SQL identifier: {value!r}")
    return value


__all__ = [
    "ClickHouseQueryClient",
    "ClickHouseQueryResult",
    "ClickHouseTraceQueryStore",
]
