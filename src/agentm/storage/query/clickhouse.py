"""ClickHouse adapter for collector-managed OTLP observability."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
import re
from typing import Protocol, runtime_checkable

from agentm.core.abi.query import (
    EventRecord,
    QueryMeta,
    SpanRecord,
)


@runtime_checkable
class ClickHouseQueryResult(Protocol):
    """Result surface returned by a ClickHouse driver."""

    result_rows: Sequence[Sequence[object]]


@runtime_checkable
class ClickHouseQueryClient(Protocol):
    """Minimal driver-neutral ClickHouse query contract."""

    def query(
        self,
        query: str,
        *,
        parameters: Mapping[str, object] | None = None,
    ) -> ClickHouseQueryResult: ...


class ClickHouseObservabilityQueryStore:
    """Read AgentM events and spans from collector-managed OTLP tables.

    This store implements only ``ObservabilityQueryStore``. Full trajectories
    remain in one authoritative ``TrajectoryStore`` backend and are never
    copied into an AgentM-specific ClickHouse schema.
    """

    __slots__ = ("_client", "_database", "_logs_table", "_traces_table")

    def __init__(
        self,
        client: ClickHouseQueryClient,
        *,
        database: str = "otel",
        logs_table: str = "otel_logs",
        traces_table: str = "otel_traces",
    ) -> None:
        if not isinstance(client, ClickHouseQueryClient):
            raise TypeError(
                "ClickHouseObservabilityQueryStore requires "
                "a ClickHouseQueryClient"
            )
        self._client = client
        self._database = _validate_identifier(
            database,
            label="ClickHouse observability database",
        )
        self._logs_table = _validate_identifier(
            logs_table,
            label="ClickHouse logs table",
        )
        self._traces_table = _validate_identifier(
            traces_table,
            label="ClickHouse traces table",
        )

    def events(self, session_id: str) -> Iterable[EventRecord]:
        rows = _query(
            self._client,
            f"""
            SELECT DISTINCT
                EventName,
                toUnixTimestamp64Nano(Timestamp) / 1000000000.0
                    AS timestamp,
                Body,
                LogAttributes
            FROM {self._database}.{self._logs_table}
            WHERE LogAttributes['agentm.session.id'] = %(session_id)s
            ORDER BY timestamp, EventName
            """,
            {"session_id": session_id},
        )
        return [
            EventRecord(
                session_id=session_id,
                name=_required_str(row[0], column="EventName"),
                timestamp=_finite_float(
                    row[1],
                    column="Timestamp",
                    default=0.0,
                ),
                payload=_body_mapping(row[2]),
                meta=QueryMeta(
                    attributes=_json_mapping(
                        row[3],
                        column="LogAttributes",
                    )
                ),
            )
            for row in rows
        ]

    def spans(self, session_id: str) -> Iterable[SpanRecord]:
        rows = _query(
            self._client,
            f"""
            SELECT DISTINCT
                SpanName,
                SpanId,
                ParentSpanId,
                toUnixTimestamp64Nano(Timestamp) / 1000000000.0
                    AS start_time,
                (
                    toUnixTimestamp64Nano(Timestamp) + Duration
                ) / 1000000000.0 AS end_time,
                SpanAttributes
            FROM {self._database}.{self._traces_table}
            WHERE SpanAttributes['agentm.session.id'] = %(session_id)s
            ORDER BY start_time, SpanId
            """,
            {"session_id": session_id},
        )
        return [
            SpanRecord(
                session_id=session_id,
                name=_required_str(row[0], column="SpanName"),
                span_id=_required_str(row[1], column="SpanId"),
                parent_span_id=_optional_str(
                    row[2],
                    column="ParentSpanId",
                ),
                start_time=_finite_float(
                    row[3],
                    column="Timestamp",
                    default=0.0,
                ),
                end_time=_finite_float(
                    row[4],
                    column="end_time",
                ),
                attributes=_json_mapping(
                    row[5],
                    column="SpanAttributes",
                ),
            )
            for row in rows
        ]


def _query(
    client: ClickHouseQueryClient,
    sql: str,
    params: Mapping[str, object],
) -> Sequence[Sequence[object]]:
    result = client.query(sql, parameters=params)
    if not isinstance(result, ClickHouseQueryResult):
        raise TypeError("ClickHouseQueryClient.query() returned an invalid result")
    return list(result.result_rows)


def _json_mapping(value: object, *, column: str) -> dict[str, object]:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(
                f"ClickHouse column {column!r} contains a non-string key"
            )
        return {str(key): item for key, item in value.items()}
    if isinstance(value, (str, bytes, bytearray)):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"ClickHouse column {column!r} is not valid JSON"
            ) from exc
        if isinstance(parsed, Mapping):
            if not all(isinstance(key, str) for key in parsed):
                raise ValueError(
                    f"ClickHouse column {column!r} contains a non-string key"
                )
            return {str(key): item for key, item in parsed.items()}
    raise ValueError(
        f"ClickHouse column {column!r} must contain a JSON object"
    )


def _body_mapping(value: object) -> Mapping[str, object]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return _json_mapping(value, column="Body")
    if isinstance(value, (str, bytes, bytearray)):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            text = (
                bytes(value).decode("utf-8")
                if isinstance(value, (bytes, bytearray))
                else value
            )
            return {"body": text}
        if isinstance(parsed, Mapping):
            return _json_mapping(parsed, column="Body")
        return {"body": parsed}
    return {"body": value}


def _required_str(value: object, *, column: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"ClickHouse column {column!r} must contain a non-empty string"
        )
    return value


def _optional_str(value: object, *, column: str) -> str | None:
    if value is None or value == "":
        return None
    return _required_str(value, column=column)


def _finite_float(
    value: object,
    *,
    column: str,
    default: float | None = None,
) -> float:
    if value is None:
        if default is None:
            raise ValueError(
                f"ClickHouse column {column!r} must contain a number"
            )
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"ClickHouse column {column!r} must contain a number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"ClickHouse column {column!r} must contain a finite number"
        )
    return number


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{label} is not a valid SQL identifier: {value!r}"
        )
    return value


__all__ = [
    "ClickHouseObservabilityQueryStore",
    "ClickHouseQueryClient",
    "ClickHouseQueryResult",
]
