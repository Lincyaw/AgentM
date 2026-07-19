"""Read-side query store over local OTLP/JSONL observability files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agentm.core.abi.query import (
    EventRecord,
    QueryMeta,
    SessionFilter,
    SessionIdentity,
    SpanRecord,
    TrajectoryQueryStore,
)
from agentm.core.abi.trajectory import Turn
from agentm.extensions.observability.otlp import (
    iter_log_records,
    iter_spans,
    otlp_unwrap,
)


class OtlpJsonlQueryStore:
    """Query events/spans from OTLP JSONL files and delegate trajectory rows."""

    def __init__(
        self,
        *,
        root: str | Path,
        trajectory: TrajectoryQueryStore | None = None,
    ) -> None:
        self._root = Path(root)
        self._trajectory = trajectory

    def sessions(
        self,
        filter: SessionFilter | None = None,
    ) -> Iterable[SessionIdentity]:
        if self._trajectory is None:
            return []
        return self._trajectory.sessions(filter)

    def turns(self, session_id: str) -> Iterable[Turn]:
        if self._trajectory is None:
            return []
        return self._trajectory.turns(session_id)

    def events(self, session_id: str) -> Iterable[EventRecord]:
        return [
            record
            for record in self._events_from_file(self._path(session_id))
            if record.session_id == session_id
        ]

    def spans(self, session_id: str) -> Iterable[SpanRecord]:
        return [
            record
            for record in self._spans_from_file(self._path(session_id))
            if record.session_id == session_id
        ]

    def _events_from_file(self, path: Path) -> list[EventRecord]:
        records: list[EventRecord] = []
        for line in _read_jsonl(path):
            for record in iter_log_records(line):
                attrs = _attributes(record.get("attributes", ()))
                session_id = _session_id(attrs)
                if session_id is None:
                    continue
                body = otlp_unwrap(record.get("body"))
                name = str(attrs.get("agentm.event.channel") or record.get("severityText") or "event")
                records.append(
                    EventRecord(
                        session_id=session_id,
                        name=name,
                        timestamp=_time_unix_nano(record.get("timeUnixNano")),
                        payload=body if isinstance(body, dict) else {"body": body},
                        meta=QueryMeta(attributes=attrs),
                    )
                )
        return records

    def _spans_from_file(self, path: Path) -> list[SpanRecord]:
        records: list[SpanRecord] = []
        for line in _read_jsonl(path):
            for span in iter_spans(line):
                attrs = _attributes(span.get("attributes", ()))
                session_id = _session_id(attrs)
                if session_id is None:
                    continue
                records.append(
                    SpanRecord(
                        session_id=session_id,
                        name=str(span.get("name") or "span"),
                        span_id=str(span.get("spanId") or ""),
                        parent_span_id=_optional_str(span.get("parentSpanId")),
                        start_time=_time_unix_nano(span.get("startTimeUnixNano")),
                        end_time=_time_unix_nano_or_none(span.get("endTimeUnixNano")),
                        attributes=attrs,
                        meta=QueryMeta(attributes={"trace_id": span.get("traceId") or ""}),
                    )
                )
        return records

    def _path(self, session_id: str) -> Path:
        return self._root / f"{session_id}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _attributes(raw: object) -> dict[str, object]:
    attrs: dict[str, object] = {}
    if not isinstance(raw, list):
        return attrs
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if isinstance(key, str):
            attrs[key] = otlp_unwrap(item.get("value"))
    return attrs


def _session_id(attrs: dict[str, object]) -> str | None:
    value = attrs.get("agentm.session.id")
    return value if isinstance(value, str) else None


def _time_unix_nano(value: object) -> float:
    result = _time_unix_nano_or_none(value)
    return 0.0 if result is None else result


def _time_unix_nano_or_none(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return int(value) / 1_000_000_000
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["OtlpJsonlQueryStore"]
