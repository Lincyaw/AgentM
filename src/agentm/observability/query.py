"""Read-side query store over local OTLP/JSONL observability files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from agentm.core.abi.query import (
    EventRecord,
    QueryMeta,
    SpanRecord,
)
from agentm.observability.otlp import (
    iter_log_records,
    iter_spans,
    otlp_unwrap,
)


class OtlpJsonlQueryStore:
    """Query events and spans from local OTLP JSONL files."""

    def __init__(
        self,
        *,
        root: str | Path,
    ) -> None:
        self._root = Path(root)

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
                name = str(
                    attrs.get("agentm.event.channel")
                    or record.get("severityText")
                    or "event"
                )
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
                        meta=QueryMeta(
                            attributes={"trace_id": span.get("traceId") or ""}
                        ),
                    )
                )
        return records

    def _path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self._root / f"{session_id}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    for line_no, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        complete = raw_line.endswith((b"\n", b"\r"))
        if line_no == len(lines) and not complete:
            break
        try:
            value = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(value)
    return rows


def _validate_session_id(session_id: str) -> None:
    if (
        not session_id
        or session_id in {".", ".."}
        or Path(session_id).name != session_id
        or "\\" in session_id
        or "\x00" in session_id
    ):
        raise ValueError(f"session_id is not a valid path token: {session_id!r}")


def _attributes(raw: object) -> dict[str, object]:
    attrs: dict[str, object] = {}
    if raw in (None, (), []):
        return attrs
    if not isinstance(raw, list):
        raise ValueError("OTLP attributes must be a list")
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("OTLP attribute must be an object")
        key = item.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("OTLP attribute has no key")
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
        raise ValueError("OTLP timestamp must be numeric")
    try:
        return int(value) / 1_000_000_000
    except (TypeError, ValueError) as exc:
        raise ValueError("OTLP timestamp must be numeric") from exc


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["OtlpJsonlQueryStore"]
