"""Canonical reader for the OTLP/JSON per-session event log.

This module is the **single seam** for reading the on-disk OTLP/JSON ndjson
event log written by :mod:`agentm.core.runtime.otel_export`. Every consumer
that previously hand-rolled "open jsonl, parse, walk ``scopeSpans`` /
``scopeLogs``, unwrap kvlist attributes" should depend on :class:`TraceReader`
instead.

The wire format is documented in ``.claude/designs/single-event-log.md``: each
line is a self-contained OTLP/JSON element (either a ``resourceSpans`` or a
``resourceLogs`` wrapper). Span lines and log lines interleave in arrival
order. Attribute values are OTLP tagged unions
(``{"stringValue": ...}``, ``{"kvlistValue": ...}``, ...); this reader
unwraps them to plain Python values so callers see ordinary ``dict`` /
``list`` / scalar types.

The reader is **lazy** — every iterator opens the file, walks line-by-line,
and yields. It never loads the whole file into memory. Convenience methods
that *must* return a list (e.g. :meth:`load_messages` because order +
completeness matters) are explicit about that.

Atoms cannot import this module directly (§11). They reach :class:`TraceReader`
through the re-export in :mod:`agentm.core.abi`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from agentm.core.observability.otlp import (
    iter_log_records as _iter_log_records_on_line,
    iter_spans as _iter_spans_on_line,
    otlp_unwrap,
)

__all__ = [
    "LogRecord",
    "Span",
    "SessionIdentity",
    "TraceReader",
    "attr",
]


def _unwrap_attributes(raw: Any) -> dict[str, Any]:
    """Convert an OTLP ``attributes`` list to a plain ``{key: value}`` dict.

    OTLP attributes are encoded as a list of ``{"key", "value": <tagged-union>}``
    pairs. This helper materialises them to a dict with values already passed
    through :func:`otlp_unwrap` so callers do one lookup, not a linear scan
    plus tag inspection.
    """
    out: dict[str, Any] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str):
            continue
        out[key] = otlp_unwrap(item.get("value"))
    return out


def _parse_ns(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True, slots=True)
class Span:
    """Plain-Python view of one OTLP span line.

    Attribute values in :attr:`attributes` are already unwrapped from their
    OTLP tagged-union form — strings come out as ``str``, kvlists as
    ``dict``, etc. The raw span dict is preserved in :attr:`raw` for callers
    that need to inspect fields the dataclass does not surface.
    """

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    kind: str | None = None
    start_time_unix_nano: int | None = None
    end_time_unix_nano: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LogRecord:
    """Plain-Python view of one OTLP log record.

    Both :attr:`attributes` and :attr:`body` are unwrapped: a kvlist body
    becomes a ``dict``; a string body becomes a ``str``; etc. The raw
    record is in :attr:`raw` for fields not surfaced here.
    """

    event_name: str
    body: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None
    severity_number: int | None = None
    severity_text: str | None = None
    time_unix_nano: int | None = None
    observed_time_unix_nano: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    """Topology identity of one session file, from its ``session.start`` record.

    These are the fields that stitch many JSONL session files into a single
    logical trace tree: ``trace_id`` is the OTel root/trace id shared across
    the parent session and every spawned child; ``parent_session_id`` is the
    edge back to the spawner (``None`` for the root); ``purpose`` distinguishes
    the role (e.g. ``root`` / ``cognitive_audit_extractor`` /
    ``cognitive_audit_auditor``). All values are pulled from the
    ``agentm.session.start`` log record's body, falling back to the
    OTel-prefixed attribute keys.
    """

    trace_id: str | None = None
    session_id: str | None = None
    parent_session_id: str | None = None
    purpose: str | None = None
    scenario: str | None = None


def _span_from_raw(raw: dict[str, Any]) -> Span:
    return Span(
        name=str(raw.get("name") or ""),
        attributes=_unwrap_attributes(raw.get("attributes")),
        trace_id=raw.get("traceId") if isinstance(raw.get("traceId"), str) else None,
        span_id=raw.get("spanId") if isinstance(raw.get("spanId"), str) else None,
        parent_span_id=(
            raw.get("parentSpanId")
            if isinstance(raw.get("parentSpanId"), str)
            else None
        ),
        kind=raw.get("kind") if isinstance(raw.get("kind"), str) else None,
        start_time_unix_nano=_parse_ns(raw.get("startTimeUnixNano")),
        end_time_unix_nano=_parse_ns(raw.get("endTimeUnixNano")),
        raw=raw,
    )


def _log_record_from_raw(raw: dict[str, Any]) -> LogRecord:
    severity_raw = raw.get("severityNumber")
    severity_number: int | None
    if isinstance(severity_raw, int):
        severity_number = severity_raw
    elif isinstance(severity_raw, str):
        try:
            severity_number = int(severity_raw)
        except (TypeError, ValueError):
            severity_number = None
    else:
        severity_number = None
    return LogRecord(
        event_name=str(raw.get("eventName") or ""),
        body=otlp_unwrap(raw.get("body")),
        attributes=_unwrap_attributes(raw.get("attributes")),
        trace_id=raw.get("traceId") if isinstance(raw.get("traceId"), str) else None,
        span_id=raw.get("spanId") if isinstance(raw.get("spanId"), str) else None,
        severity_number=severity_number,
        severity_text=(
            raw.get("severityText")
            if isinstance(raw.get("severityText"), str)
            else None
        ),
        time_unix_nano=_parse_ns(raw.get("timeUnixNano")),
        observed_time_unix_nano=_parse_ns(raw.get("observedTimeUnixNano")),
        raw=raw,
    )


def _matches_filter(
    attributes: dict[str, Any], attribute_filters: dict[str, Any] | None
) -> bool:
    if not attribute_filters:
        return True
    for key, expected in attribute_filters.items():
        if attributes.get(key) != expected:
            return False
    return True


def attr(obj: Span | LogRecord, key: str, default: Any = None) -> Any:
    """Terse attribute lookup on a :class:`Span` or :class:`LogRecord`.

    Equivalent to ``obj.attributes.get(key, default)`` — kept for symmetry
    with the hand-rolled ``span_attr`` helper in :mod:`otel_export` so
    callers porting from the low-level API have a one-line replacement.
    """
    return obj.attributes.get(key, default)


class TraceReader:
    """Read-only view of one OTLP/JSON session event log file.

    Construction is cheap — the file is opened on demand by each iterator.
    Iterators are single-shot generators; call them again to re-walk the
    file (the on-disk log is append-only, so a second walk picks up any
    new lines).

    Order: lines are yielded in file order, which is approximately wall-clock
    arrival order across both spans and logs. Within one OTLP element the
    inner ``scopeSpans[*].spans[*]`` / ``scopeLogs[*].logRecords[*]`` walk
    preserves the encoder's natural order.
    """

    def __init__(self, file_path: Path | str) -> None:
        self._path = Path(file_path)

    @property
    def file_path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Line-level iteration
    # ------------------------------------------------------------------

    def _iter_lines(self) -> Iterator[dict[str, Any]]:
        """Yield each non-empty parsed JSON line. Skips malformed lines
        silently — the writer is append-only so a partial last line can
        appear during a crash; downstream consumers should not crash on
        that.
        """
        try:
            handle = self._path.open("r", encoding="utf-8")
        except OSError:
            return
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, dict):
                    yield raw

    # ------------------------------------------------------------------
    # Generic span / log iteration
    # ------------------------------------------------------------------

    def iter_spans(
        self,
        name: str | None = None,
        attribute_filters: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Iterator[Span]:
        """Yield every span matching ``name`` and ``attribute_filters``.

        ``name`` matches exactly. ``attribute_filters`` is a dict of
        ``{attribute_key: expected_value}`` — every key must match. Pass
        ``None`` to disable a filter.

        ``session_id`` is a convenience filter that matches the
        ``agentm.session.id`` span attribute (PR-H: this lives on the
        span, not the OTel resource, so a single process can host many
        sessions sharing one ``TracerProvider``). Per-file scope is the
        default — every reader is already pointed at one session file —
        so an explicit ``session_id`` is only useful for merged or
        cross-session files.
        """
        merged_filters: dict[str, Any] | None
        if session_id is not None:
            merged_filters = dict(attribute_filters or {})
            merged_filters["agentm.session.id"] = session_id
        else:
            merged_filters = attribute_filters
        for line in self._iter_lines():
            for raw in _iter_spans_on_line(line):
                if name is not None and raw.get("name") != name:
                    continue
                span = _span_from_raw(raw)
                if not _matches_filter(span.attributes, merged_filters):
                    continue
                yield span

    def iter_log_records(
        self,
        name: str | None = None,
        attribute_filters: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[LogRecord]:
        """Yield every log record matching the supplied filters.

        ``name`` matches ``eventName`` exactly. ``parent_span_id`` is a
        convenience filter that matches the record's ``spanId`` (log
        records do not carry a separate parent reference; their ``spanId``
        is the parent span by OTLP convention).

        ``session_id`` matches the ``agentm.session.id`` log-record
        attribute (PR-H — see :meth:`iter_spans` for the rationale).
        """
        merged_filters: dict[str, Any] | None
        if session_id is not None:
            merged_filters = dict(attribute_filters or {})
            merged_filters["agentm.session.id"] = session_id
        else:
            merged_filters = attribute_filters
        for line in self._iter_lines():
            for raw in _iter_log_records_on_line(line):
                if name is not None and raw.get("eventName") != name:
                    continue
                record = _log_record_from_raw(raw)
                if not _matches_filter(record.attributes, merged_filters):
                    continue
                if parent_span_id is not None and record.span_id != parent_span_id:
                    continue
                yield record

    def iter_all(self) -> Iterator[Span | LogRecord]:
        """Yield every span and log record in file order.

        Useful for callers that need to reconstruct chronological history
        across both element kinds (e.g. session replay).
        """
        for line in self._iter_lines():
            for raw in _iter_spans_on_line(line):
                yield _span_from_raw(raw)
            for raw in _iter_log_records_on_line(line):
                yield _log_record_from_raw(raw)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def first_session_identity(self) -> SessionIdentity | None:
        """Return the topology identity from the first ``session.start`` record.

        Reads ONLY the first ``agentm.session.start`` log record (one per
        file, near the top), so it is cheap to call across a whole directory.
        Returns ``None`` when the file carries no such record (partial writes,
        extension-install-only traces from failed spawns, etc.) — callers
        skip those exactly like the trace-map builder does.

        Field provenance: the writer stamps the identity into the record's
        ``body`` as plain keys (``session_id`` / ``root_session_id`` /
        ``parent_session_id`` / ``purpose`` / ``scenario``); the same values
        are mirrored into OTel-prefixed attributes (``agentm.session.root_id``
        and friends). We prefer the body and fall back to attributes so the
        helper stays robust across writer variants. Empty attribute strings
        (the OTLP encoding of ``None``) are normalised back to ``None``.
        """

        def _norm(value: Any) -> str | None:
            if value is None or value == "":
                return None
            return str(value)

        for record in self.iter_log_records(name="agentm.session.start"):
            body = record.body if isinstance(record.body, dict) else {}
            attrs = record.attributes
            return SessionIdentity(
                trace_id=_norm(
                    body.get("root_session_id")
                    or attrs.get("agentm.session.root_id")
                ),
                session_id=_norm(
                    body.get("session_id")
                    or attrs.get("agentm.session.id")
                    or record.span_id
                ),
                parent_session_id=_norm(
                    body.get("parent_session_id")
                    or attrs.get("agentm.session.parent_id")
                ),
                purpose=_norm(
                    body.get("purpose") or attrs.get("agentm.session.purpose")
                ),
                scenario=_norm(
                    body.get("scenario") or attrs.get("agentm.session.scenario")
                ),
            )
        return None

    def load_session_header(self) -> dict[str, Any] | None:
        """Return the body of the latest ``agentm.session.header`` log record.

        Returns ``None`` if no header is present. The body is the
        SessionHeader dict (``id``, ``cwd``, ``parent_session``, ...).
        """
        latest: dict[str, Any] | None = None
        for record in self.iter_log_records(name="agentm.session.header"):
            if isinstance(record.body, dict):
                latest = record.body
        return latest

    def load_session_fingerprint(self) -> dict[str, Any] | None:
        """Return the body of the latest ``agentm.session.fingerprint`` record.

        The body carries the atom hash map and ``task_meta``. ``None`` if
        no fingerprint record is present.
        """
        latest: dict[str, Any] | None = None
        for record in self.iter_log_records(name="agentm.session.fingerprint"):
            if isinstance(record.body, dict):
                latest = record.body
        return latest

    def load_messages(self) -> list[dict[str, Any]]:
        """Return every ``agentm.message.appended`` body in file order.

        Each entry is the SessionEntry dict (``{type, id, parent_id,
        timestamp, payload}``).
        """
        out: list[dict[str, Any]] = []
        for record in self.iter_log_records(name="agentm.message.appended"):
            if isinstance(record.body, dict):
                out.append(record.body)
        return out

    def load_turn_summaries(self) -> list[dict[str, Any]]:
        """Return every ``agentm.turn.summary`` body in file order.

        Each body carries per-turn aggregates: ``tool_calls``,
        ``tool_call_count``, ``tool_error_count``, ``stop_reason``,
        ``input_tokens``, ``output_tokens``, optional ``cost_usd``.
        """
        out: list[dict[str, Any]] = []
        for record in self.iter_log_records(name="agentm.turn.summary"):
            if isinstance(record.body, dict):
                out.append(record.body)
        return out

    def chat_calls(self) -> Iterator[Span]:
        """Yield every ``chat <model>`` span (one per LLM request)."""
        for line in self._iter_lines():
            for raw in _iter_spans_on_line(line):
                name = raw.get("name")
                if isinstance(name, str) and name.startswith("chat "):
                    yield _span_from_raw(raw)

    def tool_calls(
        self,
    ) -> Iterator[tuple[Span, LogRecord | None, LogRecord | None]]:
        """Yield ``(span, args_log, result_log)`` triples for every tool call.

        The span is the ``execute_tool <tool>`` span. The optional log
        records carry the tool-call arguments and result when the writer
        emits them as separate log records (linked by ``spanId``). When
        the writer stamps args / result on the span as attributes only
        (the current writer's default), the two log slots are ``None``.
        """
        # Collect args/result logs keyed by span_id so we can pair them
        # with the spans on a single pass. Memory cost is one dict entry
        # per tool log record; typical sessions emit at most a few hundred.
        args_by_span: dict[str, LogRecord] = {}
        results_by_span: dict[str, LogRecord] = {}
        spans: list[Span] = []
        for item in self.iter_all():
            if isinstance(item, Span):
                if item.name.startswith("execute_tool "):
                    spans.append(item)
            else:
                if item.event_name in {
                    "agentm.tool.arguments",
                    "agentm.tool.call.arguments",
                } and item.span_id is not None:
                    args_by_span[item.span_id] = item
                elif item.event_name in {
                    "agentm.tool.result",
                    "agentm.tool.call.result",
                } and item.span_id is not None:
                    results_by_span[item.span_id] = item
        for span in spans:
            if span.span_id is None:
                yield (span, None, None)
                continue
            yield (
                span,
                args_by_span.get(span.span_id),
                results_by_span.get(span.span_id),
            )
