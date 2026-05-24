"""Fail-stop tests for ``core.runtime.otel_export``.

These guard PR-A of the single-event-log migration and PR-H's process-level
provider refactor: the on-disk wire format must be valid OTLP
``ResourceSpans`` / ``ResourceLogs`` ndjson, the writer must not lose data
under backpressure, shutdown must be idempotent so the per-session
``SessionShutdownEvent`` handler and explicit teardown can both run safely,
and concurrent sessions in one process must share a ``TracerProvider`` /
``LoggerProvider`` while writing to disjoint files.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agentm.core.runtime.otel_export import (
    FileLogExporter,
    FileSpanExporter,
    SessionTelemetry,
    setup_process_telemetry,
    setup_session_telemetry,
    shutdown_process_telemetry,
)


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resource_attrs(record: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for attr in record.get("resource", {}).get("attributes", []):
        key = attr["key"]
        value_wrapper = attr["value"]
        # OTLP attribute values are tagged unions; we only care about strings here.
        if "stringValue" in value_wrapper:
            out[key] = value_wrapper["stringValue"]
    return out


def _span_attr(span: dict, key: str) -> str | None:
    for attr in span.get("attributes", []) or []:
        if attr.get("key") == key:
            v = attr.get("value") or {}
            if "stringValue" in v:
                return v["stringValue"]
    return None


def _log_attr(record: dict, key: str) -> str | None:
    for attr in record.get("attributes", []) or []:
        if attr.get("key") == key:
            v = attr.get("value") or {}
            if "stringValue" in v:
                return v["stringValue"]
    return None


def test_file_span_exporter_writes_otlp_resource_spans_shape(tmp_path: Path) -> None:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    file_path = tmp_path / "spans.jsonl"
    exporter = FileSpanExporter(file_path)
    try:
        provider = TracerProvider(
            resource=Resource.create({"service.name": "agentm", "service.version": "0.1.0"}),
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("agentm", "0.1.0")
        with tracer.start_as_current_span("unit.test") as span:
            span.set_attribute("k", "v")
        provider.shutdown()
    finally:
        exporter.shutdown()

    lines = _read_lines(file_path)
    assert len(lines) == 1, "exactly one ResourceSpans element expected"
    rs = lines[0]
    attrs = _resource_attrs(rs)
    assert attrs.get("service.name") == "agentm"

    scope_spans = rs["scopeSpans"]
    assert len(scope_spans) >= 1
    spans = scope_spans[0]["spans"]
    assert len(spans) == 1
    s = spans[0]
    for required in ("name", "traceId", "spanId", "startTimeUnixNano", "endTimeUnixNano"):
        assert required in s, f"missing OTLP field {required}: {s}"
    assert s["name"] == "unit.test"


def test_file_log_exporter_writes_otlp_resource_logs_shape(tmp_path: Path) -> None:
    import logging

    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    file_path = tmp_path / "logs.jsonl"
    exporter = FileLogExporter(file_path)
    try:
        provider = LoggerProvider(
            resource=Resource.create({"service.name": "agentm", "service.version": "0.1.0"}),
        )
        provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
        py_logger = logging.getLogger("agentm.test.logs")
        py_logger.setLevel(logging.INFO)
        py_logger.addHandler(handler)
        py_logger.info("hello from agentm")
        provider.shutdown()
        py_logger.removeHandler(handler)
    finally:
        exporter.shutdown()

    lines = _read_lines(file_path)
    assert len(lines) >= 1
    rl = lines[0]
    attrs = _resource_attrs(rl)
    assert attrs.get("service.name") == "agentm"

    scope_logs = rl["scopeLogs"]
    assert len(scope_logs) >= 1
    records = scope_logs[0]["logRecords"]
    assert len(records) >= 1
    record = records[0]
    assert "timeUnixNano" in record
    assert record["body"]["stringValue"] == "hello from agentm"


def test_setup_session_telemetry_round_trips_through_file(tmp_path: Path) -> None:
    telemetry = setup_session_telemetry(
        session_id="sess-roundtrip",
        cwd=tmp_path,
        scenario_name="unit",
    )
    try:
        with telemetry.tracer.start_as_current_span(
            "roundtrip.span",
            attributes={"agentm.session.id": telemetry.session_id},
        ):
            pass
        from opentelemetry._logs import SeverityNumber

        telemetry.logger.emit(
            body="roundtrip body",
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            attributes={"agentm.session.id": telemetry.session_id},
        )
    finally:
        telemetry.shutdown()

    assert telemetry.file_path.exists()
    lines = _read_lines(telemetry.file_path)
    assert len(lines) >= 2, f"expected >=2 lines (span + log), got {len(lines)}: {lines}"

    has_span = False
    has_log = False
    for line in lines:
        attrs = _resource_attrs(line)
        # PR-H: agentm.session.id must NOT live on the resource any more.
        assert "agentm.session.id" not in attrs, (
            "agentm.session.id leaked into resource attributes; per-record only"
        )
        assert "agentm.scenario.name" not in attrs, (
            "agentm.scenario.name leaked into resource attributes; "
            "atoms emit it as a span/log attribute instead"
        )
        # service.name/version stay on the resource (process-scoped).
        assert attrs.get("service.name") == "agentm"
        if "scopeSpans" in line:
            has_span = True
            for scope in line.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    assert _span_attr(span, "agentm.session.id") == "sess-roundtrip"
        if "scopeLogs" in line:
            has_log = True
            for scope in line.get("scopeLogs", []):
                for record in scope.get("logRecords", []):
                    assert _log_attr(record, "agentm.session.id") == "sess-roundtrip"
    assert has_span and has_log


def test_session_telemetry_shutdown_is_idempotent(tmp_path: Path) -> None:
    telemetry = setup_session_telemetry(session_id="sess-idem", cwd=tmp_path)
    telemetry.shutdown()
    telemetry.shutdown()  # no exception


def test_two_sessions_share_one_tracer_provider(tmp_path: Path) -> None:
    """PR-H: process-level providers are shared across concurrent sessions."""
    a = setup_session_telemetry(session_id="sess-share-a", cwd=tmp_path)
    b = setup_session_telemetry(session_id="sess-share-b", cwd=tmp_path)
    try:
        assert a.tracer_provider is b.tracer_provider
        assert a.logger_provider is b.logger_provider
        # And both line up with what setup_process_telemetry() returns.
        tp, lp = setup_process_telemetry()
        assert tp is a.tracer_provider
        assert lp is a.logger_provider
    finally:
        a.shutdown()
        b.shutdown()


def test_session_shutdown_does_not_affect_other_session(tmp_path: Path) -> None:
    """PR-H: shutting one session down leaves the other's writers intact."""
    a = setup_session_telemetry(session_id="sess-iso-a", cwd=tmp_path)
    b = setup_session_telemetry(session_id="sess-iso-b", cwd=tmp_path)
    try:
        with a.tracer.start_as_current_span(
            "a.before", attributes={"agentm.session.id": a.session_id}
        ):
            pass
        with b.tracer.start_as_current_span(
            "b.before", attributes={"agentm.session.id": b.session_id}
        ):
            pass
        a.shutdown()
        # ``b`` must still accept writes.
        with b.tracer.start_as_current_span(
            "b.after", attributes={"agentm.session.id": b.session_id}
        ):
            pass
    finally:
        b.shutdown()

    a_lines = _read_lines(a.file_path)
    b_lines = _read_lines(b.file_path)
    a_names = [
        s.get("name")
        for line in a_lines
        for scope in line.get("scopeSpans", [])
        for s in scope.get("spans", [])
    ]
    b_names = [
        s.get("name")
        for line in b_lines
        for scope in line.get("scopeSpans", [])
        for s in scope.get("spans", [])
    ]
    assert a_names == ["a.before"]
    assert b_names == ["b.before", "b.after"]


def test_concurrent_session_telemetry_to_separate_files_dont_interleave(
    tmp_path: Path,
) -> None:
    a = setup_session_telemetry(session_id="sess-a", cwd=tmp_path)
    b = setup_session_telemetry(session_id="sess-b", cwd=tmp_path)
    try:

        def emit(t: SessionTelemetry, name: str) -> None:
            for i in range(20):
                with t.tracer.start_as_current_span(
                    f"{name}.{i}",
                    attributes={"agentm.session.id": t.session_id},
                ):
                    pass

        ta = threading.Thread(target=emit, args=(a, "a"))
        tb = threading.Thread(target=emit, args=(b, "b"))
        ta.start()
        tb.start()
        ta.join()
        tb.join()
    finally:
        a.shutdown()
        b.shutdown()

    a_lines = _read_lines(a.file_path)
    b_lines = _read_lines(b.file_path)

    def session_ids(lines: list[dict]) -> set[str]:
        ids: set[str] = set()
        for line in lines:
            for scope in line.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    sid = _span_attr(span, "agentm.session.id")
                    if sid:
                        ids.add(sid)
        return ids

    # Files are file-scoped (each session's processor only writes its file)
    # AND the per-record attribute carries the same session id.
    assert session_ids(a_lines) == {"sess-a"}
    assert session_ids(b_lines) == {"sess-b"}


def test_blocking_backpressure(tmp_path: Path) -> None:
    # Tight queue: any drop would lose conversation state once this is wired
    # into the single event log. Our blocking processor must spin until space
    # exists rather than evict.
    telemetry = setup_session_telemetry(
        session_id="sess-bp",
        cwd=tmp_path,
        max_queue_size=100,
        schedule_delay_millis=10,
    )
    try:
        for i in range(1000):
            with telemetry.tracer.start_as_current_span(
                f"bp.{i}",
                attributes={"agentm.session.id": telemetry.session_id},
            ):
                pass
    finally:
        telemetry.shutdown()

    lines = _read_lines(telemetry.file_path)
    total_spans = 0
    for line in lines:
        for scope in line.get("scopeSpans", []):
            total_spans += len(scope.get("spans", []))
    assert total_spans == 1000, f"expected 1000 spans landed, got {total_spans}"


def test_shutdown_process_telemetry_is_idempotent(tmp_path: Path) -> None:
    """Manual process-level teardown is safe to invoke repeatedly."""
    telemetry = setup_session_telemetry(session_id="sess-proc", cwd=tmp_path)
    telemetry.shutdown()
    shutdown_process_telemetry()
    shutdown_process_telemetry()  # no exception
    # A fresh session after explicit shutdown must spin up a new global pair.
    tp1, lp1 = setup_process_telemetry()
    tp2, lp2 = setup_process_telemetry()
    assert tp1 is tp2 and lp1 is lp2
    shutdown_process_telemetry()
