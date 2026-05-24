"""Fail-stop tests for ``core.runtime.otel_export``.

These guard PR-A of the single-event-log migration: the on-disk wire format
must be valid OTLP ``ResourceSpans`` / ``ResourceLogs`` ndjson, the writer
must not lose data under backpressure, and shutdown must be idempotent so
``atexit`` and explicit teardown can both run safely.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agentm.core.runtime.otel_export import (
    FileLogExporter,
    FileSpanExporter,
    SessionTelemetry,
    setup_session_telemetry,
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
        with telemetry.tracer.start_as_current_span("roundtrip.span"):
            pass
        from opentelemetry._logs import SeverityNumber

        telemetry.logger.emit(
            body="roundtrip body",
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
        )
    finally:
        telemetry.shutdown()

    assert telemetry.file_path.exists()
    lines = _read_lines(telemetry.file_path)
    assert len(lines) >= 2, f"expected >=2 lines (span + log), got {len(lines)}: {lines}"

    has_span = False
    has_log = False
    for line in lines:
        if "scopeSpans" in line:
            has_span = True
            attrs = _resource_attrs(line)
            assert attrs.get("agentm.session.id") == "sess-roundtrip"
            assert attrs.get("agentm.scenario.name") == "unit"
        if "scopeLogs" in line:
            has_log = True
    assert has_span and has_log


def test_session_telemetry_shutdown_is_idempotent(tmp_path: Path) -> None:
    telemetry = setup_session_telemetry(session_id="sess-idem", cwd=tmp_path)
    telemetry.shutdown()
    telemetry.shutdown()  # no exception


def test_concurrent_session_telemetry_to_separate_files_dont_interleave(
    tmp_path: Path,
) -> None:
    a = setup_session_telemetry(session_id="sess-a", cwd=tmp_path)
    b = setup_session_telemetry(session_id="sess-b", cwd=tmp_path)
    try:

        def emit(t: SessionTelemetry, name: str) -> None:
            for i in range(20):
                with t.tracer.start_as_current_span(f"{name}.{i}"):
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
            sid = _resource_attrs(line).get("agentm.session.id")
            if sid:
                ids.add(sid)
        return ids

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
            with telemetry.tracer.start_as_current_span(f"bp.{i}"):
                pass
    finally:
        telemetry.shutdown()

    lines = _read_lines(telemetry.file_path)
    total_spans = 0
    for line in lines:
        for scope in line.get("scopeSpans", []):
            total_spans += len(scope.get("spans", []))
    assert total_spans == 1000, f"expected 1000 spans landed, got {total_spans}"


