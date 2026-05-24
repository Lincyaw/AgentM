"""File-backed OTLP exporters for spans and logs.

Writes one OTLP ``ResourceSpans`` / ``ResourceLogs`` element per line as
ndjson at ``<cwd>/.agentm/observability/<session_id>.jsonl``. Each line is a
self-contained JSON object that would be a valid element inside
``ExportTraceServiceRequest.resource_spans[]`` or
``ExportLogsServiceRequest.resource_logs[]`` — collector pipelines (e.g.
``filelog`` + ``otlpjson`` decoders) consume this directly. Lines do **not**
carry the outer ``{"resourceSpans": [...]}`` wrap; that envelope only belongs
on the wire when batch-shipping to a collector.

The canonical OTLP-shape JSON is produced by feeding ReadableSpans /
LogRecords through the standard ``opentelemetry-exporter-otlp`` proto encoders
and converting the resulting protobuf message via ``MessageToDict``. This is
the same path the collector exporters use, so the field names
(``traceId``, ``startTimeUnixNano``, ...) match the OTLP/JSON spec exactly.

This module is PR-A of the single-event-log migration: the exporters and a
``setup_session_telemetry`` helper that wires both into per-session
Tracer/Logger providers. Nothing in the runtime calls into this module yet —
observability rewiring lands in PR-B.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Sequence

from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry._logs import Logger
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer

__all__ = [
    "BlockingBatchLogRecordProcessor",
    "BlockingBatchSpanProcessor",
    "FileLogExporter",
    "FileSpanExporter",
    "SessionTelemetry",
    "setup_session_telemetry",
]


# Once the single event log is authoritative for conversation state, dropping
# events on queue overflow would lose data. The stock BatchSpanProcessor /
# BatchLogRecordProcessor evict from a bounded deque on overflow. These
# subclasses spin-wait on queue length before delegating, converting drop into
# bounded backpressure. The queue is sized generously by default so the wait
# loop is exercised only under genuine producer/consumer mismatch.
_BACKPRESSURE_POLL_SECONDS = 0.001


def _wait_for_queue_space(batch_processor: object) -> None:
    # ``BatchSpanProcessor`` / ``BatchLogRecordProcessor`` expose an internal
    # ``_batch_processor`` with ``_queue`` (a ``collections.deque``) and
    # ``_max_queue_size``. Reach in deliberately — the alternative is a full
    # reimplementation of batching, which is far larger surface than this
    # spin-wait.
    inner = getattr(batch_processor, "_batch_processor", None)
    if inner is None:
        return
    queue = getattr(inner, "_queue", None)
    max_size = getattr(inner, "_max_queue_size", None)
    if queue is None or max_size is None:
        return
    while len(queue) >= max_size:
        time.sleep(_BACKPRESSURE_POLL_SECONDS)


class BlockingBatchSpanProcessor(BatchSpanProcessor):
    """:class:`BatchSpanProcessor` that blocks the producer on overflow."""

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        _wait_for_queue_space(self)
        super().on_end(span)


class BlockingBatchLogRecordProcessor(BatchLogRecordProcessor):
    """:class:`BatchLogRecordProcessor` that blocks the producer on overflow."""

    def on_emit(self, log_record):  # type: ignore[override, no-untyped-def]
        _wait_for_queue_space(self)
        super().on_emit(log_record)


_SCOPE_NAME = "agentm"


def _agentm_version() -> str:
    try:
        return version("agentm")
    except PackageNotFoundError:  # pragma: no cover — dev-checkout edge case
        return "0.0.0"


def _open_append(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


class FileSpanExporter(SpanExporter):
    """Writes OTLP ``ResourceSpans`` elements as ndjson, one per line.

    Each call to :meth:`export` serializes the batch through the standard
    OTLP proto encoder, then splits ``ExportTraceServiceRequest.resource_spans``
    into individual lines. The file handle is held open for the lifetime of
    the exporter and protected by a lock so concurrent batch-processor
    threads do not interleave bytes.
    """

    def __init__(self, file_path: Path) -> None:
        self._path = Path(file_path)
        self._lock = threading.Lock()
        self._fh = _open_append(self._path)
        self._closed = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        try:
            request = encode_spans(spans)
            payload = MessageToDict(
                request,
                preserving_proto_field_name=False,
                use_integers_for_enums=False,
            )
        except Exception:  # pragma: no cover — encoder bugs are unexpected
            return SpanExportResult.FAILURE
        lines = [
            json.dumps(rs, separators=(",", ":"), ensure_ascii=False)
            for rs in payload.get("resourceSpans", [])
        ]
        if not lines:
            return SpanExportResult.SUCCESS
        with self._lock:
            if self._closed:
                return SpanExportResult.FAILURE
            for line in lines:
                self._fh.write(line)
                self._fh.write("\n")
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except (OSError, ValueError):  # pragma: no cover — pipe / closed fd
                pass
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        with self._lock:
            if self._closed:
                return True
            self._fh.flush()
        return True

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._closed = True


class FileLogExporter(LogRecordExporter):
    """Writes OTLP ``ResourceLogs`` elements as ndjson, one per line.

    Symmetric to :class:`FileSpanExporter` — same on-disk wire shape, just
    the log path of the OTLP schema.
    """

    def __init__(self, file_path: Path) -> None:
        self._path = Path(file_path)
        self._lock = threading.Lock()
        self._fh = _open_append(self._path)
        self._closed = False

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        if not batch:
            return LogRecordExportResult.SUCCESS
        try:
            request = encode_logs(batch)
            payload = MessageToDict(
                request,
                preserving_proto_field_name=False,
                use_integers_for_enums=False,
            )
        except Exception:  # pragma: no cover
            return LogRecordExportResult.FAILURE
        lines = [
            json.dumps(rl, separators=(",", ":"), ensure_ascii=False)
            for rl in payload.get("resourceLogs", [])
        ]
        if not lines:
            return LogRecordExportResult.SUCCESS
        with self._lock:
            if self._closed:
                return LogRecordExportResult.FAILURE
            for line in lines:
                self._fh.write(line)
                self._fh.write("\n")
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except (OSError, ValueError):  # pragma: no cover
                pass
        return LogRecordExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        with self._lock:
            if self._closed:
                return True
            self._fh.flush()
        return True

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._closed = True


@dataclass(slots=True)
class SessionTelemetry:
    """Handle bundling per-session telemetry plumbing.

    A session owns its own :class:`TracerProvider` and :class:`LoggerProvider`
    so that resource attributes (notably ``agentm.session.id``) are scoped to
    the session, not process-global. Both providers write into the same
    ndjson file; lines from spans and logs interleave in arrival order.

    :meth:`shutdown` flushes both batch processors and closes both exporters.
    Idempotent — safe to call from ``atexit`` and explicit teardown.
    """

    session_id: str
    file_path: Path
    tracer: Tracer
    logger: Logger
    tracer_provider: TracerProvider
    logger_provider: LoggerProvider
    span_exporter: FileSpanExporter
    log_exporter: FileLogExporter
    _shutdown: bool = False

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        # Order matters: drain processors before closing the exporters they
        # write through. ``shutdown`` on the providers force-flushes and
        # tears down the batch processors.
        try:
            self.tracer_provider.shutdown()
        except Exception:  # pragma: no cover — provider rarely fails
            pass
        try:
            self.logger_provider.shutdown()
        except Exception:  # pragma: no cover
            pass
        # BatchSpanProcessor.shutdown calls exporter.shutdown; calling again
        # is fine because the exporters guard with ``self._closed``.
        self.span_exporter.shutdown()
        self.log_exporter.shutdown()


def setup_session_telemetry(
    session_id: str,
    cwd: Path,
    scenario_name: str | None = None,
    *,
    max_queue_size: int = 100_000,
    schedule_delay_millis: int = 200,
    export_timeout_millis: int = 30_000,
    max_export_batch_size: int | None = None,
) -> SessionTelemetry:
    """Build a per-session telemetry handle.

    The output path is ``<cwd>/.agentm/observability/<session_id>.jsonl``.
    The same file holds both spans and logs interleaved — the OTLP shape
    per line is what disambiguates them on read.

    BatchSpanProcessor / BatchLogRecordProcessor are configured to **block**
    on queue overflow (the default behavior in opentelemetry-sdk when
    ``max_queue_size`` is hit is to drop; we set queue size large enough
    that overflow is rare, and tests assert no drops at this size).
    """
    resource_attrs: dict[str, str] = {
        "service.name": "agentm",
        "service.version": _agentm_version(),
        "agentm.session.id": session_id,
    }
    if scenario_name is not None:
        resource_attrs["agentm.scenario.name"] = scenario_name
    resource = Resource.create(resource_attrs)

    file_path = Path(cwd) / ".agentm" / "observability" / f"{session_id}.jsonl"

    span_exporter = FileSpanExporter(file_path)
    log_exporter = FileLogExporter(file_path)

    effective_batch_size = (
        max_export_batch_size
        if max_export_batch_size is not None
        else min(512, max_queue_size)
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BlockingBatchSpanProcessor(
            span_exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_millis,
            export_timeout_millis=export_timeout_millis,
            max_export_batch_size=effective_batch_size,
        )
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BlockingBatchLogRecordProcessor(
            log_exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_millis,
            export_timeout_millis=export_timeout_millis,
            max_export_batch_size=effective_batch_size,
        )
    )

    tracer = tracer_provider.get_tracer(_SCOPE_NAME, _agentm_version())
    logger = logger_provider.get_logger(_SCOPE_NAME, _agentm_version())

    return SessionTelemetry(
        session_id=session_id,
        file_path=file_path,
        tracer=tracer,
        logger=logger,
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        span_exporter=span_exporter,
        log_exporter=log_exporter,
    )


