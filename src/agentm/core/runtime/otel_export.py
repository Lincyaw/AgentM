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

**Process-level providers (PR-H).** The OTel ecosystem assumes one process =
one ``service.name``, so there is exactly **one** ``TracerProvider`` and one
``LoggerProvider`` per Python process. The resource attached to those
providers carries only ``service.name=agentm`` and ``service.version=<pkg>``.
Per-session isolation is achieved by attaching a **per-session
SpanProcessor + LogRecordProcessor** to the shared globals; each processor
forwards exclusively to its own per-session file exporter, so writes never
cross sessions. ``agentm.session.id`` lives as a span/log **attribute**
stamped by the observability atom on every record it emits — never on the
resource block.

``setup_session_telemetry`` returns a :class:`SessionTelemetry` handle whose
``shutdown()`` drains and removes that session's processors from the global
providers, leaving the providers themselves intact for other concurrent
sessions. Global teardown is deferred to ``atexit``.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Any, Sequence

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
    "iter_log_records",
    "iter_spans",
    "otlp_unwrap",
    "setup_process_telemetry",
    "setup_session_telemetry",
    "shutdown_process_telemetry",
    "span_attr",
]


def otlp_unwrap(value: Any) -> Any:
    """Unwrap an OTLP proto-JSON tagged-union value into a plain Python object.

    OTLP encodes attribute and body values as tagged unions
    (``{"stringValue": ...}``, ``{"intValue": "12"}``,
    ``{"kvlistValue": {"values": [...]}}``, ``{"arrayValue": ...}``).
    Readers (``SessionManager._load``, the catalog indexer, the tuner
    tools) need plain Python types to pattern-match against; this helper
    is the single canonical converter.

    Returns the input unchanged when it isn't a tagged union — useful
    for already-unwrapped intermediate dicts.
    """
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        raw = value["intValue"]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        raw = value["doubleValue"]
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if "kvlistValue" in value:
        out: dict[str, Any] = {}
        for item in value["kvlistValue"].get("values", []) or []:
            key = item.get("key")
            if not isinstance(key, str):
                continue
            out[key] = otlp_unwrap(item.get("value"))
        return out
    if "arrayValue" in value:
        return [otlp_unwrap(v) for v in value["arrayValue"].get("values", []) or []]
    return value


def span_attr(span: dict[str, Any], key: str) -> Any:
    """Look up a single attribute value on an OTLP span / log record dict.

    Returns ``None`` when the key is absent. The OTLP attribute list is a
    sequence of ``{"key", "value": <tagged-union>}`` pairs; we walk it
    linearly because span attribute counts are small (a few dozen at
    most) and a per-span dict cache would cost more than it saves.
    """
    for attr in span.get("attributes", []) or []:
        if attr.get("key") == key:
            return otlp_unwrap(attr.get("value"))
    return None


def iter_spans(line: dict[str, Any]) -> "list[dict[str, Any]]":
    """Iterate spans on one OTLP ``ResourceSpans``-line dict.

    Returns a flat list of span dicts (one per ``scopeSpans[*].spans[*]``).
    The list is empty for lines that aren't ``ResourceSpans`` elements.
    """
    out: list[dict[str, Any]] = []
    for scope in line.get("scopeSpans", []) or []:
        out.extend(scope.get("spans", []) or [])
    return out


def iter_log_records(line: dict[str, Any]) -> "list[dict[str, Any]]":
    """Iterate log records on one OTLP ``ResourceLogs``-line dict."""
    out: list[dict[str, Any]] = []
    for scope in line.get("scopeLogs", []) or []:
        out.extend(scope.get("logRecords", []) or [])
    return out


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
    """:class:`BatchSpanProcessor` that blocks the producer on overflow.

    When ``session_id_filter`` is set, the processor drops spans whose
    ``agentm.session.id`` attribute does not match — this is how
    per-session file partitioning survives a shared process-level
    ``TracerProvider``. Spans without the attribute also drop (the
    observability atom is required to stamp it; bare SDK calls in
    tests can opt in by passing ``agentm.session.id`` in span attributes).
    """

    def __init__(
        self,
        *args: Any,
        session_id_filter: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._session_id_filter = session_id_filter

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        if self._session_id_filter is not None:
            attrs = span.attributes or {}
            if attrs.get("agentm.session.id") != self._session_id_filter:
                return
        _wait_for_queue_space(self)
        super().on_end(span)


class BlockingBatchLogRecordProcessor(BatchLogRecordProcessor):
    """:class:`BatchLogRecordProcessor` that blocks the producer on overflow.

    Symmetric ``session_id_filter`` behaviour — see
    :class:`BlockingBatchSpanProcessor`.
    """

    def __init__(
        self,
        *args: Any,
        session_id_filter: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._session_id_filter = session_id_filter

    def on_emit(self, log_record):  # type: ignore[override, no-untyped-def]
        if self._session_id_filter is not None:
            attrs = getattr(log_record, "log_record", log_record).attributes or {}
            if attrs.get("agentm.session.id") != self._session_id_filter:
                return
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


# --- Process-level providers ------------------------------------------------
#
# The OTel ecosystem assumes one ``service.name`` per process. We honour that
# by holding exactly one ``TracerProvider`` + one ``LoggerProvider`` for the
# whole Python process; per-session isolation happens at the processor layer.
# These globals are protected by ``_global_lock``; first call to
# :func:`setup_process_telemetry` (typically via
# :func:`setup_session_telemetry`) constructs them and registers an
# ``atexit`` hook for final teardown.

_global_lock = threading.Lock()
_global_tracer_provider: TracerProvider | None = None
_global_logger_provider: LoggerProvider | None = None
_global_atexit_registered = False


def setup_process_telemetry() -> tuple[TracerProvider, LoggerProvider]:
    """Lazily construct the process-level OTel providers.

    Idempotent — subsequent calls return the same provider instances. The
    resource attached to each provider carries only ``service.name`` and
    ``service.version``; per-session metadata (``agentm.session.id``,
    ``agentm.scenario.name``) is emitted as **per-span / per-log
    attributes** by the observability atom, not as resource attributes.

    Returns ``(tracer_provider, logger_provider)``.
    """
    global _global_tracer_provider, _global_logger_provider
    global _global_atexit_registered
    with _global_lock:
        if (
            _global_tracer_provider is not None
            and _global_logger_provider is not None
        ):
            return _global_tracer_provider, _global_logger_provider
        resource = Resource.create(
            {
                "service.name": "agentm",
                "service.version": _agentm_version(),
            }
        )
        _global_tracer_provider = TracerProvider(resource=resource)
        _global_logger_provider = LoggerProvider(resource=resource)
        if not _global_atexit_registered:
            atexit.register(shutdown_process_telemetry)
            _global_atexit_registered = True
        return _global_tracer_provider, _global_logger_provider


def shutdown_process_telemetry() -> None:
    """Tear down the process-level providers (idempotent).

    Registered as an ``atexit`` hook on first :func:`setup_process_telemetry`
    call. Safe to invoke explicitly from tests that want a clean slate.
    """
    global _global_tracer_provider, _global_logger_provider
    with _global_lock:
        tp = _global_tracer_provider
        lp = _global_logger_provider
        _global_tracer_provider = None
        _global_logger_provider = None
    if tp is not None:
        try:
            tp.shutdown()
        except Exception:  # pragma: no cover
            pass
    if lp is not None:
        try:
            lp.shutdown()
        except Exception:  # pragma: no cover
            pass


def _remove_span_processor(
    provider: TracerProvider, processor: BlockingBatchSpanProcessor
) -> None:
    """Remove a span processor from a provider's internal multi-processor.

    The ``SynchronousMultiSpanProcessor`` stores its children in a tuple;
    there is no public ``remove`` method. We rebuild the tuple under its
    lock. Best-effort — if the SDK rearranges this private attribute in a
    future release the processor stays attached but its underlying exporter
    is shut down, so it would only log to a closed handle.
    """
    multi = getattr(provider, "_active_span_processor", None)
    if multi is None:
        return
    lock = getattr(multi, "_lock", None)
    span_processors_attr = "_span_processors"
    if lock is None or not hasattr(multi, span_processors_attr):
        return
    with lock:
        current = getattr(multi, span_processors_attr)
        if processor in current:
            setattr(
                multi,
                span_processors_attr,
                tuple(p for p in current if p is not processor),
            )


def _remove_log_processor(
    provider: LoggerProvider, processor: BlockingBatchLogRecordProcessor
) -> None:
    """Symmetric helper for :class:`LoggerProvider`'s multi-processor."""
    multi = getattr(provider, "_multi_log_record_processor", None)
    if multi is None:
        return
    lock = getattr(multi, "_lock", None)
    attr_name = "_log_record_processors"
    if lock is None or not hasattr(multi, attr_name):
        return
    with lock:
        current = getattr(multi, attr_name)
        if isinstance(current, tuple):
            if processor in current:
                setattr(
                    multi,
                    attr_name,
                    tuple(p for p in current if p is not processor),
                )
        elif isinstance(current, list):
            try:
                current.remove(processor)
            except ValueError:
                pass


@dataclass(slots=True)
class SessionTelemetry:
    """Handle bundling per-session telemetry plumbing.

    Each session has its own :class:`BlockingBatchSpanProcessor` +
    :class:`BlockingBatchLogRecordProcessor` attached to the **shared**
    process-level providers. ``tracer`` and ``logger`` are scoped instrument
    objects (scope name ``agentm.session.<id>``) acquired from the global
    providers; spans/logs emitted through them flow through every attached
    processor, but our per-session processor is the only one that forwards
    to this session's file exporter, so files stay cleanly partitioned.

    ``agentm.session.id`` is **not** on the OTel resource. Atoms emitting
    spans / log records stamp it as an attribute on every record (see
    :mod:`agentm.extensions.builtin.observability`).

    :meth:`shutdown` flushes this session's processors, removes them from
    the global providers, and closes the file exporters. Idempotent — safe
    to call from explicit teardown + the ``SessionShutdownEvent`` handler.
    The global providers themselves outlive every session and are torn
    down once at process exit by an ``atexit`` hook.
    """

    session_id: str
    file_path: Path
    tracer: Tracer
    logger: Logger
    tracer_provider: TracerProvider
    logger_provider: LoggerProvider
    span_processor: BlockingBatchSpanProcessor
    log_processor: BlockingBatchLogRecordProcessor
    span_exporter: FileSpanExporter
    log_exporter: FileLogExporter
    _shutdown: bool = False

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        # Drain + remove the per-session processors so other sessions still
        # running in the same process keep emitting. Provider lives on.
        try:
            self.span_processor.shutdown()
        except Exception:  # pragma: no cover
            pass
        try:
            self.log_processor.shutdown()
        except Exception:  # pragma: no cover
            pass
        _remove_span_processor(self.tracer_provider, self.span_processor)
        _remove_log_processor(self.logger_provider, self.log_processor)
        # ``BatchSpanProcessor.shutdown`` already calls exporter.shutdown;
        # the exporters guard with ``self._closed`` so a second call is a
        # no-op.
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
    file_path: Path | None = None,
) -> SessionTelemetry:
    """Build a per-session telemetry handle.

    Constructs (lazily, once) the process-level ``TracerProvider`` +
    ``LoggerProvider`` and attaches a fresh per-session
    ``BlockingBatchSpanProcessor`` + ``BlockingBatchLogRecordProcessor`` to
    them. The processors forward exclusively to this session's file
    exporter, so concurrent sessions never cross-contaminate.

    The output path defaults to ``<cwd>/.agentm/observability/<session_id>.jsonl``.
    Callers that already know the absolute path (e.g. :class:`SessionManager`
    direct-write callers) may override via ``file_path``; the ``cwd``
    argument is then only used as a hint for resolving the default.

    ``scenario_name`` is accepted for API compatibility but is **not**
    stamped onto the resource — atoms that want to record it should
    emit it as a span / log attribute (the observability atom does
    via ``agentm.session.scenario``).

    The same file holds both spans and logs interleaved — the OTLP shape
    per line is what disambiguates them on read.

    BatchSpanProcessor / BatchLogRecordProcessor are configured to **block**
    on queue overflow (the default behavior in opentelemetry-sdk when
    ``max_queue_size`` is hit is to drop; we set queue size large enough
    that overflow is rare, and tests assert no drops at this size).
    """
    del scenario_name  # No longer on the resource; observability atom emits as attribute.

    tracer_provider, logger_provider = setup_process_telemetry()

    if file_path is None:
        file_path = Path(cwd) / ".agentm" / "observability" / f"{session_id}.jsonl"
    else:
        file_path = Path(file_path)

    span_exporter = FileSpanExporter(file_path)
    log_exporter = FileLogExporter(file_path)

    effective_batch_size = (
        max_export_batch_size
        if max_export_batch_size is not None
        else min(512, max_queue_size)
    )

    span_processor = BlockingBatchSpanProcessor(
        span_exporter,
        max_queue_size=max_queue_size,
        schedule_delay_millis=schedule_delay_millis,
        export_timeout_millis=export_timeout_millis,
        max_export_batch_size=effective_batch_size,
        session_id_filter=session_id,
    )
    tracer_provider.add_span_processor(span_processor)

    log_processor = BlockingBatchLogRecordProcessor(
        log_exporter,
        max_queue_size=max_queue_size,
        schedule_delay_millis=schedule_delay_millis,
        export_timeout_millis=export_timeout_millis,
        max_export_batch_size=effective_batch_size,
        session_id_filter=session_id,
    )
    logger_provider.add_log_record_processor(log_processor)

    # Per-session instrument scope: gives consumers a join point in addition
    # to the ``agentm.session.id`` attribute stamped by atoms.
    scope_name = f"{_SCOPE_NAME}.session.{session_id}"
    tracer = tracer_provider.get_tracer(scope_name, _agentm_version())
    logger = logger_provider.get_logger(scope_name, _agentm_version())

    return SessionTelemetry(
        session_id=session_id,
        file_path=file_path,
        tracer=tracer,
        logger=logger,
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        span_processor=span_processor,
        log_processor=log_processor,
        span_exporter=span_exporter,
        log_exporter=log_exporter,
    )


