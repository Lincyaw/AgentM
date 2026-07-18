"""File-backed OTLP exporters for spans and logs.

Writes one OTLP ``ResourceSpans`` / ``ResourceLogs`` element per line as
ndjson at ``$AGENTM_HOME/observability/<session_id>.jsonl`` by default. Each
line is a self-contained JSON object that would be a valid element inside
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
import sys
import threading
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Any, Protocol, Sequence

from google.protobuf.json_format import MessageToDict
from loguru import logger
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry._logs import Logger
from opentelemetry._logs import SeverityNumber
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
from opentelemetry.trace import Span, Tracer

from agentm.core.lib import to_jsonable

__all__ = [
    "BlockingBatchLogRecordProcessor",
    "BlockingBatchSpanProcessor",
    "FileLogExporter",
    "FileSpanExporter",
    "LocalFileSink",
    "OtlpSink",
    "SessionTelemetry",
    "TraceSink",
    "attach_loguru_otel_sink",
    "iter_log_records",
    "iter_spans",
    "otlp_export_reachable",
    "otlp_is_active",
    "otlp_unwrap",
    "resolve_observability_dir",
    "resolve_otlp_endpoint",
    "setup_process_telemetry",
    "setup_session_telemetry",
    "shutdown_process_telemetry",
]

from agentm.core.lib.observability_dir import (  # noqa: E402
    file_export_requested,
    resolve_observability_dir,
)
from agentm.core.observability.otlp import (  # noqa: E402
    iter_log_records,
    iter_spans,
    otlp_unwrap,
)


class _SessionFilterMixin:
    """Adds session_id_filter to batch processors.

    The queue is sized generously (default 100k) so overflow-drops are
    rare. If drops do occur the SDK silently evicts the oldest record —
    acceptable given the queue headroom.
    """

    _session_id_filter: str | None

    def __init__(self, *args: Any, session_id_filter: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._session_id_filter = session_id_filter

    def _should_drop(self, attrs: Any) -> bool:
        if self._session_id_filter is None:
            return False
        return (attrs or {}).get("agentm.session.id") != self._session_id_filter


class BlockingBatchSpanProcessor(_SessionFilterMixin, BatchSpanProcessor):
    """Batch span processor with session filtering."""

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        if self._should_drop(span.attributes):
            return
        super().on_end(span)


class BlockingBatchLogRecordProcessor(_SessionFilterMixin, BatchLogRecordProcessor):
    """Batch log processor with session filtering."""

    def on_emit(self, log_record):  # type: ignore[override, no-untyped-def]
        if self._should_drop(getattr(log_record, "log_record", log_record).attributes):
            return
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


_file_export_locks_guard = threading.Lock()
_file_export_locks: dict[Path, threading.Lock] = {}


def _file_export_lock(path: Path) -> threading.Lock:
    key = path.absolute()
    with _file_export_locks_guard:
        lock = _file_export_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _file_export_locks[key] = lock
        return lock


class _FileOtlpExporter:
    """Shared ndjson-file writer for OTLP spans and logs.

    Subclasses only set ``_encode``, ``_payload_key``, and result-type
    constants — the serialization, file I/O, locking, and lifecycle are
    identical for both signal types. Span and log exporters for one session
    deliberately share a path-level lock so each OTLP element lands as one
    valid NDJSON line even though the SDK exports both signals concurrently.
    """

    _encode: Any  # set by subclass
    _payload_key: str  # "resourceSpans" or "resourceLogs"

    def __init__(self, file_path: Path) -> None:
        self._path = Path(file_path)
        self._lock = _file_export_lock(self._path)
        self._fh = _open_append(self._path)
        self._closed = False

    def _export_batch(self, batch: Sequence[Any], *, success: Any, failure: Any) -> Any:
        if not batch:
            return success
        try:
            request = self._encode(batch)
            payload = MessageToDict(
                request,
                preserving_proto_field_name=False,
                use_integers_for_enums=False,
            )
        except Exception as exc:  # pragma: no cover — encoder bugs are unexpected
            logger.warning("otel_export: batch encode failed: {}", exc)
            return failure
        if not payload.get(self._payload_key):
            return success
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            if self._closed:
                return failure
            self._fh.write(line)
            self._fh.write("\n")
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except (OSError, ValueError) as exc:  # pragma: no cover
                # Best-effort durability: the line is already written+flushed,
                # so a failed fsync is non-fatal. Log for diagnosis.
                logger.debug("otel_export: fsync failed on {}: {}", self._path, exc)
        return success

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


class FileSpanExporter(_FileOtlpExporter, SpanExporter):
    """OTLP ``ResourceSpans`` → ndjson file."""

    _encode = staticmethod(encode_spans)
    _payload_key = "resourceSpans"

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._export_batch(
            spans, success=SpanExportResult.SUCCESS, failure=SpanExportResult.FAILURE,
        )


class FileLogExporter(_FileOtlpExporter, LogRecordExporter):
    """OTLP ``ResourceLogs`` → ndjson file."""

    _encode = staticmethod(encode_logs)
    _payload_key = "resourceLogs"

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        return self._export_batch(
            batch, success=LogRecordExportResult.SUCCESS, failure=LogRecordExportResult.FAILURE,
        )


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
_process_otlp_sink: "OtlpSink | None" = None

_DEFAULT_OTLP_ENDPOINT = "http://localhost:4317"


def otlp_is_active() -> bool:
    """Whether the process-level OTLP sink is attached."""
    return _process_otlp_sink is not None


def _probe_endpoint(endpoint: str, timeout: float = 2.0) -> bool:
    """Quick TCP connect to check if an OTLP collector is reachable."""
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 4317
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def resolve_otlp_endpoint() -> str:
    """Endpoint the process would export to: env override or the default."""
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or _DEFAULT_OTLP_ENDPOINT


def otlp_export_reachable() -> bool:
    """Whether the OTLP collector we would export to is reachable *now*.

    Mirrors the endpoint resolution in :func:`setup_process_telemetry` — the
    configured ``OTEL_EXPORTER_OTLP_ENDPOINT`` or the ``localhost:4317``
    default, probed via a short TCP connect. Unlike :func:`otlp_is_active`
    (which reports whether the process sink is *already* attached) this can be
    called before any session emits, so the session-store selection can decide
    whether the ClickHouse-backed store is safe: its ``create()`` persists
    nothing locally and relies entirely on this export path reaching the
    collector.
    """
    return _probe_endpoint(resolve_otlp_endpoint())


# --- Trace sinks --------------------------------------------------------------
#
# Exactly one sink handles routine export for any given record: the
# process-level :class:`OtlpSink` when a collector is reachable (spans/logs
# reach ClickHouse via the collector), otherwise a per-session
# :class:`LocalFileSink`. Selection happens in :func:`setup_process_telemetry`
# / :func:`setup_session_telemetry` and nowhere else — a single attach site is
# what rules out double export by construction. Explicit ``file_path``
# (SessionManager persistence) and the ``AGENTM_OBSERVABILITY_DIR`` opt-in
# additionally force a file sink regardless of collector reachability.


class TraceSink(Protocol):
    """Destination for the process's spans and log records."""

    def attach(self, tp: TracerProvider, lp: LoggerProvider) -> bool: ...

    def shutdown(self) -> None: ...


class OtlpSink:
    """Process-level network sink: OTLP exporters → collector.

    Configuration comes from the standard OTel env vars
    (``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``_PROTOCOL`` / ``_HEADERS`` /
    ``_INSECURE`` / ``_TIMEOUT``). Attached at most once per process; the
    providers' atexit shutdown drains its processors.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def attach(self, tp: TracerProvider, lp: LoggerProvider) -> bool:
        protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        headers_raw = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        headers: dict[str, str] | None = None
        if headers_raw:
            headers = {}
            for chunk in headers_raw.split(","):
                chunk = chunk.strip()
                if "=" in chunk:
                    k, _, v = chunk.partition("=")
                    if k.strip():
                        headers[k.strip()] = v.strip()
            headers = headers or None

        insecure_env = os.environ.get("OTEL_EXPORTER_OTLP_INSECURE")
        insecure = insecure_env is None or insecure_env.lower() == "true"
        timeout = int(float(os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", "10")))

        try:
            span_exp: Any
            log_exp: Any
            if protocol == "http/protobuf":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter as HttpSpanExp,
                )
                from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                    OTLPLogExporter as HttpLogExp,
                )

                span_exp = HttpSpanExp(
                    endpoint=self._endpoint, headers=headers, timeout=timeout,
                )
                log_exp = HttpLogExp(
                    endpoint=self._endpoint, headers=headers, timeout=timeout,
                )
            else:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter as GrpcSpanExp,
                )
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                    OTLPLogExporter as GrpcLogExp,
                )

                span_exp = GrpcSpanExp(
                    endpoint=self._endpoint, insecure=insecure,
                    headers=headers, timeout=timeout,
                )
                log_exp = GrpcLogExp(
                    endpoint=self._endpoint, insecure=insecure,
                    headers=headers, timeout=timeout,
                )
        except Exception as exc:
            # Exporter construction failed (bad endpoint, missing optional
            # deps) — run without network export rather than crashing
            # telemetry setup; the caller falls back to file export.
            logger.warning(
                "otel_export: OTLP exporter setup failed, network export disabled: {}", exc,
            )
            return False

        tp.add_span_processor(BatchSpanProcessor(span_exp))
        lp.add_log_record_processor(BatchLogRecordProcessor(log_exp))
        logger.debug("otel_export: OTLP sink attached to {}", self._endpoint)
        return True

    def shutdown(self) -> None:
        # The process-level providers own the attached processors and drain
        # them in shutdown_process_telemetry(); nothing session-scoped to do.
        return None


class LocalFileSink:
    """Per-session file sink: OTLP-shaped ndjson lines on local disk.

    Wraps a :class:`FileSpanExporter` / :class:`FileLogExporter` pair behind
    session-filtered blocking batch processors. :meth:`shutdown` drains the
    processors, detaches them from the shared providers (other sessions in
    the process keep emitting), and closes the file handles. Idempotent.
    """

    def __init__(
        self,
        session_id: str,
        path: Path,
        *,
        max_queue_size: int,
        schedule_delay_millis: int,
        export_timeout_millis: int,
        max_export_batch_size: int,
    ) -> None:
        self._session_id = session_id
        self.path = path
        self._max_queue_size = max_queue_size
        self._schedule_delay_millis = schedule_delay_millis
        self._export_timeout_millis = export_timeout_millis
        self._max_export_batch_size = max_export_batch_size
        self._tp: TracerProvider | None = None
        self._lp: LoggerProvider | None = None
        self._span_processor: BlockingBatchSpanProcessor | None = None
        self._log_processor: BlockingBatchLogRecordProcessor | None = None
        self._span_exporter: FileSpanExporter | None = None
        self._log_exporter: FileLogExporter | None = None
        self._shutdown = False

    def attach(self, tp: TracerProvider, lp: LoggerProvider) -> bool:
        self._tp = tp
        self._lp = lp
        self._span_exporter = FileSpanExporter(self.path)
        self._log_exporter = FileLogExporter(self.path)
        self._span_processor = BlockingBatchSpanProcessor(
            self._span_exporter,
            max_queue_size=self._max_queue_size,
            schedule_delay_millis=self._schedule_delay_millis,
            export_timeout_millis=self._export_timeout_millis,
            max_export_batch_size=self._max_export_batch_size,
            session_id_filter=self._session_id,
        )
        tp.add_span_processor(self._span_processor)
        self._log_processor = BlockingBatchLogRecordProcessor(
            self._log_exporter,
            max_queue_size=self._max_queue_size,
            schedule_delay_millis=self._schedule_delay_millis,
            export_timeout_millis=self._export_timeout_millis,
            max_export_batch_size=self._max_export_batch_size,
            session_id_filter=self._session_id,
        )
        lp.add_log_record_processor(self._log_processor)
        return True

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._span_processor is not None:
            try:
                self._span_processor.shutdown()
            except Exception as exc:  # pragma: no cover
                logger.debug("otel_export: span processor shutdown failed: {}", exc)
            if self._tp is not None:
                _remove_span_processor(self._tp, self._span_processor)
        if self._log_processor is not None:
            try:
                self._log_processor.shutdown()
            except Exception as exc:  # pragma: no cover
                logger.debug("otel_export: log processor shutdown failed: {}", exc)
            if self._lp is not None:
                _remove_log_processor(self._lp, self._log_processor)
        if self._span_exporter is not None:
            self._span_exporter.shutdown()
        if self._log_exporter is not None:
            self._log_exporter.shutdown()


def setup_process_telemetry() -> tuple[TracerProvider, LoggerProvider]:
    """Lazily construct the process-level OTel providers.

    Idempotent — subsequent calls return the same provider instances. The
    resource attached to each provider carries only ``service.name`` and
    ``service.version``; per-session metadata (``agentm.session.id``,
    ``agentm.scenario.name``) is emitted as **per-span / per-log
    attributes** by the observability atom, not as resource attributes.

    Sink selection happens here, once per process: when the collector at
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default ``http://localhost:4317``) is
    reachable, an :class:`OtlpSink` is attached — before any session emits
    its first event — and every record ships to the collector. When it is
    not, :func:`setup_session_telemetry` attaches a per-session
    :class:`LocalFileSink` instead.

    Returns ``(tracer_provider, logger_provider)``.
    """
    global _global_tracer_provider, _global_logger_provider
    global _global_atexit_registered, _process_otlp_sink
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
        endpoint = resolve_otlp_endpoint()
        if _probe_endpoint(endpoint):
            sink = OtlpSink(endpoint)
            if sink.attach(_global_tracer_provider, _global_logger_provider):
                _process_otlp_sink = sink
        else:
            logger.debug(
                "otel_export: collector at {} unreachable, sessions fall back to file export",
                endpoint,
            )
        if not _global_atexit_registered:
            atexit.register(shutdown_process_telemetry)
            _global_atexit_registered = True
        return _global_tracer_provider, _global_logger_provider


# --- Operational-log bridge -------------------------------------------------
# Forward loguru (operator-facing stderr logs) into the OTel *logs* signal so
# they ship to the collector → ClickHouse alongside structured trace events,
# queryable via ``agentm trace logs``. This is the log signal, NOT traces:
# operational logs become OTLP LogRecords on the process-level LoggerProvider.

# loguru level name → OTLP SeverityNumber.
_LOGURU_OTEL_SEVERITY: dict[str, SeverityNumber] = {
    "TRACE": SeverityNumber.TRACE,
    "DEBUG": SeverityNumber.DEBUG,
    "INFO": SeverityNumber.INFO,
    "SUCCESS": SeverityNumber.INFO2,
    "WARNING": SeverityNumber.WARN,
    "ERROR": SeverityNumber.ERROR,
    "CRITICAL": SeverityNumber.FATAL,
}

# Reentrancy guard: the OTel export path itself logs via loguru, so without
# this a failing exporter could recurse through the sink indefinitely.
_loguru_otel_reentry = threading.local()


def _emit_loguru_record_to_otel(message: Any) -> None:
    """loguru sink: forward one operational log record to the process OTel
    ``LoggerProvider`` (and thus the network exporter, when configured)."""
    if getattr(_loguru_otel_reentry, "active", False):
        return
    lp = _global_logger_provider
    if lp is None:
        return
    record = message.record
    _loguru_otel_reentry.active = True
    try:
        otel_logger = lp.get_logger("agentm.operational", _agentm_version())
        level_name = str(record["level"].name)
        attributes: dict[str, Any] = {
            "code.filepath": str(record["file"].path),
            "code.function": str(record["function"]),
            "code.lineno": int(record["line"]),
            "logger.name": str(record["name"]),
        }
        exc = record["exception"]
        if exc is not None and exc.type is not None:
            attributes["exception.type"] = exc.type.__name__
            attributes["exception.message"] = str(exc.value)
        otel_logger.emit(
            body=str(record["message"]),
            severity_number=_LOGURU_OTEL_SEVERITY.get(level_name, SeverityNumber.INFO),
            severity_text=level_name,
            event_name="agentm.operational.log",
            attributes=attributes,
        )
    except Exception as exc:
        # Operational logging must never break the app, and routing this
        # failure through loguru would re-enter the failing sink. stderr is
        # deliberately independent of loguru and keeps the failure visible.
        try:
            print(f"[agentm] OTLP log sink failed: {exc!r}", file=sys.stderr)
        except OSError:
            pass
    finally:
        _loguru_otel_reentry.active = False


def attach_loguru_otel_sink(*, level: str = "DEBUG") -> int | None:
    """Add a loguru sink that ships operational logs into the OTel logs
    pipeline. No-op (returns ``None``) unless ``OTEL_EXPORTER_OTLP_ENDPOINT``
    is set — there must be a collector to receive them. Set
    ``AGENTM_OTEL_LOGS=false`` to force it off. Returns the loguru sink id."""
    if os.environ.get("AGENTM_OTEL_LOGS", "").lower() in ("false", "0", "no"):
        return None
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    from loguru import logger as _logger

    # Ensure the process-level provider + network log exporter exist before
    # the first operational record arrives.
    setup_process_telemetry()
    return _logger.add(_emit_loguru_record_to_otel, level=level, format="{message}")


def shutdown_process_telemetry() -> None:
    """Tear down the process-level providers (idempotent).

    Registered as an ``atexit`` hook on first :func:`setup_process_telemetry`
    call. Safe to invoke explicitly from tests that want a clean slate.
    """
    global _global_tracer_provider, _global_logger_provider, _process_otlp_sink
    with _global_lock:
        tp = _global_tracer_provider
        lp = _global_logger_provider
        _global_tracer_provider = None
        _global_logger_provider = None
        _process_otlp_sink = None
    if tp is not None:
        try:
            tp.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.debug("otel_export: tracer provider shutdown failed: {}", exc)
    if lp is not None:
        try:
            lp.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.debug("otel_export: logger provider shutdown failed: {}", exc)


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
                # Processor already detached — expected when shutdown races.
                logger.debug("otel_export: processor already removed from provider")


@dataclass(slots=True)
class SessionTelemetry:
    """Handle bundling per-session telemetry plumbing.

    ``tracer`` and ``logger`` are scoped instrument objects (scope name
    ``agentm.session.<id>``) acquired from the shared process-level
    providers. When this session exports to disk, ``file_sink`` holds the
    per-session :class:`LocalFileSink` whose session-filtered processors
    keep files cleanly partitioned between concurrent sessions; when the
    process-level :class:`OtlpSink` handles export, ``file_sink`` is
    ``None`` and records flow only through the shared OTLP processors.

    ``agentm.session.id`` is **not** on the OTel resource. Atoms emitting
    spans / log records stamp it as an attribute on every record (see
    :mod:`agentm.extensions.builtin.observability`).

    :meth:`shutdown` drains and detaches the file sink (when present).
    Idempotent — safe to call from explicit teardown + the
    ``SessionShutdownEvent`` handler. The global providers themselves
    outlive every session and are torn down once at process exit by an
    ``atexit`` hook.

    **Observability fields** (``obs_*`` and the tracker dicts) are populated
    by the observability atom during :func:`install` and consumed by the
    per-event translators registered in ``event_otel.py``. They are part of
    the telemetry handle so the registry-based dispatch can stay §11-clean:
    events reach the substrate exclusively through this object. Atoms that
    do not install observability simply leave the fields at their
    construction defaults — unregistered event types produce no records.
    """

    session_id: str
    file_path: Path | None
    tracer: Tracer
    logger: Logger
    tracer_provider: TracerProvider
    logger_provider: LoggerProvider
    file_sink: LocalFileSink | None
    # --- Observability-atom-populated context ------------------------------
    # The observability atom stamps these in its install() so each
    # per-event translator has session-scoped metadata without re-reaching
    # for ``ExtensionAPI``. Empty defaults keep the substrate's construction
    # path side-effect-free.
    obs_root_session_id: str = ""
    obs_parent_session_id: str = ""
    obs_purpose: str = ""
    obs_scenario: str = ""
    obs_provider_name: str = ""
    obs_cwd: str = ""
    obs_redact_prompts: bool = True
    obs_session_start_ns: int = 0
    # Lifecycle-pairing tracker for paired Start/End events. See
    # :meth:`open_span` / :meth:`close_span` for the contract.
    span_tracker: dict[tuple[str, str], Span] = field(default_factory=dict)
    # Per-turn aggregator state for the ``agentm.turn.summary`` log record.
    # Mutated by the TurnStartEvent translator (records turn_start_ns,
    # rotates error counter) and the ToolResultEvent translator (increments
    # current_tool_errors on tool-result is_error).
    turn_state: dict[str, int] = field(
        default_factory=lambda: {
            "turn_start_ns": 0,
            "previous_tool_errors": 0,
            "current_tool_errors": 0,
        }
    )
    # Live ``agentm.turn`` span for the currently open turn, used as the
    # parent context for ``chat`` and ``execute_tool`` child spans so the
    # span tree groups per turn under ``invoke_agent``. Set by
    # the TurnStartEvent translator, cleared by the TurnEndEvent translator.
    # ``None`` between turns or when no turn lifecycle has fired.
    current_turn_span: Span | None = None
    _shutdown: bool = False

    # --- Lifecycle-pairing helper -----------------------------------------

    def open_span(self, kind: str, key: str, span: Span) -> None:
        """Register an open span for later closure by a paired End event.

        Many event families come in Start/End pairs whose lifetime maps to
        a single OTel span: ``BeforeAgentStartEvent``→``AgentEndEvent``,
        ``LlmRequestStartEvent``→``LlmRequestEndEvent``,
        ``ToolCallEvent``→``ToolResultEvent``. The Start translator calls
        ``tracer.start_span(...)`` and stashes the resulting :class:`Span`
        here under a ``(kind, key)`` tuple; the End translator looks it up
        with :meth:`pop_span`, sets terminal attributes, and ``.end()`` s it.

        ``kind`` is a short discriminator string (``"invoke_agent"``,
        ``"chat"``, ``"execute_tool"``) so the same correlator (e.g. a
        session id) can serve multiple span families without collision.
        ``key`` is the correlator the End event will recover —
        ``session_id`` for ``invoke_agent``, ``turn_id`` for ``chat``,
        ``tool_call_id`` for ``execute_tool``. Re-using an unclosed key
        replaces the previous span; callers that need preempt semantics
        should :meth:`pop_span` and ``.end()`` first.
        """
        self.span_tracker[(kind, key)] = span

    def pop_span(self, kind: str, key: str) -> Span | None:
        """Look up and remove the open span for ``(kind, key)``.

        Returns ``None`` when no span was opened under that key — End
        events should treat that as a benign "no pair, nothing to close"
        (some test paths and degraded sessions emit End-without-Start),
        not as an error.
        """
        return self.span_tracker.pop((kind, key), None)

    def close_open_spans(self, *, status_description: str) -> None:
        """End every still-open tracked span with an UNSET status.

        Used by the ``SessionShutdownEvent`` translator to guarantee no
        span leaks past session teardown — paired End events handle the
        happy path; this catches abnormal exits (crash, signal, forced
        shutdown) where the End event never fired.
        """
        # Local import keeps the dataclass body trace-import-light; the
        # SDK is already imported at module scope so this is free.
        from opentelemetry.trace import Status, StatusCode

        for span in list(self.span_tracker.values()):
            span.set_status(Status(StatusCode.UNSET, status_description))
            span.end()
        self.span_tracker.clear()

    # --- Attribute-coercion / log-emission helpers ------------------------

    @staticmethod
    def to_otel_attr(value: Any) -> Any:
        """Coerce a Python value to something OTel attributes accept.

        OTel attribute values are restricted to str / bool / int / float /
        and homogeneous sequences thereof. We round-trip anything richer
        through JSON so the on-disk shape stays inspectable. Shared with
        every per-event translator so the on-disk coercion is consistent.
        """
        if value is None:
            return ""
        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        return json.dumps(to_jsonable(value), default=str, ensure_ascii=False)

    def emit_log(
        self,
        event_name: str,
        *,
        body: Any = None,
        attributes: dict[str, Any] | None = None,
        severity: SeverityNumber = SeverityNumber.INFO,
    ) -> None:
        """Emit one log record through this session's OTel ``Logger``.

        The PR-A ``FileLogExporter`` writes one OTLP ``ResourceLogs``
        element per line. The ``event_name`` plays the role of the OTel
        "event name" — consumers (session_manager, indexer,
        query_traces, llmharness CLI) filter on it instead of on a
        legacy ``kind`` field.

        Attribute values are coerced through :meth:`to_otel_attr` so atoms
        can pass Python objects directly without repeating the JSON dance.
        """
        coerced: dict[str, Any] | None
        if attributes is not None:
            coerced = {
                key: self.to_otel_attr(value) for key, value in attributes.items()
            }
        else:
            coerced = None
        self.logger.emit(
            body=to_jsonable(body) if body is not None else None,
            severity_number=severity,
            severity_text=severity.name,
            event_name=event_name,
            attributes=coerced,
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        # Drain + detach the per-session file sink so other sessions still
        # running in the same process keep emitting. Provider lives on.
        if self.file_sink is not None:
            self.file_sink.shutdown()


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

    The output path defaults to ``$AGENTM_HOME/observability/<session_id>.jsonl``.
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

    # File export: explicit path (SessionManager persistence), user opt-in
    # via AGENTM_OBSERVABILITY_DIR, or automatic fallback when no OTLP
    # sink is attached (so traces are never silently lost).
    write_files = file_path is not None or file_export_requested() or not otlp_is_active()
    resolved_path: Path | None = None
    file_sink: LocalFileSink | None = None

    if write_files:
        if file_path is not None:
            resolved_path = Path(file_path)
        else:
            resolved_path = resolve_observability_dir(cwd) / f"{session_id}.jsonl"

        effective_batch_size = (
            max_export_batch_size
            if max_export_batch_size is not None
            else min(512, max_queue_size)
        )
        file_sink = LocalFileSink(
            session_id,
            resolved_path,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_millis,
            export_timeout_millis=export_timeout_millis,
            max_export_batch_size=effective_batch_size,
        )
        file_sink.attach(tracer_provider, logger_provider)

    # Per-session instrument scope: gives consumers a join point in addition
    # to the ``agentm.session.id`` attribute stamped by atoms.
    scope_name = f"{_SCOPE_NAME}.session.{session_id}"
    tracer = tracer_provider.get_tracer(scope_name, _agentm_version())
    logger = logger_provider.get_logger(scope_name, _agentm_version())

    return SessionTelemetry(
        session_id=session_id,
        file_path=resolved_path,
        tracer=tracer,
        logger=logger,
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        file_sink=file_sink,
    )
