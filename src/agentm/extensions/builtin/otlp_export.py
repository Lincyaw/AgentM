"""Builtin ``otlp_export`` atom: ship spans + logs to a remote OTLP endpoint.

AgentM's canonical wire format is OTLP/JSON ndjson written to
``<cwd>/.agentm/observability/<session_id>.jsonl`` (see ``observability.py``
+ ``core.runtime.otel_export.FileSpanExporter``/``FileLogExporter``). That
file path is enough for self-contained replay, evolution, and llmharness
distillation. For *live* observability against a remote collector
(Jaeger / Tempo / Phoenix / Grafana Cloud / Honeycomb) you mount this
atom on top, which attaches process-level OTLP exporters to the same
``TracerProvider`` + ``LoggerProvider`` the file exporters already feed.

Mount via ``agentm --extension agentm.extensions.builtin.otlp_export``.
The first session to install this atom in a process attaches the
exporters; subsequent sessions reuse the same processors — no per-session
duplication.

Config keys (all optional; env vars take precedence per OTel convention):

| key          | env var                          | default                    |
|--------------|----------------------------------|----------------------------|
| ``endpoint`` | ``OTEL_EXPORTER_OTLP_ENDPOINT``  | ``http://localhost:4317``  |
| ``protocol`` | ``OTEL_EXPORTER_OTLP_PROTOCOL``  | ``grpc``                   |
| ``headers``  | ``OTEL_EXPORTER_OTLP_HEADERS``   | none                       |
| ``insecure`` | ``OTEL_EXPORTER_OTLP_INSECURE``  | ``true`` (grpc only)       |
| ``timeout``  | ``OTEL_EXPORTER_OTLP_TIMEOUT``   | ``10`` (seconds)           |

Only ``grpc`` and ``http/protobuf`` are accepted — ``http/json`` is not
shipped by the upstream SDK. Headers come as a comma-separated
``k=v,k=v`` string, matching the OTel env convention.
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import Any, Final

from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi import ExtensionAPI, ExtensionLoadError
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class OtlpExportConfig(BaseModel):
    endpoint: str | None = None
    protocol: str | None = None
    headers: str | None = None
    insecure: bool | None = None
    timeout: float | None = None


MANIFEST = ExtensionManifest(
    name="otlp_export",
    description=(
        "Attach process-level OTLP span + log exporters so every session's "
        "spans and logs are forwarded to a remote collector in addition to "
        "the on-disk ndjson file. Idempotent across sessions in one process."
    ),
    registers=(),  # Attaches to process-global OTel providers; no surface.
    config_schema=OtlpExportConfig,
    requires=("observability",),
    api_version=1,
    tier=1,
)

# Process-level guard. ``install`` may run once per session in one Python
# process; the global TracerProvider only wants the OTLP processor attached
# once or every export would be duplicated.
_lock = threading.Lock()
_attached = False


def _parse_headers(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    out: dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
    return out or None


def install(api: ExtensionAPI, config: OtlpExportConfig) -> None:
    # Reach the process-level SDK TracerProvider + LoggerProvider through
    # the per-session telemetry handle. The substrate does not register
    # these as OTel globals (so ``opentelemetry.trace.get_tracer_provider``
    # returns ``ProxyTracerProvider``); the handle holds direct references
    # we can attach our network processors to.
    telemetry = api.get_session_telemetry()
    global _attached
    with _lock:
        if _attached:
            return

        endpoint = (
            config.endpoint
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "http://localhost:4317"
        )
        protocol = (
            config.protocol or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "grpc"
        )
        if protocol not in ("grpc", "http/protobuf"):
            raise ExtensionLoadError(
                __name__,
                ValueError(
                    f"otlp_export: unsupported protocol {protocol!r}; "
                    "expected 'grpc' or 'http/protobuf'"
                ),
            )
        headers = _parse_headers(
            config.headers or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        )
        insecure_raw = config.insecure
        if insecure_raw is None:
            env_insecure = os.environ.get("OTEL_EXPORTER_OTLP_INSECURE")
            insecure = env_insecure is None or env_insecure.lower() == "true"
        else:
            insecure = bool(insecure_raw)
        timeout_raw: float | str = (
            config.timeout or os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT") or 10
        )
        timeout = int(float(timeout_raw))

        if protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter as GrpcLogExporter,
            )

            span_exporter: Any = GrpcSpanExporter(
                endpoint=endpoint,
                insecure=insecure,
                headers=headers,
                timeout=timeout,
            )
            log_exporter: Any = GrpcLogExporter(
                endpoint=endpoint,
                insecure=insecure,
                headers=headers,
                timeout=timeout,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                OTLPLogExporter as HttpLogExporter,
            )

            span_exporter = HttpSpanExporter(
                endpoint=endpoint,
                headers=headers,
                timeout=timeout,
            )
            log_exporter = HttpLogExporter(
                endpoint=endpoint,
                headers=headers,
                timeout=timeout,
            )

        span_processor = BatchSpanProcessor(span_exporter)
        log_processor = BatchLogRecordProcessor(log_exporter)
        telemetry.tracer_provider.add_span_processor(span_processor)
        telemetry.logger_provider.add_log_record_processor(log_processor)

        def _shutdown() -> None:
            try:
                span_processor.shutdown()
            except Exception:
                pass
            try:
                log_processor.shutdown()
            except Exception:
                pass

        atexit.register(_shutdown)
        _attached = True


__all__: Final = ["MANIFEST", "install"]
