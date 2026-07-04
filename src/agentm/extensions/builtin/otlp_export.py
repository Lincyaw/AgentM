"""Builtin ``otlp_export`` atom: ship spans + logs to a remote OTLP endpoint.

AgentM's canonical wire format is OTLP/JSON ndjson written to
``$AGENTM_HOME/observability/<session_id>.jsonl`` by default (see
``observability.py`` +
``core.runtime.otel_export.FileSpanExporter``/``FileLogExporter``). That file
path is enough for self-contained replay, evolution, and llmharness
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
| ``endpoint`` | ``OTEL_EXPORTER_OTLP_ENDPOINT``  | **required**               |
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
from dataclasses import dataclass
from typing import Any, Final

from loguru import logger
from agentm.core.abi import ExtensionAPI, ExtensionLoadError
from agentm.extensions import ExtensionManifest
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel


_DEFAULT_PROTOCOL = "grpc"
_DEFAULT_GRPC_INSECURE = True
_DEFAULT_TIMEOUT_SECONDS = 10
_SUPPORTED_PROTOCOLS: Final = frozenset({"grpc", "http/protobuf"})


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


@dataclass(frozen=True, slots=True)
class _ResolvedOtlpConfig:
    endpoint: str
    protocol: str
    headers: dict[str, str] | None
    insecure: bool
    timeout: int


class _ProcessOtlpExportState:
    """Process-level idempotence guard for OTLP processor attachment."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attached = False

    def attach_once(self, runtime: "_OtlpExportRuntime") -> None:
        with self._lock:
            if self._attached:
                return
            runtime.attach()
            self._attached = True


class _OtlpExportRuntime:
    """Resolve config and attach OTLP exporters to the process telemetry handle."""

    def __init__(self, api: ExtensionAPI, config: OtlpExportConfig) -> None:
        self._api = api
        self._config = config

    def attach(self) -> None:
        # Reach the process-level SDK TracerProvider + LoggerProvider through
        # the per-session telemetry handle. The substrate does not register
        # these as OTel globals (so ``opentelemetry.trace.get_tracer_provider``
        # returns ``ProxyTracerProvider``); the handle holds direct references
        # we can attach our network processors to.
        #
        # Since core.observability.otel_export.setup_process_telemetry() now
        # attaches OTLP exporters early when OTEL_EXPORTER_OTLP_ENDPOINT is
        # set, this extension is only needed when the env var was NOT set at
        # provider-construction time but the user explicitly mounts the
        # extension with a config override.
        telemetry = self._api.get_session_telemetry()
        resolved = self._resolve_config()
        span_exporter, log_exporter = self._build_exporters(resolved)
        span_processor = BatchSpanProcessor(span_exporter)
        log_processor = BatchLogRecordProcessor(log_exporter)
        telemetry.tracer_provider.add_span_processor(span_processor)
        telemetry.logger_provider.add_log_record_processor(log_processor)
        self._register_shutdown(span_processor, log_processor)

    def _resolve_config(self) -> _ResolvedOtlpConfig:
        endpoint = self._config.endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            raise ExtensionLoadError(
                __name__,
                RuntimeError(
                    "No OTLP endpoint configured. Set OTEL_EXPORTER_OTLP_ENDPOINT "
                    "or pass endpoint in the extension config.\n\n"
                    "  Quick start:\n"
                    "    docker compose -f tools/otel/docker-compose.yaml up -d\n"
                    "    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317"
                ),
            )
        protocol = (
            self._config.protocol
            or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
            or _DEFAULT_PROTOCOL
        )
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise ExtensionLoadError(
                __name__,
                ValueError(
                    f"otlp_export: unsupported protocol {protocol!r}; "
                    "expected 'grpc' or 'http/protobuf'"
                ),
            )
        return _ResolvedOtlpConfig(
            endpoint=endpoint,
            protocol=protocol,
            headers=_parse_headers(
                self._config.headers or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
            ),
            insecure=self._resolve_insecure(),
            timeout=self._resolve_timeout(),
        )

    def _resolve_insecure(self) -> bool:
        if self._config.insecure is not None:
            return bool(self._config.insecure)
        env_insecure = os.environ.get("OTEL_EXPORTER_OTLP_INSECURE")
        if env_insecure is None:
            return _DEFAULT_GRPC_INSECURE
        return env_insecure.lower() == "true"

    def _resolve_timeout(self) -> int:
        timeout_raw: float | str = (
            self._config.timeout
            if self._config.timeout is not None
            else os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT") or _DEFAULT_TIMEOUT_SECONDS
        )
        return int(float(timeout_raw))

    def _build_exporters(self, config: _ResolvedOtlpConfig) -> tuple[Any, Any]:
        if config.protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter as GrpcLogExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcSpanExporter,
            )

            span_exporter: Any = GrpcSpanExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
                headers=config.headers,
                timeout=config.timeout,
            )
            log_exporter: Any = GrpcLogExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
                headers=config.headers,
                timeout=config.timeout,
            )
            return span_exporter, log_exporter

        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as HttpLogExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpSpanExporter,
        )

        span_exporter = HttpSpanExporter(
            endpoint=config.endpoint,
            headers=config.headers,
            timeout=config.timeout,
        )
        log_exporter = HttpLogExporter(
            endpoint=config.endpoint,
            headers=config.headers,
            timeout=config.timeout,
        )
        return span_exporter, log_exporter

    def _register_shutdown(
        self,
        span_processor: BatchSpanProcessor,
        log_processor: BatchLogRecordProcessor,
    ) -> None:
        def _shutdown() -> None:
            try:
                span_processor.shutdown()
            except Exception as exc:
                logger.debug("otlp_export: span processor shutdown failed: {}", exc)
            try:
                log_processor.shutdown()
            except Exception as exc:
                logger.debug("otlp_export: log processor shutdown failed: {}", exc)

        atexit.register(_shutdown)


_PROCESS_STATE = _ProcessOtlpExportState()


def install(api: ExtensionAPI, config: OtlpExportConfig) -> None:
    _PROCESS_STATE.attach_once(_OtlpExportRuntime(api, config))


__all__: Final = ["MANIFEST", "install"]
