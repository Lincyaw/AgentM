"""Atom-facing telemetry Protocol.

Defines the structural contract for the per-session OTel handle returned by
the session services. The concrete implementation
(:class:`~agentm.core.runtime.otel_export.SessionTelemetry`) is a dataclass
that satisfies this Protocol; atoms depend only on this module for the
telemetry handle's shape.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from opentelemetry._logs import SeverityNumber


@runtime_checkable
class SessionTelemetry(Protocol):
    """Atom-facing view of the per-session OTel telemetry handle.

    ``obs_*`` fields are written by the ``observability`` atom at install
    time and read by per-event OTel translators in ``event_otel.py``.
    """

    session_id: str

    obs_root_session_id: str
    obs_parent_session_id: str
    obs_purpose: str
    obs_scenario: str
    obs_provider_name: str
    obs_cwd: str
    obs_redact_prompts: bool
    obs_session_start_ns: int

    # OTel providers — typed Any to avoid pulling opentelemetry SDK into ABI.
    tracer_provider: Any
    logger_provider: Any

    def emit_log(
        self,
        event_name: str,
        *,
        body: Any = None,
        attributes: dict[str, Any] | None = None,
        severity: SeverityNumber = SeverityNumber.INFO,
    ) -> None: ...

    def shutdown(self) -> None: ...


__all__ = ["SessionTelemetry"]
