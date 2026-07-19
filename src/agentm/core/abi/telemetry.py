"""Atom-facing telemetry Protocol.

Core defines only AgentM's stable telemetry surface. Concrete backends such
as OTel, JSONL, ClickHouse, or host-provided sinks adapt to this Protocol
outside the core substrate.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

TelemetrySeverity = Literal["trace", "debug", "info", "warning", "error", "fatal"]


@runtime_checkable
class SessionTelemetry(Protocol):
    """Atom-facing view of the per-session telemetry handle.

    ``obs_*`` fields are written by the ``observability`` atom at install
    time and read by backend-specific event translators.
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

    def emit_log(
        self,
        event_name: str,
        *,
        body: Any = None,
        attributes: dict[str, Any] | None = None,
        severity: TelemetrySeverity = "info",
    ) -> None: ...

    def shutdown(self) -> None: ...


__all__ = ["SessionTelemetry", "TelemetrySeverity"]
