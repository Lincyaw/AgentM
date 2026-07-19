"""Observability atom for SDK sessions.

The atom has two responsibilities:

1. Attach a per-session OTel telemetry handle.
2. Bridge the EventBus into telemetry, separating bus mechanics
   (dispatch/handler records) from semantic event translators.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    ApiRegisterEvent,
    BeforeSendEvent,
    BusPriority,
    DiagnosticEvent,
    Event,
    EventBusObserver,
    ExtensionInstallEvent,
    Handler,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    RunEndEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
    SessionTelemetry,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnBeginEvent,
    TurnCommittedEvent,
)
from agentm.core.lib.serialization import to_jsonable
from agentm.core.lib.redact import redact_messages
from agentm.extensions.observability.otel_dispatch import dispatch_otel
from agentm.extensions.observability.otel_export import setup_session_telemetry
from agentm.extensions import ExtensionManifest

_DEFAULT_EXCLUDE_CHANNELS: frozenset[str] = frozenset({StreamDeltaEvent.CHANNEL})
_REDACTED_CHANNELS: frozenset[str] = frozenset(
    {
        BeforeSendEvent.CHANNEL,
        LlmRequestStartEvent.CHANNEL,
        LlmRequestEndEvent.CHANNEL,
    }
)

_TO_OTEL_CHANNELS: tuple[str, ...] = (
    SessionReadyEvent.CHANNEL,
    SessionShutdownEvent.CHANNEL,
    ApiRegisterEvent.CHANNEL,
    DiagnosticEvent.CHANNEL,
    ExtensionInstallEvent.CHANNEL,
    RunEndEvent.CHANNEL,
    LlmRequestStartEvent.CHANNEL,
    LlmRequestEndEvent.CHANNEL,
    ToolCallEvent.CHANNEL,
    ToolResultEvent.CHANNEL,
    TurnBeginEvent.CHANNEL,
    TurnCommittedEvent.CHANNEL,
)


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_handler_records: bool = True
    exclude_channels: list[str] | None = None
    redact_prompts: bool = True
    export: Literal["auto", "local_file", "otlp"] = "auto"


MANIFEST = ExtensionManifest(
    name="observability",
    description="Attach session-scoped OTel spans and logs to the EventBus.",
    registers=(
        "event:session_ready",
        "event:session_shutdown",
        "event:turn_begin",
        "event:turn_committed",
        "event:tool_call",
        "event:tool_result",
        "event:llm_request_start",
        "event:llm_request_end",
        "event:run_end",
        "event:api_register",
        "event:extension_install",
        "event:diagnostic",
    ),
    config_schema=ObservabilityConfig,
    requires=(),
    priority=AtomInstallPriority.OBSERVABILITY,
)


def _handler_label(handler: Handler) -> str:
    module = getattr(handler, "__module__", None)
    qualname = getattr(handler, "__qualname__", None)
    if not isinstance(module, str):
        module = type(handler).__module__
    if not isinstance(qualname, str):
        qualname = type(handler).__qualname__
    return f"{module}.{qualname}"


def _format_traceback(exc: BaseException | None) -> str | None:
    if exc is None or exc.__traceback__ is None:
        return None
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class _ObservabilityObserver(EventBusObserver):
    def __init__(
        self,
        *,
        session: AtomAPI,
        telemetry: SessionTelemetry,
        include_handlers: bool,
        exclude_channels: frozenset[str],
        redact_prompts: bool,
    ) -> None:
        self._session = session
        self._telemetry = telemetry
        self._include_handlers = include_handlers
        self._exclude_channels = exclude_channels
        self._redact_prompts = redact_prompts

    def on_emit_start(self, channel: str, event: Any, dispatch_id: str) -> None:
        del channel, event, dispatch_id

    def on_handler_start(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        dispatch_id: str,
        owner: str | None,
    ) -> None:
        del channel, handler, event, dispatch_id, owner

    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
        dispatch_id: str,
        owner: str | None = None,
    ) -> None:
        del event, result
        if not self._include_handlers or channel in self._exclude_channels:
            return
        attrs: dict[str, Any] = {
            "agentm.session.id": self._session.ctx.session_id,
            "agentm.event.channel": channel,
            "agentm.event.dispatch_id": dispatch_id,
            "agentm.handler.name": _handler_label(handler),
            "agentm.handler.duration_ns": duration_ns,
            "agentm.handler.raised": error is not None,
        }
        if owner is not None:
            attrs["agentm.handler.extension"] = owner
        if error is not None:
            attrs["agentm.handler.error.type"] = type(error).__name__
            trace = _format_traceback(error)
            if trace is not None:
                attrs["agentm.handler.error.traceback"] = trace
        self._telemetry.emit_log(
            "agentm.handler.invoke",
            attributes=attrs,
            severity="error" if error is not None else "info",
        )

    def on_emit_end(
        self,
        channel: str,
        event: Any,
        results: list[Any],
        dispatch_id: str,
    ) -> None:
        if channel in self._exclude_channels:
            return
        attrs: dict[str, Any] = {
            "agentm.session.id": self._session.ctx.session_id,
            "agentm.event.channel": channel,
            "agentm.event.dispatch_id": dispatch_id,
            "agentm.handler.count": len(results),
        }
        payload = self._event_payload(channel, event)
        if payload is not None:
            attrs["agentm.event.payload"] = payload
        self._telemetry.emit_log("agentm.event.dispatch", attributes=attrs)

    def _event_payload(self, channel: str, event: Any) -> Any:
        try:
            payload = to_jsonable(event)
        except Exception as exc:
            logger.debug(
                "observability: event payload serialization failed with {}",
                type(exc).__name__,
            )
            return {
                "type": "serialization_error",
                "error_type": type(exc).__name__,
            }
        if channel == BeforeSendEvent.CHANNEL and isinstance(payload, dict):
            payload.pop("messages", None)
        if self._redact_prompts and channel in _REDACTED_CHANNELS:
            return redact_messages(payload)
        return payload


class _ObservabilityRuntime:
    def __init__(self, *, session: AtomAPI, config: ObservabilityConfig) -> None:
        self._session = session
        self._config = config
        self._telemetry = setup_session_telemetry(
            session.ctx.session_id,
            export_mode=config.export,
        )
        user_excludes = config.exclude_channels
        self._exclude_channels = frozenset(_DEFAULT_EXCLUDE_CHANNELS) | (
            frozenset(user_excludes) if user_excludes else frozenset()
        )

    def install(self) -> None:
        import agentm.extensions.observability.event_otel  # noqa: F401

        self._session.services.register(
            "session_telemetry",
            self._telemetry,
            scope="session",
        )
        self._stamp_session_metadata()
        self._emit_session_start()
        self._session.bus.add_observer(
            _ObservabilityObserver(
                session=self._session,
                telemetry=self._telemetry,
                include_handlers=self._config.include_handler_records,
                exclude_channels=self._exclude_channels,
                redact_prompts=self._config.redact_prompts,
            )
        )
        for channel in _TO_OTEL_CHANNELS:
            self._session.on(channel, self._dispatch_to_otel, priority=BusPriority.POST)

    def _stamp_session_metadata(self) -> None:
        provider = self._session.get_provider()
        self._telemetry.obs_root_session_id = self._session.ctx.root_session_id
        self._telemetry.obs_parent_session_id = (
            self._session.ctx.parent_session_id or ""
        )
        self._telemetry.obs_purpose = self._session.ctx.purpose
        self._telemetry.obs_scenario = self._session.ctx.scenario or ""
        self._telemetry.obs_provider_name = getattr(provider, "name", "") or ""
        self._telemetry.obs_cwd = self._session.ctx.cwd
        self._telemetry.obs_redact_prompts = self._config.redact_prompts
        self._telemetry.obs_session_start_ns = time.time_ns()

    def _emit_session_start(self) -> None:
        body = {
            "session_id": self._session.ctx.session_id,
            "root_session_id": self._session.ctx.root_session_id,
            "parent_session_id": self._session.ctx.parent_session_id,
            "purpose": self._session.ctx.purpose,
            "scenario": self._session.ctx.scenario,
            "cwd": self._session.ctx.cwd,
            "lineage": {
                "session_id": self._session.ctx.session_id,
                "root_session_id": self._session.ctx.root_session_id,
                "parent_session_id": self._session.ctx.parent_session_id or "",
                "purpose": self._session.ctx.purpose,
            },
            "experiment": to_jsonable(self._session.experiment),
        }
        self._telemetry.emit_log(
            "agentm.session.start",
            body=body,
            attributes={
                "agentm.session.id": self._session.ctx.session_id,
                "agentm.session.root_id": self._session.ctx.root_session_id,
                "agentm.session.parent_id": self._session.ctx.parent_session_id
                or "",
                "agentm.session.purpose": self._session.ctx.purpose,
                "agentm.session.scenario": self._session.ctx.scenario or "",
                "agentm.session.cwd": self._session.ctx.cwd,
            },
        )

    def _dispatch_to_otel(self, event: Event) -> None:
        dispatch_otel(event, self._telemetry)


def install(session: AtomAPI, config: ObservabilityConfig) -> None:
    _ObservabilityRuntime(session=session, config=config).install()


__all__ = [
    "MANIFEST",
    "ObservabilityConfig",
    "install",
]
