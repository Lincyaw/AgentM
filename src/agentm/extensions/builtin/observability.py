"""Observability atom — wires the registry-based ``Event → OTel`` mapping.

Per-event OTel translators are registered in ``runtime/event_otel.py`` via
:func:`~agentm.core.lib.otel_dispatch.register_otel`. This atom is what
wires that mapping into the runtime:

* On install, it stamps session-scoped metadata onto the telemetry handle
  (so per-event translators have access to ``scenario``, ``purpose``,
  ``redact_prompts``, etc.).
* It subscribes one uniform handler per channel that calls
  :func:`~agentm.core.lib.otel_dispatch.dispatch_otel` — so adding a new
  Event class with a registered translator automatically reaches the wire.
* It owns the bus-observer side of the contract:
  ``agentm.event.dispatch``, ``agentm.handler.invoke``, and
  ``agentm.handler.mutated`` records come from observer hooks (``on_emit_*``
  / ``on_handler_done``), not from individual Event classes — those records
  describe bus-level activity, not per-event semantics.
* It owns the fingerprint / atom-snapshot bookkeeping for
  ``agentm.session.fingerprint`` and ``agentm.atom.reload`` — those depend
  on catalog state that does not belong on the per-session
  telemetry handle.
* It emits the ``agentm.session.start`` install-time record and stamps
  ``obs_session_start_ns`` so :class:`SessionShutdownEvent` can compute the
  ``agentm.session.end`` duration.

The on-disk shape is unchanged from the pre-refactor writer — semconv tests
in ``tests/unit/extensions/test_observability_semconv.py`` lock it down.
"""

from __future__ import annotations

import hashlib
from loguru import logger
from pathlib import Path
import time
import traceback
from typing import Any

from agentm.core.abi import (
    ActiveSetFingerprint,
    RunEndEvent,
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    BeforeRunEvent,
    BeforeSendEvent,
    DiagnosticEvent,
    Event,
    EventBusObserver,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    ExtensionUnloadEvent,
    Handler,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    MessageAppendedEvent,
    SessionHeaderEmittedEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
    SessionTelemetry,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCommittedEvent,
    TurnBeginEvent,
)
from agentm.core.observability.otel_dispatch import dispatch_otel
from agentm.core.observability.otel_export import setup_session_telemetry
from pydantic import BaseModel

from agentm.core.lib import redact_messages, to_jsonable
from agentm.extensions import ExtensionManifest
from opentelemetry._logs import SeverityNumber


# Channels whose per-emission dispatch records add no diagnostic value over
# the higher-level spans/records this atom writes — ``stream_delta`` fires
# once per LLM token chunk; the assembled assistant message lands on
# ``agentm.message.appended`` so the raw deltas are pure bloat.
_DEFAULT_EXCLUDE_CHANNELS: frozenset[str] = frozenset({StreamDeltaEvent.CHANNEL})

# Channels whose serialized event payload can carry user-supplied prompt
# content. When ``redact_prompts=True`` (default), :func:`redact_messages`
# rewrites the payload to a ``{role, chars, sha256_prefix}`` stub before it
# hits the log record body — same shape as the old writer, narrower scope
# (the ``before_send_to_llm`` channel no longer produces a record).
_REDACTED_CHANNELS: frozenset[str] = frozenset(
    {
        LlmRequestStartEvent.CHANNEL,
        LlmRequestEndEvent.CHANNEL,
        ApiSendUserMessageEvent.CHANNEL,
    }
)

_ATTRIBUTE_STRING_LIMIT = 512


def _nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _event_mutable_fields(event: Any) -> tuple[str, ...]:
    hook = getattr(type(event), "HOOK", None)
    fields = getattr(hook, "mutable_fields", ())
    return tuple(fields) if isinstance(fields, tuple) else ()


def _add_metadata_attribute(attrs: dict[str, Any], key: str, value: Any) -> None:
    if key in attrs or value is None:
        return
    if isinstance(value, (bool, int, float)):
        attrs[key] = value
        return
    if isinstance(value, str):
        attrs[key] = (
            value
            if len(value) <= _ATTRIBUTE_STRING_LIMIT
            else value[:_ATTRIBUTE_STRING_LIMIT]
        )


def _session_metadata_attributes(
    lineage: dict[str, Any] | None,
    experiment: dict[str, Any] | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if lineage:
        lineage_paths = {
            "kind": "agentm.session.lineage.kind",
            "entrypoint": "agentm.session.lineage.entrypoint",
            "source_session_id": "agentm.session.lineage.source_session_id",
            "parent_session_id": "agentm.session.lineage.parent_session_id",
            "root_session_id": "agentm.session.lineage.root_session_id",
            "task_id": "agentm.session.lineage.task_id",
            "persona": "agentm.session.lineage.persona",
            "purpose": "agentm.session.lineage.purpose",
            "trace_label": "agentm.session.lineage.trace_label",
            "workflow_run_id": "agentm.session.lineage.workflow_run_id",
            "workflow_node_id": "agentm.session.lineage.workflow_node_id",
            "fork_point.up_to": "agentm.session.lineage.fork.up_to",
            "fork_point.message_id": "agentm.session.lineage.fork.message_id",
            "fork_point.turn_id": "agentm.session.lineage.fork.turn_id",
            "fork_point.turn_index": "agentm.session.lineage.fork.turn_index",
        }
        for path, attr_key in lineage_paths.items():
            _add_metadata_attribute(attrs, attr_key, _nested_get(lineage, path))

    if experiment:
        experiment_paths = {
            "kind": "agentm.session.experiment.kind",
            "id": "agentm.session.experiment.id",
            "experiment_id": "agentm.session.experiment.id",
            "case_id": "agentm.session.experiment.case_id",
            "baseline_session_id": "agentm.session.experiment.baseline_session_id",
            "reminder_id": "agentm.session.experiment.reminder_id",
            "insert_turn_index": "agentm.session.experiment.insert_turn_index",
            "insert_message_id": "agentm.session.experiment.insert_message_id",
            "reminder_text_hash": "agentm.session.experiment.reminder_text_hash",
            "variant": "agentm.session.experiment.variant",
        }
        for path, attr_key in experiment_paths.items():
            _add_metadata_attribute(attrs, attr_key, _nested_get(experiment, path))
        reminder_text = experiment.get("reminder_text")
        if isinstance(reminder_text, str) and reminder_text:
            _add_metadata_attribute(
                attrs,
                "agentm.session.experiment.reminder_text_hash",
                hashlib.sha256(reminder_text.encode("utf-8")).hexdigest(),
            )
    return attrs


# Channels whose translation lives on :meth:`Event.to_otel`. The atom
# subscribes a uniform delegator per channel; if a new Event class lands
# with its own :meth:`to_otel` override, append its channel here and the
# wire-format extension is complete — no other observability code touched.
_TO_OTEL_CHANNELS: tuple[str, ...] = (
    SessionHeaderEmittedEvent.CHANNEL,
    MessageAppendedEvent.CHANNEL,
    SessionReadyEvent.CHANNEL,
    SessionShutdownEvent.CHANNEL,
    ApiRegisterEvent.CHANNEL,
    ApiSendUserMessageEvent.CHANNEL,
    DiagnosticEvent.CHANNEL,
    ExtensionUnloadEvent.CHANNEL,
    BeforeRunEvent.CHANNEL,
    RunEndEvent.CHANNEL,
    LlmRequestStartEvent.CHANNEL,
    LlmRequestEndEvent.CHANNEL,
    ToolCallEvent.CHANNEL,
    ToolResultEvent.CHANNEL,
    TurnBeginEvent.CHANNEL,
    TurnCommittedEvent.CHANNEL,
)


class ObservabilityConfig(BaseModel):
    include_handler_records: bool = True
    include_mutation_diff: bool = True
    exclude_channels: list[str] | None = None
    redact_prompts: bool = True


MANIFEST = ExtensionManifest(
    name="observability",
    description=(
        "Stream every span, log record, event dispatch, and handler "
        "invocation to a per-session OTLP/JSON ndjson file via the "
        "OpenTelemetry SDK. Single event log."
    ),
    registers=(
        "event:agent_end",
        "event:before_agent_start",
        "event:turn_start",
        "event:turn_end",
        "event:tool_call",
        "event:tool_result",
        "event:llm_request_start",
        "event:llm_request_end",
        "event:api_register",
        "event:api_send_user_message",
        "event:extension_install",
        "event:extension_reload",
        "event:extension_unload",
        "event:session_ready",
        "event:session_shutdown",
        "event:session_header_emitted",
        "event:message_appended",
        "event:diagnostic",
    ),
    config_schema=ObservabilityConfig,
    requires=(),
)


def _handler_label(handler: Handler) -> str:
    mod = getattr(handler, "__module__", None) or "<unknown>"
    qual = getattr(handler, "__qualname__", None) or repr(handler)
    return f"{mod}.{qual}"


def _format_traceback(exc: BaseException | None) -> str | None:
    if exc is None or exc.__traceback__ is None:
        return None
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _deep_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        out: list[dict[str, Any]] = []
        keys = list(before.keys()) + [k for k in after if k not in before]
        for key in keys:
            out.extend(
                _deep_diff(
                    before.get(key, "<missing>"),
                    after.get(key, "<missing>"),
                    f"{path}.{key}" if path else str(key),
                )
            )
            if len(out) > 50:
                out.append({"path": path or "<root>", "_truncated": True})
                break
        return out
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [
                {
                    "path": path or "<root>",
                    "before_len": len(before),
                    "after_len": len(after),
                }
            ]
        out2: list[dict[str, Any]] = []
        for i, (b, a) in enumerate(zip(before, after, strict=False)):
            out2.extend(_deep_diff(b, a, f"{path}[{i}]"))
            if len(out2) > 50:
                out2.append({"path": path or "<root>", "_truncated": True})
                break
        return out2
    return [{"path": path or "<root>", "before": before, "after": after}]


def install(session: Any, config: ObservabilityConfig) -> None:
    _ObservabilityRuntime(session=session, config=config).install()


class _ObservabilityObserver(EventBusObserver):
    """Bus-level records: dispatch, handler invocation, and mutation diffs."""

    def __init__(
        self,
        *,
        session: Any, telemetry: SessionTelemetry,
        include_handlers: bool,
        include_diff: bool,
        exclude_channels: frozenset[str],
        redact_prompts: bool,
    ) -> None:
        self._session = session
        self._telemetry = telemetry
        self._include_handlers = include_handlers
        self._include_diff = include_diff
        self._exclude_channels = exclude_channels
        self._redact_prompts = redact_prompts
        self._pending_diff_snapshots: list[tuple[str, Handler, Any, Any]] = []

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
        del dispatch_id, owner
        if not self._include_diff or not _event_mutable_fields(event):
            return
        try:
            before = to_jsonable(event)
        except Exception as exc:
            logger.debug(
                "observability: pre-handler snapshot serialization failed: {}",
                exc,
            )
            return
        self._pending_diff_snapshots.append((channel, handler, before, event))

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
        del result
        self._record_mutation(channel, handler, event, error, dispatch_id)
        if not self._include_handlers or channel in self._exclude_channels:
            return
        attrs = self._handler_attributes(
            channel=channel,
            handler=handler,
            event=event,
            error=error,
            duration_ns=duration_ns,
            owner=owner,
            dispatch_id=dispatch_id,
        )
        self._telemetry.emit_log(
            "agentm.handler.invoke",
            attributes=attrs,
            severity=SeverityNumber.ERROR if error is not None else SeverityNumber.INFO,
        )

    def on_emit_end(
        self, channel: str, event: Any, results: list[Any], dispatch_id: str
    ) -> None:
        if channel in self._exclude_channels:
            return
        attrs: dict[str, Any] = {
            "agentm.session.id": self._session.id,
            "agentm.event.channel": channel,
            "agentm.event.dispatch_id": dispatch_id,
            "agentm.handler.count": len(results),
        }
        payload = self._event_payload(channel, event)
        if payload is not None:
            attrs["agentm.event.payload"] = payload
        self._telemetry.emit_log("agentm.event.dispatch", attributes=attrs)

    def _handler_attributes(
        self,
        *,
        channel: str,
        handler: Handler,
        event: Any,
        error: BaseException | None,
        duration_ns: int,
        owner: str | None,
        dispatch_id: str,
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "agentm.session.id": self._session.id,
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
            tb = _format_traceback(error)
            if tb is not None:
                attrs["agentm.handler.error.traceback"] = tb
        return attrs

    def _event_payload(self, channel: str, event: Any) -> Any:
        try:
            payload = to_jsonable(event)
        except Exception as exc:
            logger.debug("observability: event payload serialization failed: {}", exc)
            return None
        if channel == BeforeSendEvent.CHANNEL and isinstance(payload, dict):
            payload.pop("messages", None)
        if self._redact_prompts and channel in _REDACTED_CHANNELS:
            return redact_messages(payload)
        return payload

    def _record_mutation(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        error: BaseException | None,
        dispatch_id: str,
    ) -> None:
        if not self._include_diff or not _event_mutable_fields(event):
            return
        before, snapshot_event = self._pop_diff_snapshot(channel, handler)
        if before is None:
            return
        try:
            after = to_jsonable(snapshot_event)
            diff = _deep_diff(before, after)
        except Exception as exc:
            logger.debug("observability: post-handler diff computation failed: {}", exc)
            return
        if not diff:
            return
        self._telemetry.emit_log(
            "agentm.handler.mutated",
            attributes={
                "agentm.session.id": self._session.id,
                "agentm.event.channel": channel,
                "agentm.event.dispatch_id": dispatch_id,
                "agentm.handler.name": _handler_label(handler),
                "agentm.handler.mutations": diff[:100],
                "agentm.handler.raised": error is not None,
            },
        )

    def _pop_diff_snapshot(
        self, channel: str, handler: Handler
    ) -> tuple[Any | None, Any | None]:
        for index in range(len(self._pending_diff_snapshots) - 1, -1, -1):
            snap_channel, snap_handler, snap_before, snap_event = (
                self._pending_diff_snapshots[index]
            )
            if snap_channel == channel and snap_handler is handler:
                del self._pending_diff_snapshots[index]
                return snap_before, snap_event
        return None, None


class _ObservabilityRuntime:
    """Install-time wiring and observability-owned fingerprint bookkeeping."""

    def __init__(self, *, session: Any, config: ObservabilityConfig) -> None:
        self._session = session
        self._config = config
        self._telemetry = setup_session_telemetry(
            session.id,
            Path(session.ctx.cwd or "."),
            scenario_name=session.ctx.scenario,
        )
        user_excludes = config.exclude_channels
        self._exclude_channels = frozenset(_DEFAULT_EXCLUDE_CHANNELS) | (
            frozenset(user_excludes) if user_excludes else frozenset()
        )
        self._active_atom_hashes: dict[str, str] = {}
        self._current_fingerprint: ActiveSetFingerprint | None = None

    def install(self) -> None:
        import agentm.core.observability.event_otel  # noqa: F401

        self._session.services.register("session_telemetry", self._telemetry)
        self._stamp_session_metadata()
        self._emit_session_start()
        self._session.add_observer(
            _ObservabilityObserver(
                session=self._session,
                telemetry=self._telemetry,
                include_handlers=self._config.include_handler_records,
                include_diff=self._config.include_mutation_diff,
                exclude_channels=self._exclude_channels,
                redact_prompts=self._config.redact_prompts,
            )
        )
        self._subscribe_otel_dispatch()

    def _stamp_session_metadata(self) -> None:
        # Every per-event Event.to_otel translator reads these session fields.
        provider_cfg = None  # v2: provider access pending
        self._telemetry.obs_root_session_id = self._session.ctx.root_session_id
        self._telemetry.obs_parent_session_id = self._session.ctx.parent_session_id or ""
        self._telemetry.obs_purpose = self._session.ctx.purpose
        self._telemetry.obs_scenario = self._session.ctx.scenario or ""
        self._telemetry.obs_provider_name = (
            provider_cfg.name if provider_cfg is not None else ""
        )
        self._telemetry.obs_cwd = self._session.ctx.cwd
        self._telemetry.obs_redact_prompts = self._config.redact_prompts
        self._telemetry.obs_session_start_ns = time.time_ns()

    def _emit_session_start(self) -> None:
        body: dict[str, Any] = {
            "session_id": self._session.id,
            "root_session_id": self._session.ctx.root_session_id,
            "parent_session_id": self._session.ctx.parent_session_id,
            "purpose": self._session.ctx.purpose,
            "scenario": self._session.ctx.scenario,
            "cwd": self._session.ctx.cwd,
        }
        if None is not None:
            body["lineage"] = to_jsonable(None)
        if None is not None:
            body["experiment"] = to_jsonable(None)
        attrs: dict[str, Any] = {
            "agentm.session.id": self._session.id,
            "agentm.session.root_id": self._session.ctx.root_session_id,
            "agentm.session.parent_id": self._session.ctx.parent_session_id or "",
            "agentm.session.purpose": self._session.ctx.purpose,
            "agentm.session.scenario": self._session.ctx.scenario or "",
            "agentm.session.cwd": self._session.ctx.cwd,
        }
        attrs.update(_session_metadata_attributes(self._session.lineage, self._session.experiment))
        self._telemetry.emit_log("agentm.session.start", body=body, attributes=attrs)

    def _subscribe_otel_dispatch(self) -> None:
        # Declarative dispatcher: one handler per Event.to_otel-backed channel.
        for channel in _TO_OTEL_CHANNELS:
            self._session.bus.on(channel, self._dispatch_to_otel)

    def _subscribe_fingerprint_bookkeeping(self) -> None:
        self._session.bus.on(SessionReadyEvent.CHANNEL, self._on_session_ready_fingerprint)
        self._session.bus.on(ExtensionInstallEvent.CHANNEL, self._on_extension_install)
        self._session.bus.on(ExtensionReloadEvent.CHANNEL, self._on_extension_reload)

    def _dispatch_to_otel(self, event: Event) -> None:
        dispatch_otel(event, self._telemetry)

    def _compute_loaded_fingerprint(
        self, *, scenario: str | None = None
    ) -> ActiveSetFingerprint:
        self._active_atom_hashes = {}
        for atom in self._session.list_atoms():
            version_hash = self._session.freeze_current(atom.name)
            if atom.current_hash is not None and atom.current_hash != version_hash:
                raise RuntimeError(
                    f"atom identity mismatch for {atom.name!r}: "
                    f"list_atoms={atom.current_hash}, freeze={version_hash}"
                )
            self._active_atom_hashes[atom.name] = version_hash
        # v2: catalog.compute_active_set_fingerprint pending
        return None

    def _on_session_ready_fingerprint(self, event: SessionReadyEvent) -> None:
        # SessionReadyEvent's simple log lives in Event.to_otel; this companion
        # record depends on catalog services and the loaded builtin set.
        self._current_fingerprint = self._compute_loaded_fingerprint(
            scenario=self._session.ctx.scenario
        )
        fp_body: dict[str, Any] = dict(self._current_fingerprint)
        fp_body["task_meta"] = {
            "type": None,
            "difficulty": None,
            "external_id": None,
            "task_class": getattr(event, "task_class", None),
            "eval_run_id": getattr(event, "eval_run_id", None),
            "task_id": getattr(event, "eval_task_id", None),
        }
        self._telemetry.emit_log(
            "agentm.session.fingerprint",
            body=fp_body,
            attributes={
                "agentm.session.id": event.session_id,
                "agentm.session.scenario": self._session.ctx.scenario or "",
                "agentm.task.class": getattr(event, "task_class", "") or "",
                "agentm.task.eval_run_id": getattr(event, "eval_run_id", "") or "",
                "agentm.task.eval_task_id": getattr(event, "eval_task_id", "") or "",
            },
        )

    def _on_extension_install(self, event: ExtensionInstallEvent) -> None:
        self._telemetry.emit_log(
            "agentm.extension.install",
            body={"config": to_jsonable(event.config)},
            attributes={
                "agentm.session.id": self._session.id,
                "agentm.extension.module_path": event.module_path,
                "agentm.extension.phase": event.phase,
                "agentm.extension.duration_ns": event.duration_ns,
                "agentm.extension.trigger": event.trigger,
                "agentm.extension.error": event.error or "",
            },
            severity=SeverityNumber.ERROR if event.error else SeverityNumber.INFO,
        )

    def _on_extension_reload(self, event: ExtensionReloadEvent) -> None:
        if event.name in self._active_atom_hashes or event.old_hash is None:
            self._active_atom_hashes[event.name] = event.new_hash
        # v2: catalog.compute_active_set_fingerprint pending
        fingerprint_after = None
        self._current_fingerprint = fingerprint_after
        self._telemetry.emit_log(
            "agentm.atom.reload",
            body={
                "name": event.name,
                "old_hash": event.old_hash,
                "new_hash": event.new_hash,
                "trigger": event.trigger,
                "tier": event.tier,
                "fingerprint_after": fingerprint_after,
            },
            attributes={
                "agentm.session.id": self._session.id,
                "agentm.atom.name": event.name,
                "agentm.atom.new_hash": event.new_hash,
                "agentm.atom.old_hash": event.old_hash or "",
                "agentm.atom.trigger": event.trigger,
                "agentm.atom.tier": event.tier,
                "agentm.atom.error": event.error or "",
            },
            severity=SeverityNumber.ERROR if event.error else SeverityNumber.INFO,
        )


# Keep ``logger`` referenced (used implicitly by the SDK in error paths).
_ = logger
