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
* It owns the fingerprint / atom-hash bookkeeping for
  ``agentm.session.fingerprint`` and ``agentm.atom.reload`` — those depend
  on catalog + resource-writer state that does not belong on the per-session
  telemetry handle.
* It emits the ``agentm.session.start`` install-time record and stamps
  ``obs_session_start_ns`` so :class:`SessionShutdownEvent` can compute the
  ``agentm.session.end`` duration.

The on-disk shape is unchanged from the pre-refactor writer — semconv tests
in ``tests/unit/extensions/test_observability_semconv.py`` lock it down.
"""

from __future__ import annotations

import hashlib
import inspect
from loguru import logger
import time
import traceback
from typing import Any

from agentm.core.abi import (
    ActiveSetFingerprint,
    AgentEndEvent,
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    BeforeAgentStartEvent,
    BeforeCompactEvent,
    BeforeSendToLlmEvent,
    ContextEvent,
    DiagnosticEvent,
    Event,
    EventBusObserver,
    ExtensionAPI,
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
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.core.observability.otel_dispatch import dispatch_otel
from pydantic import BaseModel

from agentm.core.lib import redact_messages, to_jsonable
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import discover_builtin
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

# Mutable channels we diff for handler.invoke "what changed" attribute. Same
# set the old writer tracked; restricting the diff to these avoids O(N·M)
# double-serialization on every handler invocation.
_MUTABLE_CHANNELS = frozenset(
    {
        BeforeAgentStartEvent.CHANNEL,
        ContextEvent.CHANNEL,
        ToolCallEvent.CHANNEL,
        ToolResultEvent.CHANNEL,
        BeforeCompactEvent.CHANNEL,
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


def _add_metadata_attribute(
    attrs: dict[str, Any], key: str, value: Any
) -> None:
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
    BeforeAgentStartEvent.CHANNEL,
    AgentEndEvent.CHANNEL,
    LlmRequestStartEvent.CHANNEL,
    LlmRequestEndEvent.CHANNEL,
    ToolCallEvent.CHANNEL,
    ToolResultEvent.CHANNEL,
    TurnStartEvent.CHANNEL,
    TurnEndEvent.CHANNEL,
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

def install(api: ExtensionAPI, config: ObservabilityConfig) -> None:
    telemetry: SessionTelemetry = api.get_session_telemetry()

    include_handlers = config.include_handler_records
    include_diff = config.include_mutation_diff
    user_excludes = config.exclude_channels
    exclude_channels = frozenset(_DEFAULT_EXCLUDE_CHANNELS) | (
        frozenset(user_excludes) if user_excludes else frozenset()
    )
    redact_prompts = config.redact_prompts

    # --- Stamp session-scoped metadata onto the telemetry handle ----------
    # Every per-event :meth:`Event.to_otel` translator reads from these
    # fields; the substrate constructs the handle with empty defaults, and
    # the observability atom is what brings them to life.
    provider_cfg = api.provider
    telemetry.obs_root_session_id = api.root_session_id
    telemetry.obs_parent_session_id = api.parent_session_id or ""
    telemetry.obs_purpose = api.purpose
    telemetry.obs_scenario = api.scenario or ""
    telemetry.obs_provider_name = provider_cfg.name if provider_cfg is not None else ""
    telemetry.obs_cwd = api.cwd
    telemetry.obs_redact_prompts = redact_prompts
    telemetry.obs_session_start_ns = time.time_ns()

    pending_diff_snapshots: list[tuple[str, Handler, Any, Any]] = []

    # --- Session bootstrap log record -------------------------------------
    session_start_body: dict[str, Any] = {
        "session_id": api.session_id,
        "root_session_id": api.root_session_id,
        "parent_session_id": api.parent_session_id,
        "purpose": api.purpose,
        "scenario": api.scenario,
        "cwd": api.cwd,
    }
    if api.lineage is not None:
        session_start_body["lineage"] = to_jsonable(api.lineage)
    if api.experiment is not None:
        session_start_body["experiment"] = to_jsonable(api.experiment)
    session_start_attrs: dict[str, Any] = {
        "agentm.session.id": api.session_id,
        "agentm.session.root_id": api.root_session_id,
        "agentm.session.parent_id": api.parent_session_id or "",
        "agentm.session.purpose": api.purpose,
        "agentm.session.scenario": api.scenario or "",
        "agentm.session.cwd": api.cwd,
    }
    session_start_attrs.update(
        _session_metadata_attributes(api.lineage, api.experiment)
    )
    telemetry.emit_log(
        "agentm.session.start",
        body=session_start_body,
        attributes=session_start_attrs,
    )

    # --- Bus observer for dispatch / handler.invoke records ---------------

    class _Observer(EventBusObserver):
        def on_emit_start(self, channel: str, event: Any) -> None:
            del channel, event

        def on_handler_start(
            self, channel: str, handler: Handler, event: Any
        ) -> None:
            if not include_diff or channel not in _MUTABLE_CHANNELS:
                return
            try:
                before = to_jsonable(event)
            except Exception as exc:
                logger.debug("observability: pre-handler snapshot serialization failed: {}", exc)
                return
            pending_diff_snapshots.append((channel, handler, before, event))

        def on_handler_done(
            self,
            channel: str,
            handler: Handler,
            event: Any,
            result: Any,
            error: BaseException | None,
            duration_ns: int,
            owner: str | None = None,
        ) -> None:
            del result
            self._record_mutation(channel, handler, event, error)
            if not include_handlers or channel in exclude_channels:
                return
            dispatch_id = getattr(event, "dispatch_id", "") or ""
            attrs: dict[str, Any] = {
                "agentm.session.id": api.session_id,
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
            telemetry.emit_log(
                "agentm.handler.invoke",
                attributes=attrs,
                severity=(
                    SeverityNumber.ERROR if error is not None else SeverityNumber.INFO
                ),
            )

        def on_emit_end(self, channel: str, event: Any, results: list[Any]) -> None:
            if channel in exclude_channels:
                return
            dispatch_id = getattr(event, "dispatch_id", "") or ""
            attrs: dict[str, Any] = {
                "agentm.session.id": api.session_id,
                "agentm.event.channel": channel,
                "agentm.event.dispatch_id": dispatch_id,
                "agentm.handler.count": len(results),
            }
            payload: Any = None
            try:
                payload = to_jsonable(event)
            except Exception as exc:
                logger.debug("observability: event payload serialization failed: {}", exc)
                payload = None
            if channel == BeforeSendToLlmEvent.CHANNEL and isinstance(payload, dict):
                payload.pop("messages", None)
            if redact_prompts and channel in _REDACTED_CHANNELS:
                payload = redact_messages(payload)
            if payload is not None:
                attrs["agentm.event.payload"] = payload
            telemetry.emit_log(
                "agentm.event.dispatch",
                attributes=attrs,
            )

        def _record_mutation(
            self,
            channel: str,
            handler: Handler,
            event: Any,
            error: BaseException | None,
        ) -> None:
            if not include_diff or channel not in _MUTABLE_CHANNELS:
                return
            before: Any | None = None
            snapshot_event: Any | None = None
            for index in range(len(pending_diff_snapshots) - 1, -1, -1):
                snap_channel, snap_handler, snap_before, snap_event = (
                    pending_diff_snapshots[index]
                )
                if snap_channel == channel and snap_handler is handler:
                    before = snap_before
                    snapshot_event = snap_event
                    del pending_diff_snapshots[index]
                    break
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
            telemetry.emit_log(
                "agentm.handler.mutated",
                attributes={
                    "agentm.session.id": api.session_id,
                    "agentm.event.channel": channel,
                    "agentm.event.dispatch_id": getattr(event, "dispatch_id", "")
                    or "",
                    "agentm.handler.name": _handler_label(handler),
                    "agentm.handler.mutations": diff[:100],
                    "agentm.handler.raised": error is not None,
                },
            )

    api.add_observer(_Observer())

    # --- Fingerprint state -------------------------------------------------
    # ``agentm.session.fingerprint`` + ``agentm.atom.reload`` need the
    # catalog hash machinery and the loaded-builtin set, neither of which is
    # exposed on :class:`SessionTelemetry`. We keep this bookkeeping local
    # to the atom — it is observability-owned state, not per-event payload.

    discovered_builtin = discover_builtin()
    builtin_by_module_path = {
        entry.module_path: entry for entry in discovered_builtin.values()
    }
    loaded_builtin_module_paths: set[str] = set()
    active_atom_hashes: dict[str, str] = {}
    current_fingerprint: ActiveSetFingerprint | None = None

    def _compute_loaded_fingerprint(
        *, scenario: str | None = None
    ) -> ActiveSetFingerprint:
        nonlocal active_atom_hashes
        active_atom_hashes = {}
        writer = api.get_resource_writer()
        for module_path in sorted(loaded_builtin_module_paths):
            entry = builtin_by_module_path.get(module_path)
            if entry is None:
                continue
            source_path = inspect.getsourcefile(entry.module)
            if source_path is None:
                continue
            version_hash: str | None = None
            version_hash = writer.current_version_for_path(source_path)
            if version_hash is None:
                source = inspect.getsource(entry.module)
                version_hash = api.catalog.compute_atom_hash(source)
            active_atom_hashes[entry.name] = version_hash
        return api.catalog.compute_active_set_fingerprint(
            loaded=active_atom_hashes,
            scenario=scenario,
            core_hash=None,
        )

    # SessionReadyEvent's simple log emission moved into ``Event.to_otel``;
    # the fingerprint companion record stays here because it depends on
    # ``loaded_builtin_module_paths`` and the catalog services.
    def _on_session_ready_fingerprint(event: SessionReadyEvent) -> None:
        nonlocal current_fingerprint
        current_fingerprint = _compute_loaded_fingerprint(scenario=api.scenario)
        fp_body: dict[str, Any] = dict(current_fingerprint)
        fp_body["task_meta"] = {
            "type": None,
            "difficulty": None,
            "external_id": None,
            "task_class": getattr(event, "task_class", None),
            "eval_run_id": getattr(event, "eval_run_id", None),
            "task_id": getattr(event, "eval_task_id", None),
        }
        telemetry.emit_log(
            "agentm.session.fingerprint",
            body=fp_body,
            attributes={
                "agentm.session.id": event.session_id,
                "agentm.session.scenario": api.scenario or "",
                "agentm.task.class": getattr(event, "task_class", "") or "",
                "agentm.task.eval_run_id": getattr(event, "eval_run_id", "") or "",
                "agentm.task.eval_task_id": getattr(event, "eval_task_id", "") or "",
            },
        )

    def _on_extension_install(event: ExtensionInstallEvent) -> None:
        if event.phase == "end" and event.module_path in builtin_by_module_path:
            loaded_builtin_module_paths.add(event.module_path)
        telemetry.emit_log(
            "agentm.extension.install",
            body={"config": to_jsonable(event.config)},
            attributes={
                "agentm.session.id": api.session_id,
                "agentm.extension.module_path": event.module_path,
                "agentm.extension.phase": event.phase,
                "agentm.extension.duration_ns": event.duration_ns,
                "agentm.extension.trigger": event.trigger,
                "agentm.extension.error": event.error or "",
            },
            severity=(SeverityNumber.ERROR if event.error else SeverityNumber.INFO),
        )

    def _on_extension_reload(event: ExtensionReloadEvent) -> None:
        nonlocal current_fingerprint
        if event.name in active_atom_hashes or event.old_hash is None:
            active_atom_hashes[event.name] = event.new_hash
        fingerprint_after = api.catalog.compute_active_set_fingerprint(
            loaded=active_atom_hashes,
            scenario=(
                None
                if current_fingerprint is None
                else current_fingerprint.get("scenario")
            ),
            core_hash=None,
        )
        current_fingerprint = fingerprint_after
        telemetry.emit_log(
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
                "agentm.session.id": api.session_id,
                "agentm.atom.name": event.name,
                "agentm.atom.new_hash": event.new_hash,
                "agentm.atom.old_hash": event.old_hash or "",
                "agentm.atom.trigger": event.trigger,
                "agentm.atom.tier": event.tier,
                "agentm.atom.error": event.error or "",
            },
            severity=(SeverityNumber.ERROR if event.error else SeverityNumber.INFO),
        )

    # --- Wire subscriptions -----------------------------------------------
    # The declarative dispatcher: one handler per channel, uniformly
    # delegating to ``dispatch_otel(event, telemetry)``. New Event
    # subclasses land on the wire by registering a translator in
    # ``event_otel.py`` and adding their channel to ``_TO_OTEL_CHANNELS``.

    def _dispatch_to_otel(event: Event) -> None:
        dispatch_otel(event, telemetry)

    for channel in _TO_OTEL_CHANNELS:
        api.on(channel, _dispatch_to_otel)

    # Observability-owned bookkeeping that doesn't fit the per-event registry.
    api.on(SessionReadyEvent.CHANNEL, _on_session_ready_fingerprint)
    api.on(ExtensionInstallEvent.CHANNEL, _on_extension_install)
    api.on(ExtensionReloadEvent.CHANNEL, _on_extension_reload)

# Keep ``logger`` referenced (used implicitly by the SDK in error paths).
_ = logger
