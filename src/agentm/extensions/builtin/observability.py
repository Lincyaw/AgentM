"""Observability atom — emits the single per-session event log in OTLP/JSON
ndjson via the OpenTelemetry SDK.

This atom is the writer side of ``.claude/designs/single-event-log.md``:

* **Spans** (paired start/end, written through ``api.get_session_telemetry().tracer``):
  - ``invoke_agent <scenario_or_purpose>`` — INTERNAL, from
    :class:`BeforeAgentStartEvent` to :class:`AgentEndEvent`. The
    session-scoped span for one agent run.
  - ``chat <model>`` — CLIENT, from :class:`LlmRequestStartEvent` to
    :class:`LlmRequestEndEvent`. The LLM-call span; usage attributes
    land on the paired ``agentm.turn.summary`` log record (usage is
    only available at :class:`TurnEndEvent` time, after the chat span
    has already ended).
  - ``execute_tool <tool_name>`` — INTERNAL, from :class:`ToolCallEvent`
    to :class:`ToolResultEvent`, paired by ``tool_call_id``. Captures
    arguments and result as gen_ai semconv attributes on the span.

* **Log records** (severity INFO, body or attributes; written through
  ``api.get_session_telemetry().logger``):
  - ``agentm.session.header`` — header at every
    :class:`SessionHeaderEmittedEvent`. Body = the SessionHeader dict.
    Read by :class:`SessionManager._load` to reconstruct the persisted
    header without depending on an agent run.
  - ``agentm.message.appended`` — every :class:`MessageAppendedEvent`.
    Body = the existing SessionEntry record verbatim
    (``{type, id, parent_id, timestamp, payload}``). Read by
    :class:`SessionManager._load` to rebuild the in-memory trajectory.
  - ``agentm.session.fingerprint`` — fingerprint at
    :class:`SessionReadyEvent`. Attributes carry the atom hashes +
    task metadata; consumers (catalog indexer, ``tool_query_traces``)
    filter on these.
  - ``agentm.turn.summary`` — per-turn aggregation at
    :class:`TurnEndEvent`. Attributes carry tool_calls / tool_call_count
    / tool_error_count / stop_reason / input_tokens / output_tokens.
    Read by ``tool_guard_watch`` and ``catalog/indexer``.
  - ``agentm.session.start`` / ``agentm.session.ready`` /
    ``agentm.session.end`` / ``agentm.extension.install`` /
    ``agentm.extension.reload`` / ``agentm.extension.unload`` /
    ``agentm.api.register`` / ``agentm.api.send_user_message`` /
    ``agentm.atom.reload`` / ``agentm.diagnostic`` — tier-1 lifecycle
    log records. Consumers like ``build_trace_map`` and the baseline
    grader pick them out by event name.
  - ``agentm.event.dispatch`` (one per ``EventBus.emit`` end) +
    ``agentm.handler.invoke`` (one per handler done) — both carry
    ``agentm.event.dispatch_id`` sourced from the :class:`Event` field,
    which the bus reassigns on every emit. Consumers join the fanout
    post-hoc on that id. See ``.claude/designs/single-event-log.md``
    "Bus correlation".

The on-disk wire format is one OTLP ``ResourceSpans`` / ``ResourceLogs``
element per line (PR-A). Atoms downstream of this writer never see the
legacy ``kind == "otel/span/v0"`` envelope — they walk the OTLP shape and
pattern-match on event name + span name.

``redact_prompts`` keeps the existing behaviour for ``api.send_user_message``
and the message bodies surfaced through the ``chat`` span. The
``BeforeSendToLlmEvent`` redaction shim from the old observability writer
is gone: the trajectory canonical copy is the ``agentm.message.appended``
log record stream, so ``before_send_to_llm`` produces no record at all.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import traceback
from typing import Any

from agentm.core.abi import (
    AgentEndEvent,
    BeforeSendToLlmEvent,
    EventBusObserver,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    SessionTelemetry,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.core.abi.events import (
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    BeforeAgentStartEvent,
    BeforeCompactEvent,
    ContextEvent,
    DiagnosticEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    ExtensionUnloadEvent,
    MessageAppendedEvent,
    SessionHeaderEmittedEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.extension import ExtensionAPI, Handler
from agentm.core.abi.messages import ToolCallBlock
from agentm.core.lib import redact_messages, to_jsonable
from agentm.extensions import ExtensionManifest
from agentm.extensions.discover import discover_builtin
from opentelemetry._logs import Logger as ApiLogger
from opentelemetry._logs import SeverityNumber
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
)


logger = logging.getLogger(__name__)


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
    config_schema={
        "type": "object",
        "properties": {
            "include_handler_records": {"type": "boolean"},
            "include_mutation_diff": {"type": "boolean"},
            "exclude_channels": {
                "type": "array",
                "items": {"type": "string"},
            },
            "redact_prompts": {
                "type": "boolean",
                "description": (
                    "When True (default), LLM request/response bodies and "
                    "api_send_user_message content are recorded as "
                    "{chars, sha256_prefix} stubs instead of full text. Set "
                    "False to capture raw prompt content for debugging "
                    "(only do this on a trusted machine — operators must "
                    "explicitly opt in to logging user-pasted secrets)."
                ),
            },
        },
        "additionalProperties": False,
    },
    requires=(),
)


_HANDLER_OWNER_ATTR = "_agentm_obs_owner"


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


def _to_otel_attr(value: Any) -> Any:
    """Coerce a Python value to something OTel attributes accept.

    OTel attribute values are restricted to str / bool / int / float / and
    homogeneous sequences thereof. We round-trip anything richer through
    JSON so the on-disk shape stays inspectable.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(to_jsonable(value), default=str, ensure_ascii=False)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    telemetry: SessionTelemetry = api.get_session_telemetry()
    tracer = telemetry.tracer
    log = telemetry.logger
    # PR-H: every log record we emit must carry ``agentm.session.id`` as an
    # attribute (it's no longer on the OTel resource), so the process-level
    # TracerProvider / LoggerProvider can fan many sessions into disjoint
    # per-session files based on this attribute alone.
    session_id_attr = api.session_id

    include_handlers = bool(config.get("include_handler_records", True))
    include_diff = bool(config.get("include_mutation_diff", True))
    user_excludes = config.get("exclude_channels")
    exclude_channels = frozenset(_DEFAULT_EXCLUDE_CHANNELS) | (
        frozenset(str(ch) for ch in user_excludes) if user_excludes else frozenset()
    )
    redact_prompts = bool(config.get("redact_prompts", True))

    # --- Open-span maps (paired start/end events) --------------------------
    open_invoke_agent: dict[str, Span] = {}  # session_id -> span
    open_chat_spans: dict[int, Span] = {}    # turn_id -> chat span
    open_tool_spans: dict[str, Span] = {}    # tool_call_id -> execute_tool span

    aggregator_state: dict[str, int] = {
        "turn_start_ns": 0,
        "previous_tool_errors": 0,
        "current_tool_errors": 0,
    }

    pending_diff_snapshots: list[tuple[str, Handler, Any, Any]] = []

    # --- Session bootstrap log record -------------------------------------

    session_start_ns = time.time_ns()
    _emit_log(
        log,
        "agentm.session.start",
        body={
            "session_id": api.session_id,
            "root_session_id": api.root_session_id,
            "parent_session_id": api.parent_session_id,
            "purpose": api.purpose,
            "scenario": api.scenario,
            "cwd": api.cwd,
        },
        attributes={
            "agentm.session.id": api.session_id,
            "agentm.session.root_id": api.root_session_id,
            "agentm.session.parent_id": api.parent_session_id or "",
            "agentm.session.purpose": api.purpose,
            "agentm.session.scenario": api.scenario or "",
        },
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
            except Exception:
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
        ) -> None:
            del result
            self._record_mutation(channel, handler, event, error)
            if not include_handlers or channel in exclude_channels:
                return
            owner = getattr(handler, _HANDLER_OWNER_ATTR, None)
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
            _emit_log(
                log,
                "agentm.handler.invoke",
                body=None,
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
            except Exception:
                payload = None
            if channel == BeforeSendToLlmEvent.CHANNEL and isinstance(payload, dict):
                payload.pop("messages", None)
            if redact_prompts and channel in _REDACTED_CHANNELS:
                payload = redact_messages(payload)
            if payload is not None:
                attrs["agentm.event.payload"] = _to_otel_attr(payload)
            _emit_log(
                log,
                "agentm.event.dispatch",
                body=None,
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
            except Exception:
                return
            if not diff:
                return
            _emit_log(
                log,
                "agentm.handler.mutated",
                body=None,
                attributes={
                    "agentm.session.id": api.session_id,
                    "agentm.event.channel": channel,
                    "agentm.event.dispatch_id": getattr(event, "dispatch_id", "")
                    or "",
                    "agentm.handler.name": _handler_label(handler),
                    "agentm.handler.mutations": _to_otel_attr(diff[:100]),
                    "agentm.handler.raised": error is not None,
                },
            )

    api.add_observer(_Observer())

    # --- Fingerprint state -------------------------------------------------

    discovered_builtin = discover_builtin()
    builtin_by_module_path = {
        entry.module_path: entry for entry in discovered_builtin.values()
    }
    loaded_builtin_module_paths: set[str] = set()
    active_atom_hashes: dict[str, str] = {}
    current_fingerprint: dict[str, Any] | None = None

    def _compute_loaded_fingerprint(*, scenario: str | None = None) -> dict[str, Any]:
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

    # --- Handlers (lifecycle + identity) ----------------------------------

    def _on_session_header(event: SessionHeaderEmittedEvent) -> None:
        _emit_log(
            log,
            "agentm.session.header",
            body=event.record,
            attributes={
                "agentm.session.id": api.session_id,
                "agentm.session.header.id": str(event.record.get("id", "")),
            },
        )

    def _on_message_appended(event: MessageAppendedEvent) -> None:
        _emit_log(
            log,
            "agentm.message.appended",
            body=event.record,
            attributes={
                "agentm.session.id": api.session_id,
                "agentm.message.id": str(event.record.get("id", "")),
                "agentm.message.parent_id": str(event.record.get("parent_id", "") or ""),
                "agentm.message.type": str(event.record.get("type", "")),
            },
        )

    def _on_session_ready(event: SessionReadyEvent) -> None:
        nonlocal current_fingerprint
        _emit_log(
            log,
            "agentm.session.ready",
            body={
                "session_id": event.session_id,
                "cwd": event.cwd,
                "tool_names": list(event.tool_names),
                "command_names": list(event.command_names),
                "extension_module_paths": list(event.extension_module_paths),
                "model": to_jsonable(event.model),
            },
            attributes={
                "agentm.session.id": event.session_id,
                "agentm.session.scenario": api.scenario or "",
                "agentm.session.tool_count": len(event.tool_names),
                "agentm.session.command_count": len(event.command_names),
                "agentm.session.extension_count": len(event.extension_module_paths),
            },
        )
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
        _emit_log(
            log,
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
        _emit_log(
            log,
            "agentm.extension.install",
            body={"config": to_jsonable(event.config)},
            attributes={
                "agentm.session.id": session_id_attr,
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
        _emit_log(
            log,
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
                "agentm.session.id": session_id_attr,
                "agentm.atom.name": event.name,
                "agentm.atom.new_hash": event.new_hash,
                "agentm.atom.old_hash": event.old_hash or "",
                "agentm.atom.trigger": event.trigger,
                "agentm.atom.tier": event.tier,
                "agentm.atom.error": event.error or "",
            },
            severity=(SeverityNumber.ERROR if event.error else SeverityNumber.INFO),
        )

    def _on_extension_unload(event: ExtensionUnloadEvent) -> None:
        _emit_log(
            log,
            "agentm.extension.unload",
            body={"trigger": event.trigger, "tier": event.tier},
            attributes={
                "agentm.session.id": session_id_attr,
                "agentm.extension.name": event.name,
                "agentm.extension.module_path": event.module_path,
                "agentm.extension.trigger": event.trigger,
                "agentm.extension.error": event.error or "",
            },
            severity=(SeverityNumber.ERROR if event.error else SeverityNumber.INFO),
        )

    def _on_api_register(event: ApiRegisterEvent) -> None:
        _emit_log(
            log,
            "agentm.api.register",
            body={"payload": to_jsonable(event.payload)},
            attributes={
                "agentm.session.id": session_id_attr,
                "agentm.api.kind": event.kind,
                "agentm.api.name": event.name,
                "agentm.api.extension": event.extension,
            },
        )

    def _on_api_send(event: ApiSendUserMessageEvent) -> None:
        raw_content: Any = to_jsonable(event.content)
        content_chars = (
            len(event.content) if isinstance(event.content, str) else None
        )
        if redact_prompts:
            stub_payload = redact_messages({"content": raw_content})
            attributes_content: Any = stub_payload["content"]
        else:
            attributes_content = raw_content
        _emit_log(
            log,
            "agentm.api.send_user_message",
            body={"content": attributes_content},
            attributes={
                "agentm.session.id": session_id_attr,
                "agentm.api.extension": event.extension,
                "agentm.api.content_chars": (
                    content_chars if content_chars is not None else -1
                ),
            },
        )

    def _on_diagnostic(event: DiagnosticEvent) -> None:
        sev = {
            "info": SeverityNumber.INFO,
            "warning": SeverityNumber.WARN,
            "error": SeverityNumber.ERROR,
        }.get(event.level, SeverityNumber.INFO)
        _emit_log(
            log,
            "agentm.diagnostic",
            body={"message": event.message, "level": event.level},
            attributes={
                "agentm.session.id": session_id_attr,
                "agentm.diagnostic.level": event.level,
                "agentm.diagnostic.source": event.source,
            },
            severity=sev,
        )

    # --- Span pairs: invoke_agent / chat / execute_tool -------------------

    def _on_before_agent_start(event: BeforeAgentStartEvent) -> None:
        del event
        existing = open_invoke_agent.pop(api.session_id, None)
        if existing is not None:
            existing.set_status(Status(StatusCode.UNSET, "preempted by new run"))
            existing.end()
        scenario_label = api.scenario or api.purpose or "agent"
        span = tracer.start_span(
            f"invoke_agent {scenario_label}",
            kind=SpanKind.INTERNAL,
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": scenario_label,
                "gen_ai.conversation.id": api.session_id,
                "agentm.session.id": api.session_id,
                "agentm.session.purpose": api.purpose,
                "agentm.session.scenario": api.scenario or "",
            },
        )
        open_invoke_agent[api.session_id] = span

    def _on_agent_end(event: AgentEndEvent) -> None:
        span = open_invoke_agent.pop(api.session_id, None)
        cause = event.cause
        cause_attrs: dict[str, Any] = {
            "agentm.agent.message_count": len(event.messages),
            "agentm.agent.cause_kind": type(cause).__name__,
            "agentm.agent.cause_final": type(cause).final,
        }
        cause_payload = to_jsonable(cause)
        if isinstance(cause_payload, dict):
            for key, value in cause_payload.items():
                cause_attrs[f"agentm.agent.cause.{key}"] = _to_otel_attr(value)
        if span is not None:
            for key, value in cause_attrs.items():
                span.set_attribute(key, value)
            span.set_status(Status(StatusCode.OK))
            span.end()
        _emit_log(
            log,
            "agentm.agent.end",
            body={"cause": cause_payload, "message_count": len(event.messages)},
            attributes={"agentm.session.id": session_id_attr, **cause_attrs},
        )

    def _on_llm_start(event: LlmRequestStartEvent) -> None:
        model_id = event.model_id or "unknown"
        span = tracer.start_span(
            f"chat {model_id}",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model_id,
                "gen_ai.conversation.id": api.session_id,
                "agentm.session.id": api.session_id,
                "agentm.turn.index": event.turn_index,
                "agentm.turn.id": event.turn_id,
                "agentm.llm.message_count": event.message_count,
                "agentm.llm.tool_count": event.tool_count,
                "agentm.llm.system_chars": event.system_chars,
            },
        )
        provider = api.provider
        if provider is not None:
            span.set_attribute("gen_ai.provider.name", provider.name)
        open_chat_spans[event.turn_id] = span

    def _on_llm_end(event: LlmRequestEndEvent) -> None:
        span = open_chat_spans.pop(event.turn_id, None)
        if span is None:
            return
        span.set_attribute("agentm.llm.chunk_count", event.chunk_count)
        span.set_attribute("agentm.llm.duration_ns", event.duration_ns)
        if event.error is not None:
            span.set_attribute("agentm.llm.error", event.error)
            span.set_status(Status(StatusCode.ERROR, event.error))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()

    def _on_tool_call(event: ToolCallEvent) -> None:
        args = dict(event.args)
        span = tracer.start_span(
            f"execute_tool {event.tool_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": event.tool_name,
                "gen_ai.tool.call.id": event.tool_call_id,
                "gen_ai.conversation.id": api.session_id,
                "agentm.session.id": api.session_id,
                "gen_ai.tool.call.arguments": _to_otel_attr(args),
            },
        )
        open_tool_spans[event.tool_call_id] = span

    def _on_tool_result(event: ToolResultEvent) -> None:
        span = open_tool_spans.pop(event.tool_call_id, None)
        if span is None:
            return
        result_payload = to_jsonable(event.result)
        span.set_attribute(
            "gen_ai.tool.call.result", _to_otel_attr(result_payload)
        )
        is_error = bool(getattr(event.result, "is_error", False))
        if is_error:
            span.set_attribute("agentm.tool.is_error", True)
            span.set_status(Status(StatusCode.ERROR, "tool result is_error=True"))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()

    # --- Per-turn aggregator (turn.summary log record) --------------------

    def _on_turn_start(event: TurnStartEvent) -> None:
        del event
        aggregator_state["turn_start_ns"] = time.time_ns()
        aggregator_state["previous_tool_errors"] = aggregator_state[
            "current_tool_errors"
        ]
        aggregator_state["current_tool_errors"] = 0

    def _on_tool_result_for_aggregator(event: ToolResultEvent) -> None:
        if getattr(event.result, "is_error", False):
            aggregator_state["current_tool_errors"] += 1

    def _on_turn_end(event: TurnEndEvent) -> None:
        end_ns = time.time_ns()
        tool_calls = [
            block.name
            for block in event.message.content
            if isinstance(block, ToolCallBlock)
        ]
        usage = getattr(event.message, "usage", None)
        body: dict[str, Any] = {
            "turn_index": event.turn_index,
            "turn_id": event.turn_id,
            "duration_ns": end_ns - aggregator_state["turn_start_ns"],
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "tool_error_count": aggregator_state["previous_tool_errors"],
            "stop_reason": event.message.stop_reason,
            "content_block_types": [
                getattr(c, "type", type(c).__name__) for c in event.message.content
            ],
        }
        attrs: dict[str, Any] = {
            "agentm.session.id": api.session_id,
            "agentm.turn.index": event.turn_index,
            "agentm.turn.id": event.turn_id,
            "agentm.turn.duration_ns": end_ns - aggregator_state["turn_start_ns"],
            "agentm.turn.tool_call_count": len(tool_calls),
            "agentm.turn.tool_error_count": aggregator_state["previous_tool_errors"],
            "agentm.turn.stop_reason": event.message.stop_reason or "",
        }
        if usage is not None:
            body["input_tokens"] = usage.input_tokens
            body["output_tokens"] = usage.output_tokens
            body["cache_read"] = usage.cache_read
            body["cache_write"] = usage.cache_write
            attrs["gen_ai.usage.input_tokens"] = usage.input_tokens
            attrs["gen_ai.usage.output_tokens"] = usage.output_tokens
            attrs["gen_ai.usage.cache_read_tokens"] = usage.cache_read
            attrs["gen_ai.usage.cache_write_tokens"] = usage.cache_write
        _emit_log(log, "agentm.turn.summary", body=body, attributes=attrs)
        aggregator_state["previous_tool_errors"] = 0

    # --- Session shutdown ---------------------------------------------------

    def _on_session_shutdown(_event: SessionShutdownEvent) -> None:
        end_ns = time.time_ns()
        for span in list(open_invoke_agent.values()):
            span.set_status(Status(StatusCode.UNSET, "session shutdown"))
            span.end()
        open_invoke_agent.clear()
        for span in list(open_chat_spans.values()):
            span.set_status(Status(StatusCode.UNSET, "session shutdown"))
            span.end()
        open_chat_spans.clear()
        for span in list(open_tool_spans.values()):
            span.set_status(Status(StatusCode.UNSET, "session shutdown"))
            span.end()
        open_tool_spans.clear()
        _emit_log(
            log,
            "agentm.session.end",
            body={
                "session_id": api.session_id,
                "root_session_id": api.root_session_id,
                "parent_session_id": api.parent_session_id,
                "purpose": api.purpose,
                "scenario": api.scenario,
                "duration_ns": end_ns - session_start_ns,
            },
            attributes={
                "agentm.session.id": api.session_id,
                "agentm.session.duration_ns": end_ns - session_start_ns,
            },
        )

    # --- Wire subscriptions -----------------------------------------------

    api.on(SessionHeaderEmittedEvent.CHANNEL, _on_session_header)
    api.on(MessageAppendedEvent.CHANNEL, _on_message_appended)
    api.on(SessionReadyEvent.CHANNEL, _on_session_ready)
    api.on(SessionShutdownEvent.CHANNEL, _on_session_shutdown)
    api.on(ExtensionInstallEvent.CHANNEL, _on_extension_install)
    api.on(ExtensionReloadEvent.CHANNEL, _on_extension_reload)
    api.on(ExtensionUnloadEvent.CHANNEL, _on_extension_unload)
    api.on(ApiRegisterEvent.CHANNEL, _on_api_register)
    api.on(ApiSendUserMessageEvent.CHANNEL, _on_api_send)
    api.on(DiagnosticEvent.CHANNEL, _on_diagnostic)

    api.on(BeforeAgentStartEvent.CHANNEL, _on_before_agent_start)
    api.on(AgentEndEvent.CHANNEL, _on_agent_end)
    api.on(LlmRequestStartEvent.CHANNEL, _on_llm_start)
    api.on(LlmRequestEndEvent.CHANNEL, _on_llm_end)
    api.on(ToolCallEvent.CHANNEL, _on_tool_call)
    api.on(ToolResultEvent.CHANNEL, _on_tool_result)

    api.on(TurnStartEvent.CHANNEL, _on_turn_start)
    api.on(ToolResultEvent.CHANNEL, _on_tool_result_for_aggregator)
    api.on(TurnEndEvent.CHANNEL, _on_turn_end)


def _emit_log(
    log: ApiLogger,
    event_name: str,
    *,
    body: Any = None,
    attributes: dict[str, Any] | None = None,
    severity: SeverityNumber = SeverityNumber.INFO,
) -> None:
    """Emit one log record through the per-session OTel ``Logger``.

    The PR-A ``FileLogExporter`` writes one OTLP ``ResourceLogs`` element
    per line to the per-session JSONL file. The ``event_name`` plays the
    role of the OTel "event name" — consumers (session_manager, indexer,
    tool_query_traces, llmharness CLI) filter on it instead of on the
    legacy ``kind`` field.
    """
    if attributes is not None:
        attributes = {
            key: _to_otel_attr(value) for key, value in attributes.items()
        }
    log.emit(
        body=to_jsonable(body) if body is not None else None,
        severity_number=severity,
        severity_text=severity.name,
        event_name=event_name,
        attributes=attributes,
    )


# Keep ``logger`` referenced (used implicitly by the SDK in error paths).
_ = logger
