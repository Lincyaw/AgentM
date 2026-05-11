"""Builtin OTel tracing atom — exports real OpenTelemetry spans over OTLP.

Mirrors :mod:`agentm.extensions.builtin.observability` (which writes JSONL),
but produces *true* OTel spans suitable for an OTel collector. The two atoms
are independent: enable one, the other, or both.

Span hierarchy (explicit parenting via OTel ``Context`` — no Python
current-span semantics, since the agent loop is async and a single tracer
serves many concurrent in-flight tool calls):

* ``agentm.session`` — root, opened in :func:`install`, closed on
  ``SessionShutdownEvent``.
* ``agentm.turn`` — child of session, opened on ``TurnStartEvent``,
  closed on ``TurnEndEvent``. Carries turn-level usage attributes.
* ``agentm.llm.request`` — child of current turn (or session), opened on
  ``LlmRequestStartEvent``, closed on ``LlmRequestEndEvent``. Uses GenAI
  semconv attributes (``gen_ai.system``, ``gen_ai.request.model``, …).
* ``agentm.tool.execute`` — child of current turn, paired by
  ``tool_call_id`` between ``ToolCallEvent`` and ``ToolResultEvent``.
* ``agentm.event:<channel>`` — generic event-dispatch span produced by the
  ``EventBusObserver``. Gives a complete timeline for low-traffic channels;
  ``stream_delta`` is excluded by default to avoid per-token bloat.
* Instant spans (start+end immediately): ``agentm.extension.install``,
  ``agentm.atom.reload``, ``agentm.api.register``.

Configuration honours the standard ``OTEL_*`` env vars (endpoint, headers,
service name, resource attributes). Atom config overrides env. If
:mod:`opentelemetry` is not installed, ``install`` emits a diagnostic and
becomes a no-op so the session still boots.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from agentm.core.abi import (
    EventBusObserver,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.core.abi.events import (
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    DiagnosticEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.extension import ExtensionAPI, Handler
from agentm.core.lib import to_jsonable
from agentm.extensions import ExtensionManifest

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from opentelemetry.context import Context
    from opentelemetry.trace import Tracer


MANIFEST = ExtensionManifest(
    name="otel_tracing",
    description=(
        "Export real OpenTelemetry spans over OTLP for a session's "
        "lifecycle, turns, LLM requests, tool calls, and event-bus "
        "dispatches. Pair with an OTel collector to centralize traces."
    ),
    registers=(
        "event:agent_start",
        "event:agent_end",
        "event:turn_start",
        "event:turn_end",
        "event:tool_call",
        "event:tool_result",
        "event:tool_error",
        "event:llm_request_start",
        "event:llm_request_end",
        "event:api_register",
        "event:api_send_user_message",
        "event:extension_install",
        "event:extension_reload",
        "event:session_ready",
        "event:session_shutdown",
        "event:diagnostic",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": (
                    "OTLP endpoint. Overrides OTEL_EXPORTER_OTLP_ENDPOINT. "
                    "gRPC default 'http://localhost:4317'; HTTP default "
                    "'http://localhost:4318'."
                ),
            },
            "protocol": {
                "type": "string",
                "enum": ["grpc", "http/protobuf"],
                "description": "OTLP transport (default 'grpc').",
            },
            "service_name": {
                "type": "string",
                "description": "service.name resource attribute (default 'agentm').",
            },
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "OTLP headers (auth tokens etc.).",
            },
            "resource_attributes": {
                "type": "object",
                "additionalProperties": {"type": ["string", "number", "boolean"]},
                "description": "Extra resource attributes merged into Resource.create().",
            },
            "insecure": {
                "type": "boolean",
                "description": "Use insecure channel for gRPC (default true).",
            },
            "include_event_spans": {
                "type": "boolean",
                "description": (
                    "Emit one span per EventBus dispatch (default true). "
                    "Disable for high-throughput sessions where only "
                    "lifecycle/turn/tool/llm spans are wanted."
                ),
            },
            "exclude_channels": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Channels whose event-dispatch spans are suppressed. "
                    "Defaults to {'stream_delta'} — one span per LLM token "
                    "is rarely useful."
                ),
            },
        },
        "additionalProperties": False,
    },
    requires=(),
)


# Process-global state: TracerProvider is set once per process. Subsequent
# sessions reuse it (each gets a fresh root span / trace_id automatically
# when start_span has no parent context).
_provider_lock = threading.Lock()
_provider_installed = False
_DEFAULT_EXCLUDE_CHANNELS: frozenset[str] = frozenset({StreamDeltaEvent.CHANNEL})


def _ensure_provider(
    *,
    service_name: str,
    endpoint: str | None,
    protocol: str,
    headers: dict[str, str] | None,
    insecure: bool,
    resource_attributes: dict[str, Any] | None,
) -> bool:
    """Install a TracerProvider + OTLP exporter once per process.

    Returns ``True`` on success, ``False`` if the SDK is unavailable. If a
    non-default TracerProvider is already installed (e.g. by application
    code or ``opentelemetry-instrument``), we leave it alone and just emit
    spans through whatever is wired up.
    """

    global _provider_installed
    with _provider_lock:
        if _provider_installed:
            return True
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            return False

        existing = trace.get_tracer_provider()
        # ProxyTracerProvider is the no-op default. Anything else means the
        # host has already configured tracing — respect that, just mark us
        # installed and let spans flow through it.
        if type(existing).__name__ != "ProxyTracerProvider":
            _provider_installed = True
            return True

        resource_kwargs: dict[str, Any] = {"service.name": service_name}
        if resource_attributes:
            resource_kwargs.update(resource_attributes)
        resource = Resource.create(resource_kwargs)

        exporter: Any
        if protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcOTLPSpanExporter,
            )

            exporter = GrpcOTLPSpanExporter(
                endpoint=endpoint,
                headers=headers,
                insecure=insecure,
            )

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _provider_installed = True
        return True


def _now_ns() -> int:
    return time.time_ns()


def _handler_label(handler: Handler) -> str:
    mod = getattr(handler, "__module__", None) or "<unknown>"
    qual = getattr(handler, "__qualname__", None) or repr(handler)
    return f"{mod}.{qual}"


class _OtelObserver(EventBusObserver):
    """Wraps every EventBus dispatch in a span. Suppressed channels still
    cycle frames so parent tracking stays consistent for siblings.
    """

    def __init__(
        self,
        tracer: Tracer,
        get_parent_ctx: Any,
        exclude_channels: frozenset[str],
    ) -> None:
        self._tracer = tracer
        self._get_parent_ctx = get_parent_ctx  # callable -> Context
        self._exclude = exclude_channels
        # Stack of (span | None, token, suppressed). Parent for each new
        # emit is the innermost non-suppressed ancestor span; if none,
        # falls back to the session-root context.
        self._stack: list[tuple[Any, Any, bool]] = []

    def _innermost_parent(self) -> Context:
        from opentelemetry import trace as _trace

        for span, _ctx, suppressed in reversed(self._stack):
            if not suppressed and span is not None:
                return _trace.set_span_in_context(span)
        return self._get_parent_ctx()

    def on_emit_start(self, channel: str, event: Any) -> None:
        suppressed = channel in self._exclude
        if suppressed:
            self._stack.append((None, None, True))
            return
        parent_ctx = self._innermost_parent()
        span = self._tracer.start_span(
            f"agentm.event:{channel}",
            context=parent_ctx,
            attributes={"agentm.channel": channel},
        )
        self._stack.append((span, parent_ctx, False))

    def on_handler_start(self, channel: str, handler: Handler, event: Any) -> None:
        # Per-handler spans are created on completion so we know the real
        # duration; see :meth:`on_handler_done`. We deliberately don't open
        # a span here because async handlers can interleave and we'd need a
        # per-(channel, handler) key to match start/end.
        del channel, handler, event

    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
    ) -> None:
        # Emit one ``agentm.handler.invoke`` span per handler invocation,
        # parented to the current emit span. Mirrors the JSONL
        # observability atom's ``handler.invoke`` records so OTel and
        # JSONL views agree on cardinality.
        if not self._stack:
            return
        span, _ctx, suppressed = self._stack[-1]
        if suppressed or span is None:
            return
        from opentelemetry import trace as _trace
        from opentelemetry.trace import Status, StatusCode

        end_time_ns = _now_ns()
        start_time_ns = end_time_ns - duration_ns
        owner = getattr(handler, "_agentm_obs_owner", None)
        parent_ctx = _trace.set_span_in_context(span)
        attrs: dict[str, Any] = {
            "agentm.channel": channel,
            "agentm.handler": _handler_label(handler),
            "agentm.handler.duration_ns": duration_ns,
        }
        if owner:
            attrs["agentm.extension"] = owner
        h_span = self._tracer.start_span(
            f"agentm.handler:{channel}",
            context=parent_ctx,
            start_time=start_time_ns,
            attributes=attrs,
        )
        if error is not None:
            h_span.set_status(Status(StatusCode.ERROR, repr(error)))
            h_span.set_attribute("agentm.error", repr(error))
        # ``result`` is included as a stringified preview — handlers can
        # return arbitrary objects (None, dicts, lists), keep it bounded.
        try:
            result_str = repr(result)
            if len(result_str) > 1024:
                result_str = result_str[:1024] + "...<truncated>"
            h_span.set_attribute("agentm.handler.result", result_str)
        except Exception:
            pass
        h_span.end(end_time=end_time_ns)

    def on_emit_end(self, channel: str, event: Any, results: list[Any]) -> None:
        if not self._stack:
            return
        span, _ctx, suppressed = self._stack.pop()
        if suppressed or span is None:
            return
        span.set_attribute("agentm.handler_count", len(results))
        span.end()


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    service_name = str(config.get("service_name", "agentm"))
    endpoint = config.get("endpoint") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    protocol = str(
        config.get("protocol")
        or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
        or "grpc"
    )
    headers_cfg = config.get("headers")
    headers: dict[str, str] | None = (
        {str(k): str(v) for k, v in headers_cfg.items()} if headers_cfg else None
    )
    insecure = bool(config.get("insecure", True))
    resource_attributes = config.get("resource_attributes") or {}
    include_event_spans = bool(config.get("include_event_spans", True))
    user_excludes = config.get("exclude_channels")
    exclude_channels = frozenset(_DEFAULT_EXCLUDE_CHANNELS) | (
        frozenset(str(ch) for ch in user_excludes) if user_excludes else frozenset()
    )

    ok = _ensure_provider(
        service_name=service_name,
        endpoint=endpoint,
        protocol=protocol,
        headers=headers,
        insecure=insecure,
        resource_attributes=resource_attributes,
    )
    if not ok:
        logger.warning(
            "otel_tracing: opentelemetry SDK not installed; "
            "install with `uv sync --extra otel`. Atom is a no-op for this session."
        )
        return

    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    tracer = trace.get_tracer("agentm", "0.1.0")

    # --- session root span ----------------------------------------------------
    session_attrs: dict[str, Any] = {
        "agentm.session_id": api.session_id,
        "agentm.cwd": api.cwd,
    }
    session_span = tracer.start_span(
        "agentm.session",
        attributes=session_attrs,
        kind=SpanKind.INTERNAL,
    )
    session_ctx = trace.set_span_in_context(session_span)

    # Per-session live-span tables. Tool calls and turns can interleave
    # asynchronously, so we cannot rely on the OTel current-span semantics
    # — we look spans up explicitly by key.
    turn_spans: dict[int, Any] = {}
    tool_spans: dict[str, Any] = {}
    llm_spans: dict[int, Any] = {}
    extension_install_spans: dict[str, Any] = {}

    def _current_turn_ctx() -> Context:
        # Use the most recent open turn span (highest turn_index) as parent;
        # fall back to the session if no turn is open.
        if turn_spans:
            latest = max(turn_spans)
            return trace.set_span_in_context(turn_spans[latest])
        return session_ctx

    if include_event_spans:
        observer = _OtelObserver(
            tracer,
            get_parent_ctx=lambda: _current_turn_ctx(),
            exclude_channels=exclude_channels,
        )
        api.add_observer(observer)

    # --- turn lifecycle -------------------------------------------------------

    def _on_turn_start(event: TurnStartEvent) -> None:
        span = tracer.start_span(
            "agentm.turn",
            context=session_ctx,
            attributes={
                "agentm.turn_index": event.turn_index,
                "agentm.turn_id": getattr(event, "turn_id", 0),
            },
        )
        turn_spans[event.turn_index] = span

    def _on_turn_end(event: TurnEndEvent) -> None:
        span = turn_spans.pop(event.turn_index, None)
        if span is None:
            return
        try:
            stop_reason = getattr(event.message, "stop_reason", None)
            if stop_reason is not None:
                span.set_attribute("agentm.turn.stop_reason", str(stop_reason))
            usage = getattr(event.message, "usage", None)
            if usage is not None:
                span.set_attribute("gen_ai.usage.input_tokens", int(usage.input_tokens))
                span.set_attribute("gen_ai.usage.output_tokens", int(usage.output_tokens))
                span.set_attribute("agentm.cache.read", int(usage.cache_read))
                span.set_attribute("agentm.cache.write", int(usage.cache_write))
        finally:
            span.end()

    # --- LLM request lifecycle -----------------------------------------------

    def _on_llm_start(event: LlmRequestStartEvent) -> None:
        parent_ctx = _current_turn_ctx()
        span = tracer.start_span(
            "agentm.llm.request",
            context=parent_ctx,
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.system": "agentm",
                "gen_ai.request.model": event.model_id or "",
                "agentm.turn_index": event.turn_index,
                "agentm.message_count": event.message_count,
                "agentm.tool_count": event.tool_count,
                "agentm.system_chars": event.system_chars,
            },
        )
        llm_spans[event.turn_index] = span

    def _on_llm_end(event: LlmRequestEndEvent) -> None:
        span = llm_spans.pop(event.turn_index, None)
        if span is None:
            return
        try:
            span.set_attribute("agentm.chunk_count", event.chunk_count)
            span.set_attribute("agentm.duration_ns", event.duration_ns)
            if event.error:
                span.set_status(Status(StatusCode.ERROR, event.error))
                span.set_attribute("agentm.error", event.error)
        finally:
            span.end()

    # --- tool execution lifecycle --------------------------------------------

    def _on_tool_call(event: ToolCallEvent) -> None:
        parent_ctx = _current_turn_ctx()
        span = tracer.start_span(
            "agentm.tool.execute",
            context=parent_ctx,
            attributes={
                "agentm.tool.name": event.tool_name,
                "agentm.tool.call_id": event.tool_call_id,
            },
        )
        # Args may be large; bound the serialized payload.
        try:
            args_json = to_jsonable(event.args)
            args_str = str(args_json)
            if len(args_str) > 4096:
                args_str = args_str[:4096] + "...<truncated>"
            span.set_attribute("agentm.tool.args", args_str)
        except Exception:
            pass
        tool_spans[event.tool_call_id] = span
        last_tool_span_by_name[event.tool_name] = span

    def _on_tool_result(event: ToolResultEvent) -> None:
        span = tool_spans.pop(event.tool_call_id, None)
        if last_tool_span_by_name.get(event.tool_name) is span:
            last_tool_span_by_name.pop(event.tool_name, None)
        if span is None:
            return
        try:
            is_error = bool(getattr(event.result, "is_error", False))
            span.set_attribute("agentm.tool.is_error", is_error)
            if is_error:
                span.set_status(Status(StatusCode.ERROR, "tool reported error"))
            # Bounded preview of tool result content. Real values can be
            # huge (bash stdout, file dumps) so cap aggressively.
            try:
                content = getattr(event.result, "content", None)
                if content is not None:
                    preview = str(to_jsonable(content))
                    if len(preview) > 4096:
                        preview = preview[:4096] + "...<truncated>"
                    span.set_attribute("agentm.tool.result", preview)
                    span.set_attribute("agentm.tool.result_chars", len(preview))
            except Exception:
                pass
        finally:
            span.end()

    # Most recently opened span keyed by tool_name; used to attach
    # ToolErrorEvent metadata to the matching tool span. ToolErrorEvent
    # itself doesn't carry tool_call_id, but the loop emits it between
    # ToolCallEvent and ToolResultEvent for the same call, so the latest
    # open span for that tool_name is the right target.
    last_tool_span_by_name: dict[str, Any] = {}

    def _on_tool_error(event: ToolErrorEvent) -> None:
        span = last_tool_span_by_name.get(event.tool_name)
        if span is None:
            return
        span.set_attribute("agentm.tool.error_kind", event.kind)
        span.set_attribute("agentm.tool.error_reason", event.reason)
        if event.exception is not None:
            span.set_attribute("agentm.tool.exception", repr(event.exception))

    # --- one-shot / instant spans --------------------------------------------

    def _on_extension_install(event: ExtensionInstallEvent) -> None:
        # Pair phase=start/end on module_path so install duration is
        # captured as a real span rather than an instant.
        if event.phase == "start":
            span = tracer.start_span(
                f"agentm.extension.install:{event.module_path}",
                context=session_ctx,
                attributes={
                    "agentm.extension.module_path": event.module_path,
                },
            )
            extension_install_spans[event.module_path] = span
            return

        span = extension_install_spans.pop(event.module_path, None)
        if span is None:
            return
        try:
            span.set_attribute("agentm.extension.duration_ns", event.duration_ns)
            if event.error:
                span.set_status(Status(StatusCode.ERROR, event.error))
                span.set_attribute("agentm.error", event.error)
        finally:
            span.end()

    def _on_extension_reload(event: ExtensionReloadEvent) -> None:
        span = tracer.start_span(
            f"agentm.atom.reload:{event.name}",
            context=session_ctx,
            attributes={
                "agentm.atom.name": event.name,
                "agentm.atom.old_hash": event.old_hash or "",
                "agentm.atom.new_hash": event.new_hash or "",
                "agentm.atom.trigger": event.trigger or "",
                "agentm.atom.tier": event.tier or "",
            },
        )
        if event.error:
            span.set_status(Status(StatusCode.ERROR, event.error))
            span.set_attribute("agentm.error", event.error)
        span.end()

    def _on_api_register(event: ApiRegisterEvent) -> None:
        span = tracer.start_span(
            f"agentm.api.register:{event.kind}:{event.name}",
            context=session_ctx,
            attributes={
                "agentm.api.kind": event.kind,
                "agentm.api.name": event.name,
                "agentm.api.extension": event.extension or "",
            },
        )
        span.end()

    def _on_api_send(event: ApiSendUserMessageEvent) -> None:
        chars = len(event.content) if isinstance(event.content, str) else 0
        span = tracer.start_span(
            "agentm.api.send_user_message",
            context=session_ctx,
            attributes={
                "agentm.api.extension": event.extension or "",
                "agentm.message.chars": chars,
            },
        )
        span.end()

    def _on_diagnostic(event: DiagnosticEvent) -> None:
        # Emit as an event on the session span so diagnostics aren't lost
        # but don't proliferate trees.
        attrs: dict[str, Any] = {
            "agentm.diagnostic.level": event.level,
            "agentm.diagnostic.source": event.source,
            "agentm.diagnostic.message": event.message,
        }
        session_span.add_event("agentm.diagnostic", attributes=attrs)
        if event.level == "error":
            session_span.set_status(Status(StatusCode.ERROR, event.message))

    def _on_ready(event: SessionReadyEvent) -> None:
        session_span.set_attribute(
            "agentm.tool_names", ",".join(event.tool_names or [])
        )
        if getattr(event, "model", None) is not None:
            model_id = getattr(event.model, "id", None) or getattr(
                event.model, "name", None
            )
            if model_id:
                session_span.set_attribute("gen_ai.request.model", str(model_id))

    def _on_shutdown(_: SessionShutdownEvent) -> None:
        # Close any still-open spans defensively so the trace tree is
        # well-formed even on abnormal termination paths.
        for span in list(tool_spans.values()):
            span.set_status(Status(StatusCode.ERROR, "session shutdown before tool_result"))
            span.end()
        tool_spans.clear()
        for span in list(llm_spans.values()):
            span.set_status(Status(StatusCode.ERROR, "session shutdown mid-request"))
            span.end()
        llm_spans.clear()
        for span in list(turn_spans.values()):
            span.set_status(Status(StatusCode.ERROR, "session shutdown mid-turn"))
            span.end()
        turn_spans.clear()
        for span in list(extension_install_spans.values()):
            span.end()
        extension_install_spans.clear()
        session_span.end()
        # Force-flush so the collector sees everything before the process
        # exits. The provider may not be ours (host may have installed one);
        # call only if it implements force_flush.
        provider = trace.get_tracer_provider()
        flush = getattr(provider, "force_flush", None)
        if callable(flush):
            try:
                flush(timeout_millis=2_000)
            except Exception:
                logger.debug("otel_tracing: force_flush failed", exc_info=True)

    api.on(TurnStartEvent.CHANNEL, _on_turn_start)
    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
    api.on(LlmRequestStartEvent.CHANNEL, _on_llm_start)
    api.on(LlmRequestEndEvent.CHANNEL, _on_llm_end)
    api.on(ToolCallEvent.CHANNEL, _on_tool_call)
    api.on(ToolResultEvent.CHANNEL, _on_tool_result)
    api.on(ToolErrorEvent.CHANNEL, _on_tool_error)
    api.on(ExtensionInstallEvent.CHANNEL, _on_extension_install)
    api.on(ExtensionReloadEvent.CHANNEL, _on_extension_reload)
    api.on(ApiRegisterEvent.CHANNEL, _on_api_register)
    api.on(ApiSendUserMessageEvent.CHANNEL, _on_api_send)
    api.on(DiagnosticEvent.CHANNEL, _on_diagnostic)
    api.on(SessionReadyEvent.CHANNEL, _on_ready)
    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)
