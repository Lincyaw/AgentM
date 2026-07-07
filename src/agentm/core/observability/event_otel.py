"""Registry-based OTel translators for kernel events.

Each concrete event subclass has a translator function registered via
:func:`agentm.core.lib.otel_dispatch.register_otel`. The observability
atom dispatches through :func:`~agentm.core.lib.otel_dispatch.dispatch_otel`
instead of calling ``event.to_otel(telemetry)`` directly.

Importing this module is a side-effect: it populates the registry. The
``runtime/__init__.py`` imports it unconditionally so the registry is
filled before any session starts.
"""

from __future__ import annotations

import time
from typing import Any

from opentelemetry._logs import SeverityNumber
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    set_span_in_context,
)

from agentm.core.abi.events import (
    AgentEndEvent,
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    BeforeAgentStartEvent,
    DiagnosticEvent,
    ExtensionUnloadEvent,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    MessageAppendedEvent,
    SessionHeaderEmittedEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.core.abi.messages import (
    AgentMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)
from agentm.core.observability.otel_dispatch import register_otel

# Span-kind keys for ``SessionTelemetry.span_tracker``.
_SPAN_INVOKE_AGENT = "invoke_agent"
_SPAN_TURN = "turn"
_SPAN_CHAT = "chat"
_SPAN_EXECUTE_TOOL = "execute_tool"


def _remote_parent_context(root_session_id: str, parent_session_id: str) -> Any:
    """Build an OTel context whose current span is a *remote* parent carrying
    ``trace_id=root_session_id`` and ``span_id=parent_session_id``.

    Returns ``None`` (fresh trace) unless BOTH ids are valid lowercase hex of
    OTel width (trace_id 32, span_id 16) and non-zero.
    """
    if not root_session_id or not parent_session_id:
        return None
    if len(root_session_id) != 32 or len(parent_session_id) != 16:
        return None
    try:
        trace_id = int(root_session_id, 16)
        span_id = int(parent_session_id, 16)
    except (ValueError, TypeError):
        return None
    if trace_id == 0 or span_id == 0:
        return None
    parent = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return set_span_in_context(NonRecordingSpan(parent))


def _before_agent_start_to_otel(
    self: "BeforeAgentStartEvent", telemetry: Any
) -> None:
    """Open the ``invoke_agent <scenario>`` INTERNAL span for this session."""
    session_id = telemetry.session_id
    preempted = telemetry.pop_span(_SPAN_INVOKE_AGENT, session_id)
    if preempted is not None:
        preempted.set_status(Status(StatusCode.UNSET, "preempted by new run"))
        preempted.end()
    scenario_label = telemetry.obs_scenario or telemetry.obs_purpose or "agent"
    parent_ctx = _remote_parent_context(
        telemetry.obs_root_session_id, telemetry.obs_parent_session_id
    )
    attrs: dict[str, str | None] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": scenario_label,
        "gen_ai.conversation.id": session_id,
        "agentm.session.id": session_id,
        "agentm.session.root_id": telemetry.obs_root_session_id,
        "agentm.session.parent_id": telemetry.obs_parent_session_id,
        "agentm.session.purpose": telemetry.obs_purpose,
        "agentm.session.scenario": telemetry.obs_scenario,
    }
    if self.system:
        attrs["agentm.system_prompt"] = self.system
    span = telemetry.tracer.start_span(
        f"invoke_agent {scenario_label}",
        context=parent_ctx,
        kind=SpanKind.INTERNAL,
        attributes=attrs,
    )
    telemetry.open_span(_SPAN_INVOKE_AGENT, session_id, span)


def _agent_end_to_otel(
    self: "AgentEndEvent", telemetry: Any
) -> None:
    """Close the matching ``invoke_agent`` span and emit ``agentm.agent.end``."""
    session_id = telemetry.session_id
    trailing_turn_span = telemetry.current_turn_span
    if trailing_turn_span is not None:
        trailing_turn_span.set_status(Status(StatusCode.OK))
        trailing_turn_span.end()
        stale_tracker_keys = [
            tracker_key
            for tracker_key, tracked in telemetry.span_tracker.items()
            if tracked is trailing_turn_span
        ]
        for tracker_key in stale_tracker_keys:
            telemetry.span_tracker.pop(tracker_key, None)
        telemetry.current_turn_span = None
    span = telemetry.pop_span(_SPAN_INVOKE_AGENT, session_id)
    cause = self.cause
    cause_attrs: dict[str, Any] = {
        "agentm.agent.message_count": len(self.messages),
        "agentm.agent.cause_kind": type(cause).__name__,
        "agentm.agent.cause_final": type(cause).final,
    }
    from agentm.core.lib import to_jsonable

    cause_payload = to_jsonable(cause)
    if isinstance(cause_payload, dict):
        for key, value in cause_payload.items():
            cause_attrs[f"agentm.agent.cause.{key}"] = telemetry.to_otel_attr(value)
    if span is not None:
        for key, value in cause_attrs.items():
            span.set_attribute(key, value)
        span.set_status(Status(StatusCode.OK))
        span.end()
    telemetry.emit_log(
        "agentm.agent.end",
        body={"cause": cause_payload, "message_count": len(self.messages)},
        attributes={"agentm.session.id": session_id, **cause_attrs},
    )


def _llm_request_start_to_otel(
    self: "LlmRequestStartEvent", telemetry: Any
) -> None:
    """Open the ``chat <model>`` CLIENT span for one LLM call."""
    model_id = self.model_id or "unknown"
    turn_span = telemetry.current_turn_span
    parent_ctx = set_span_in_context(turn_span) if turn_span is not None else None
    span = telemetry.tracer.start_span(
        f"chat {model_id}",
        kind=SpanKind.CLIENT,
        context=parent_ctx,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model_id,
            "gen_ai.conversation.id": telemetry.session_id,
            "agentm.session.id": telemetry.session_id,
            "agentm.turn.index": self.turn_index,
            "agentm.turn.id": self.turn_id,
            "agentm.llm.message_count": self.message_count,
            "agentm.llm.tool_count": self.tool_count,
            "agentm.llm.system_chars": self.system_chars,
        },
    )
    if telemetry.obs_provider_name:
        span.set_attribute("gen_ai.provider.name", telemetry.obs_provider_name)
    telemetry.open_span(_SPAN_CHAT, str(self.turn_id), span)
    if self.system_text is not None:
        telemetry.emit_log(
            "agentm.llm.system_prompt",
            body={
                "turn_index": self.turn_index,
                "turn_id": self.turn_id,
                "text": self.system_text,
            },
            attributes={
                "agentm.session.id": telemetry.session_id,
                "agentm.turn.index": self.turn_index,
                "agentm.turn.id": self.turn_id,
                "agentm.llm.system_chars": self.system_chars,
            },
        )


def _llm_request_end_to_otel(
    self: "LlmRequestEndEvent", telemetry: Any
) -> None:
    """Close the ``chat`` span; stamp duration / error on the way out."""
    span = telemetry.pop_span(_SPAN_CHAT, str(self.turn_id))
    if span is None:
        return
    span.set_attribute("agentm.llm.chunk_count", self.chunk_count)
    span.set_attribute("agentm.llm.duration_ns", self.duration_ns)
    if self.error is not None:
        span.set_attribute("agentm.llm.error", self.error)
        span.set_status(Status(StatusCode.ERROR, self.error))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def _tool_call_to_otel(
    self: "ToolCallEvent", telemetry: Any
) -> None:
    """Open the ``execute_tool <name>`` span and stamp the call arguments."""
    args = dict(self.args)
    turn_span = telemetry.current_turn_span
    parent_ctx = set_span_in_context(turn_span) if turn_span is not None else None
    span = telemetry.tracer.start_span(
        f"execute_tool {self.tool_name}",
        kind=SpanKind.INTERNAL,
        context=parent_ctx,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": self.tool_name,
            "gen_ai.tool.call.id": self.tool_call_id,
            "gen_ai.conversation.id": telemetry.session_id,
            "agentm.session.id": telemetry.session_id,
            "gen_ai.tool.call.arguments": telemetry.to_otel_attr(args),
        },
    )
    telemetry.open_span(_SPAN_EXECUTE_TOOL, self.tool_call_id, span)


def _tool_result_to_otel(
    self: "ToolResultEvent", telemetry: Any
) -> None:
    """Close the matching ``execute_tool`` span, stamp the result."""
    from agentm.core.lib import to_jsonable

    if getattr(self.result, "is_error", False):
        telemetry.turn_state["current_tool_errors"] += 1
    span = telemetry.pop_span(_SPAN_EXECUTE_TOOL, self.tool_call_id)
    if span is None:
        return
    result_payload = to_jsonable(self.result)
    span.set_attribute(
        "gen_ai.tool.call.result", telemetry.to_otel_attr(result_payload)
    )
    is_error = bool(getattr(self.result, "is_error", False))
    if is_error:
        span.set_attribute("agentm.tool.is_error", True)
        span.set_status(Status(StatusCode.ERROR, "tool result is_error=True"))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def _turn_start_to_otel(
    self: "TurnStartEvent", telemetry: Any
) -> None:
    """Open the ``agentm.turn`` INTERNAL span and rotate per-turn state."""
    telemetry.turn_state["turn_start_ns"] = time.time_ns()
    telemetry.turn_state["previous_tool_errors"] = telemetry.turn_state[
        "current_tool_errors"
    ]
    telemetry.turn_state["current_tool_errors"] = 0
    prev_turn_span = telemetry.current_turn_span
    if prev_turn_span is not None:
        prev_turn_span.set_status(Status(StatusCode.OK))
        prev_turn_span.end()
        stale_keys = [
            key
            for key, span in telemetry.span_tracker.items()
            if span is prev_turn_span
        ]
        for key in stale_keys:
            telemetry.span_tracker.pop(key, None)
    parent_span = telemetry.span_tracker.get(
        (_SPAN_INVOKE_AGENT, telemetry.session_id)
    )
    parent_ctx = set_span_in_context(parent_span) if parent_span is not None else None
    span = telemetry.tracer.start_span(
        "agentm.turn",
        kind=SpanKind.INTERNAL,
        context=parent_ctx,
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.turn.index": self.turn_index,
            "agentm.turn.id": self.turn_id,
        },
    )
    telemetry.open_span(_SPAN_TURN, str(self.turn_id), span)
    telemetry.current_turn_span = span


def _message_chars(msg: AgentMessage) -> int:
    total = 0
    for block in msg.content:
        if isinstance(block, TextContent):
            total += len(block.text)
        elif isinstance(block, ToolCallBlock):
            total += len(block.name) + len(str(block.arguments))
        elif isinstance(block, ToolResultBlock):
            for sub in block.content:
                if isinstance(sub, TextContent):
                    total += len(sub.text)
    return total


def _context_breakdown(
    messages: tuple[AgentMessage, ...],
) -> dict[str, Any] | None:
    if not messages:
        return None
    call_id_to_name: dict[str, str] = {}
    system_chars = 0
    user_chars = 0
    assistant_chars = 0
    tool_result_by_name: dict[str, int] = {}
    tool_result_total = 0

    for msg in messages:
        chars = _message_chars(msg)
        role = getattr(msg, "role", "")
        if role == "system":
            system_chars += chars
        elif role == "user":
            user_chars += chars
        elif role == "assistant":
            assistant_chars += chars
            for block in msg.content:
                if isinstance(block, ToolCallBlock):
                    call_id_to_name[block.id] = block.name
        elif isinstance(msg, ToolResultMessage):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    block_chars = sum(
                        len(sub.text)
                        for sub in block.content
                        if isinstance(sub, TextContent)
                    )
                    name = call_id_to_name.get(block.tool_call_id, "unknown")
                    tool_result_by_name[name] = (
                        tool_result_by_name.get(name, 0) + block_chars
                    )
                    tool_result_total += block_chars

    total = system_chars + user_chars + assistant_chars + tool_result_total
    if total == 0:
        return None

    sorted_tools = dict(
        sorted(tool_result_by_name.items(), key=lambda kv: -kv[1])
    )
    return {
        "total_chars": total,
        "system": system_chars,
        "user": user_chars,
        "assistant": assistant_chars,
        "tool_result": tool_result_total,
        "tool_result_by_name": sorted_tools,
    }


def _turn_end_to_otel(
    self: "TurnEndEvent", telemetry: Any
) -> None:
    """Emit one ``agentm.turn.summary`` log record per completed turn."""
    end_ns = time.time_ns()
    tool_calls = [
        block.name
        for block in self.message.content
        if isinstance(block, ToolCallBlock)
    ]
    usage = getattr(self.message, "usage", None)
    duration_ns = end_ns - telemetry.turn_state["turn_start_ns"]
    body: dict[str, Any] = {
        "turn_index": self.turn_index,
        "turn_id": self.turn_id,
        "duration_ns": duration_ns,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "tool_error_count": telemetry.turn_state["previous_tool_errors"],
        "stop_reason": self.message.stop_reason,
        "content_block_types": [
            getattr(c, "type", type(c).__name__) for c in self.message.content
        ],
    }
    attrs: dict[str, Any] = {
        "agentm.session.id": telemetry.session_id,
        "agentm.turn.index": self.turn_index,
        "agentm.turn.id": self.turn_id,
        "agentm.turn.duration_ns": duration_ns,
        "agentm.turn.tool_call_count": len(tool_calls),
        "agentm.turn.tool_error_count": telemetry.turn_state[
            "previous_tool_errors"
        ],
        "agentm.turn.stop_reason": self.message.stop_reason or "",
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
    turn_span = telemetry.span_tracker.get((_SPAN_TURN, str(self.turn_id)))
    if turn_span is not None:
        turn_span.set_attribute("agentm.turn.duration_ns", duration_ns)
        turn_span.set_attribute("agentm.turn.tool_call_count", len(tool_calls))
        turn_span.set_attribute(
            "agentm.turn.tool_error_count",
            telemetry.turn_state["previous_tool_errors"],
        )
        turn_span.set_attribute(
            "agentm.turn.stop_reason", self.message.stop_reason or ""
        )
        if usage is not None:
            turn_span.set_attribute(
                "gen_ai.usage.input_tokens", usage.input_tokens
            )
            turn_span.set_attribute(
                "gen_ai.usage.output_tokens", usage.output_tokens
            )
            turn_span.set_attribute(
                "gen_ai.usage.cache_read_tokens", usage.cache_read
            )
            turn_span.set_attribute(
                "gen_ai.usage.cache_write_tokens", usage.cache_write
            )
    context_breakdown = _context_breakdown(self.messages)
    if context_breakdown:
        body["context_breakdown"] = context_breakdown
    telemetry.emit_log("agentm.turn.summary", body=body, attributes=attrs)
    telemetry.turn_state["previous_tool_errors"] = 0


def _session_header_to_otel(
    self: "SessionHeaderEmittedEvent", telemetry: Any
) -> None:
    """Emit ``agentm.session.header`` — body is the SessionHeader dict verbatim."""
    telemetry.emit_log(
        "agentm.session.header",
        body=self.record,
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.session.header.id": str(self.record.get("id", "")),
        },
    )


def _message_appended_to_otel(
    self: "MessageAppendedEvent", telemetry: Any
) -> None:
    """Emit ``agentm.message.appended`` — SessionEntry dict verbatim in body."""
    telemetry.emit_log(
        "agentm.message.appended",
        body=self.record,
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.message.id": str(self.record.get("id", "")),
            "agentm.message.parent_id": str(self.record.get("parent_id", "") or ""),
            "agentm.message.type": str(self.record.get("type", "")),
        },
    )


def _api_register_to_otel(
    self: "ApiRegisterEvent", telemetry: Any
) -> None:
    from agentm.core.lib import to_jsonable

    telemetry.emit_log(
        "agentm.api.register",
        body={"payload": to_jsonable(self.payload)},
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.api.kind": self.kind,
            "agentm.api.name": self.name,
            "agentm.api.extension": self.extension,
        },
    )


def _api_send_user_message_to_otel(
    self: "ApiSendUserMessageEvent", telemetry: Any
) -> None:
    """Emit ``agentm.api.send_user_message`` with optional prompt redaction."""
    from agentm.core.lib import redact_messages, to_jsonable

    raw_content: Any = to_jsonable(self.content)
    content_chars = len(self.content) if isinstance(self.content, str) else None
    if telemetry.obs_redact_prompts:
        stub_payload = redact_messages({"content": raw_content})
        attributes_content: Any = stub_payload["content"]
    else:
        attributes_content = raw_content
    telemetry.emit_log(
        "agentm.api.send_user_message",
        body={"content": attributes_content},
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.api.extension": self.extension,
            "agentm.api.content_chars": (
                content_chars if content_chars is not None else -1
            ),
        },
    )


def _diagnostic_to_otel(
    self: "DiagnosticEvent", telemetry: Any
) -> None:
    sev = {
        "info": SeverityNumber.INFO,
        "warning": SeverityNumber.WARN,
        "error": SeverityNumber.ERROR,
    }.get(self.level, SeverityNumber.INFO)
    telemetry.emit_log(
        "agentm.diagnostic",
        body={"message": self.message, "level": self.level},
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.diagnostic.level": self.level,
            "agentm.diagnostic.source": self.source,
        },
        severity=sev,
    )


def _extension_unload_to_otel(
    self: "ExtensionUnloadEvent", telemetry: Any
) -> None:
    telemetry.emit_log(
        "agentm.extension.unload",
        body={"trigger": self.trigger, "tier": self.tier},
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.extension.name": self.name,
            "agentm.extension.module_path": self.module_path,
            "agentm.extension.trigger": self.trigger,
            "agentm.extension.error": self.error or "",
        },
        severity=(SeverityNumber.ERROR if self.error else SeverityNumber.INFO),
    )


def _session_ready_to_otel(
    self: "SessionReadyEvent", telemetry: Any
) -> None:
    """Emit ``agentm.session.ready`` with the final tool / command / extension lists."""
    from agentm.core.lib import to_jsonable

    telemetry.emit_log(
        "agentm.session.ready",
        body={
            "session_id": self.session_id,
            "cwd": self.cwd,
            "tool_names": list(self.tool_names),
            "command_names": list(self.command_names),
            "extension_module_paths": list(self.extension_module_paths),
            "model": to_jsonable(self.model),
        },
        attributes={
            "agentm.session.id": self.session_id,
            "agentm.session.scenario": telemetry.obs_scenario,
            "agentm.session.tool_count": len(self.tool_names),
            "agentm.session.command_count": len(self.command_names),
            "agentm.session.extension_count": len(self.extension_module_paths),
        },
    )


def _session_shutdown_to_otel(
    self: "SessionShutdownEvent", telemetry: Any
) -> None:
    """Drain every still-open tracked span; emit ``agentm.session.end``."""
    del self
    end_ns = time.time_ns()
    telemetry.close_open_spans(status_description="session shutdown")
    duration_ns = (
        end_ns - telemetry.obs_session_start_ns
        if telemetry.obs_session_start_ns
        else 0
    )
    telemetry.emit_log(
        "agentm.session.end",
        body={
            "session_id": telemetry.session_id,
            "root_session_id": telemetry.obs_root_session_id,
            "parent_session_id": telemetry.obs_parent_session_id or None,
            "purpose": telemetry.obs_purpose,
            "scenario": telemetry.obs_scenario or None,
            "duration_ns": duration_ns,
        },
        attributes={
            "agentm.session.id": telemetry.session_id,
            "agentm.session.duration_ns": duration_ns,
        },
    )


# --- Registry population ---------------------------------------------------

register_otel(BeforeAgentStartEvent, _before_agent_start_to_otel)
register_otel(AgentEndEvent, _agent_end_to_otel)
register_otel(LlmRequestStartEvent, _llm_request_start_to_otel)
register_otel(LlmRequestEndEvent, _llm_request_end_to_otel)
register_otel(ToolCallEvent, _tool_call_to_otel)
register_otel(ToolResultEvent, _tool_result_to_otel)
register_otel(TurnStartEvent, _turn_start_to_otel)
register_otel(TurnEndEvent, _turn_end_to_otel)
register_otel(SessionHeaderEmittedEvent, _session_header_to_otel)
register_otel(MessageAppendedEvent, _message_appended_to_otel)
register_otel(ApiRegisterEvent, _api_register_to_otel)
register_otel(ApiSendUserMessageEvent, _api_send_user_message_to_otel)
register_otel(DiagnosticEvent, _diagnostic_to_otel)
register_otel(ExtensionUnloadEvent, _extension_unload_to_otel)
register_otel(SessionReadyEvent, _session_ready_to_otel)
register_otel(SessionShutdownEvent, _session_shutdown_to_otel)
