"""OTel translators for v2 session events."""

# code-health: ignore-file[AM022] -- adapts dynamic OpenTelemetry SDK objects

from __future__ import annotations

import time
from typing import Any

from opentelemetry.trace import SpanKind, Status, StatusCode, set_span_in_context

from agentm.core.abi.events import (
    ApiRegisterEvent,
    DiagnosticEvent,
    ExtensionInstallEvent,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    RunEndEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnBeginEvent,
    TurnCommittedEvent,
)
from agentm.core.abi.messages import TextContent, ToolCallBlock, ToolResultBlock
from agentm.core.lib.serialization import to_jsonable
from agentm.extensions.observability.otel_dispatch import register_otel

_SPAN_RUN = "run"
_SPAN_TURN = "turn"
_SPAN_CHAT = "chat"
_SPAN_TOOL = "tool"


def _span_context(span: Any) -> Any:
    return set_span_in_context(span) if span is not None else None


def _attrs(telemetry: Any, **extra: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "agentm.session.id": telemetry.session_id,
        "agentm.session.root_id": telemetry.obs_root_session_id,
        "agentm.session.parent_id": telemetry.obs_parent_session_id,
        "agentm.session.purpose": telemetry.obs_purpose,
        "agentm.session.scenario": telemetry.obs_scenario,
    }
    attrs.update(extra)
    return attrs


def _tool_text_chars(block: ToolResultBlock) -> int:
    total = 0
    for item in block.content:
        if isinstance(item, TextContent):
            total += len(item.text)
    return total


def _open_run_span(event: SessionReadyEvent, telemetry: Any) -> None:
    span = telemetry.tracer.start_span(
        f"agentm.session {telemetry.obs_scenario or telemetry.obs_purpose or event.session_id}",
        kind=SpanKind.INTERNAL,
        attributes=_attrs(
            telemetry,
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": telemetry.obs_scenario
                or telemetry.obs_purpose
                or "agentm",
                "gen_ai.conversation.id": event.session_id,
                "agentm.session.cwd": event.cwd,
                "agentm.session.tool_count": len(event.tool_names),
                "agentm.session.extension_count": len(event.extension_module_paths),
            },
        ),
    )
    telemetry.open_span(_SPAN_RUN, telemetry.session_id, span)


def _session_ready_to_otel(event: SessionReadyEvent, telemetry: Any) -> None:
    _open_run_span(event, telemetry)
    telemetry.emit_log(
        "agentm.session.ready",
        body={
            "session_id": event.session_id,
            "root_session_id": event.root_session_id,
            "parent_session_id": event.parent_session_id,
            "cwd": event.cwd,
            "tool_names": list(event.tool_names),
            "extension_module_paths": list(event.extension_module_paths),
            "model": to_jsonable(event.model),
        },
        attributes=_attrs(
            telemetry,
            **{
                "agentm.session.tool_count": len(event.tool_names),
                "agentm.session.extension_count": len(event.extension_module_paths),
            },
        ),
    )


def _turn_begin_to_otel(event: TurnBeginEvent, telemetry: Any) -> None:
    telemetry.turn_state["turn_start_ns"] = time.time_ns()
    telemetry.turn_state["current_tool_errors"] = 0
    run_span = telemetry.span_tracker.get((_SPAN_RUN, telemetry.session_id))
    span = telemetry.tracer.start_span(
        "agentm.turn",
        kind=SpanKind.INTERNAL,
        context=_span_context(run_span),
        attributes=_attrs(
            telemetry,
            **{
                "agentm.turn.index": event.turn_index,
                "agentm.turn.id": event.turn_id,
                "agentm.trigger": telemetry.to_otel_attr(to_jsonable(event.trigger)),
            },
        ),
    )
    telemetry.open_span(_SPAN_TURN, event.turn_id, span)
    telemetry.current_turn_span = span


def _llm_start_to_otel(event: LlmRequestStartEvent, telemetry: Any) -> None:
    span = telemetry.tracer.start_span(
        f"chat {event.model_id or 'unknown'}",
        kind=SpanKind.CLIENT,
        context=_span_context(telemetry.current_turn_span),
        attributes=_attrs(
            telemetry,
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": event.model_id,
                "gen_ai.conversation.id": telemetry.session_id,
                "agentm.turn.index": event.turn_index,
                "agentm.turn.id": event.turn_id,
                "agentm.llm.message_count": event.message_count,
                "agentm.llm.tool_count": event.tool_count,
                "agentm.llm.system_chars": event.system_chars,
            },
        ),
    )
    if telemetry.obs_provider_name:
        span.set_attribute("gen_ai.provider.name", telemetry.obs_provider_name)
    telemetry.open_span(_SPAN_CHAT, event.turn_id, span)
    if event.system_text is not None and not telemetry.obs_redact_prompts:
        telemetry.emit_log(
            "agentm.llm.system_prompt",
            body={"turn_id": event.turn_id, "text": event.system_text},
            attributes=_attrs(
                telemetry,
                **{
                    "agentm.turn.index": event.turn_index,
                    "agentm.turn.id": event.turn_id,
                    "agentm.llm.system_chars": event.system_chars,
                },
            ),
        )


def _llm_end_to_otel(event: LlmRequestEndEvent, telemetry: Any) -> None:
    span = telemetry.pop_span(_SPAN_CHAT, event.turn_id)
    if span is None:
        return
    span.set_attribute("agentm.llm.chunk_count", event.chunk_count)
    span.set_attribute("agentm.llm.duration_ns", event.duration_ns)
    if event.error:
        span.set_attribute("agentm.llm.error", event.error)
        span.set_status(Status(StatusCode.ERROR, event.error))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def _tool_call_to_otel(event: ToolCallEvent, telemetry: Any) -> None:
    span = telemetry.tracer.start_span(
        f"execute_tool {event.tool_name}",
        kind=SpanKind.INTERNAL,
        context=_span_context(telemetry.current_turn_span),
        attributes=_attrs(
            telemetry,
            **{
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": event.tool_name,
                "gen_ai.tool.call.id": event.tool_call_id,
                "gen_ai.tool.call.arguments": telemetry.to_otel_attr(event.args),
            },
        ),
    )
    telemetry.open_span(_SPAN_TOOL, event.tool_call_id, span)


def _tool_result_to_otel(event: ToolResultEvent, telemetry: Any) -> None:
    is_error = event.result.is_error if event.result is not None else False
    if is_error:
        telemetry.turn_state["current_tool_errors"] += 1
    span = telemetry.pop_span(_SPAN_TOOL, event.tool_call_id)
    if span is None:
        return
    span.set_attribute("agentm.tool.is_error", is_error)
    span.set_attribute("gen_ai.tool.call.result", telemetry.to_otel_attr(event.result))
    if is_error:
        span.set_status(Status(StatusCode.ERROR, "tool result is_error=True"))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def _turn_committed_to_otel(event: TurnCommittedEvent, telemetry: Any) -> None:
    if event.turn is None:
        return
    turn = event.turn
    end_ns = time.time_ns()
    start_ns = telemetry.turn_state.get("turn_start_ns", 0)
    observed_duration_ns = end_ns - start_ns if start_ns else 0
    duration_ns = turn.meta.duration_ns or observed_duration_ns
    tool_calls: list[str] = []
    tool_error_count = 0
    tool_result_chars = 0
    for round_ in turn.rounds:
        for block in round_.response.content:
            if isinstance(block, ToolCallBlock):
                tool_calls.append(block.name)
        for record in round_.tool_results:
            tool_error_count += 1 if record.result.is_error else 0
            tool_result_chars += _tool_text_chars(record.result)

    span = telemetry.pop_span(_SPAN_TURN, turn.id)
    if span is not None:
        span.set_attribute("agentm.turn.duration_ns", duration_ns)
        span.set_attribute("agentm.turn.tool_call_count", len(tool_calls))
        span.set_attribute("agentm.turn.tool_error_count", tool_error_count)
        span.set_attribute("gen_ai.usage.input_tokens", turn.meta.total_input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", turn.meta.total_output_tokens)
        span.set_status(Status(StatusCode.OK))
        span.end()
    if telemetry.current_turn_span is span:
        telemetry.current_turn_span = None

    telemetry.emit_log(
        "agentm.turn.committed",
        body={
            "turn_index": turn.index,
            "turn_id": turn.id,
            "duration_ns": duration_ns,
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "tool_error_count": tool_error_count,
            "tool_result_chars": tool_result_chars,
            "outcome": to_jsonable(turn.outcome),
            "meta": to_jsonable(turn.meta),
        },
        attributes=_attrs(
            telemetry,
            **{
                "agentm.turn.index": turn.index,
                "agentm.turn.id": turn.id,
                "agentm.turn.duration_ns": duration_ns,
                "agentm.turn.tool_call_count": len(tool_calls),
                "agentm.turn.tool_error_count": tool_error_count,
                "gen_ai.usage.input_tokens": turn.meta.total_input_tokens,
                "gen_ai.usage.output_tokens": turn.meta.total_output_tokens,
            },
        ),
    )
    telemetry.turn_state["current_tool_errors"] = 0


def _run_end_to_otel(event: RunEndEvent, telemetry: Any) -> None:
    body = {
        "outcome": to_jsonable(event.outcome),
        "meta": to_jsonable(event.meta),
    }
    attrs = _attrs(telemetry)
    run_span = telemetry.span_tracker.get((_SPAN_RUN, telemetry.session_id))
    if run_span is not None:
        run_span.set_attribute("agentm.run.outcome", telemetry.to_otel_attr(body))
    telemetry.emit_log("agentm.run.end", body=body, attributes=attrs)


def _api_register_to_otel(event: ApiRegisterEvent, telemetry: Any) -> None:
    telemetry.emit_log(
        "agentm.api.register",
        body={"payload": to_jsonable(event.payload)},
        attributes=_attrs(
            telemetry,
            **{
                "agentm.api.kind": event.kind,
                "agentm.api.name": event.name,
                "agentm.api.extension": event.extension,
            },
        ),
    )


def _diagnostic_to_otel(event: DiagnosticEvent, telemetry: Any) -> None:
    severity = {
        "info": "info",
        "warning": "warning",
        "error": "error",
    }.get(event.level, "info")
    telemetry.emit_log(
        "agentm.diagnostic",
        body={"level": event.level, "message": event.message},
        attributes=_attrs(
            telemetry,
            **{
                "agentm.diagnostic.level": event.level,
                "agentm.diagnostic.source": event.source,
            },
        ),
        severity=severity,
    )


def _extension_install_to_otel(event: ExtensionInstallEvent, telemetry: Any) -> None:
    telemetry.emit_log(
        "agentm.extension.install",
        attributes=_attrs(
            telemetry,
            **{
                "agentm.extension.name": event.name,
                "agentm.extension.module_path": event.module_path,
                "agentm.extension.phase": event.phase,
                "agentm.extension.duration_ns": event.duration_ns,
                "agentm.extension.trigger": event.trigger,
                "agentm.extension.error": event.error or "",
            },
        ),
        severity="error" if event.error else "info",
    )


def _session_shutdown_to_otel(event: SessionShutdownEvent, telemetry: Any) -> None:
    del event
    end_ns = time.time_ns()
    run_span = telemetry.pop_span(_SPAN_RUN, telemetry.session_id)
    if run_span is not None:
        run_span.set_status(Status(StatusCode.OK))
        run_span.end()
    telemetry.close_open_spans(status_description="session shutdown")
    duration_ns = (
        end_ns - telemetry.obs_session_start_ns if telemetry.obs_session_start_ns else 0
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
        attributes=_attrs(
            telemetry,
            **{"agentm.session.duration_ns": duration_ns},
        ),
    )


register_otel(SessionReadyEvent, _session_ready_to_otel)
register_otel(TurnBeginEvent, _turn_begin_to_otel)
register_otel(LlmRequestStartEvent, _llm_start_to_otel)
register_otel(LlmRequestEndEvent, _llm_end_to_otel)
register_otel(ToolCallEvent, _tool_call_to_otel)
register_otel(ToolResultEvent, _tool_result_to_otel)
register_otel(TurnCommittedEvent, _turn_committed_to_otel)
register_otel(RunEndEvent, _run_end_to_otel)
register_otel(ApiRegisterEvent, _api_register_to_otel)
register_otel(DiagnosticEvent, _diagnostic_to_otel)
register_otel(ExtensionInstallEvent, _extension_install_to_otel)
register_otel(SessionShutdownEvent, _session_shutdown_to_otel)
