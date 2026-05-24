"""Semconv contract for the rewritten observability atom.

These tests exist to lock the on-disk shape readers (session_manager,
catalog indexer, tool_query_traces, llmharness CLI, rca graders) depend
on. They drive a real ``observability.install`` against a real
``ExtensionAPI`` + ``EventBus`` and emit a synthetic LLM turn + a
synthetic tool call, then parse the resulting OTLP/JSON ndjson file
and assert on the gen_ai semconv attributes and event-name vocabulary.

The fail-stop property: if any of these assertions break, every reader
downstream breaks. We never assert exact bytes — only structural
invariants — because OTLP encodes ids in base64 and timestamps as
strings of nanoseconds, neither of which is meaningful to test
verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agentm.core.abi import (
    AgentEndEvent,
    AssistantMessage,
    EventBus,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    ModelEndTurn,
    TextContent,
    ToolCallBlock,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
    text_message,
)
from agentm.core.abi.events import (
    BeforeAgentStartEvent,
    Event,
    MessageAppendedEvent,
    SessionHeaderEmittedEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.core.runtime.extension import (
    _ExtensionAPIImpl,
    build_extension_api_scope,
)
from agentm.core.runtime.session_manager import (
    _entry_to_record,
    _header_to_record,
)
from agentm.core.abi.session import (
    CURRENT_SESSION_VERSION,
    SessionHeader,
    message_entry,
)
from agentm.extensions.builtin import observability


class _SessionView:
    def get_messages(self) -> list[Any]:
        return []

    def get_branch(self) -> list[Any]:
        return []

    def get_leaf_id(self) -> str | None:
        return None

    def get_entry(self, entry_id: str) -> Any | None:
        del entry_id
        return None

    def get_loop_config(self) -> Any:
        return None

    def append_entry(self, type: str, payload: Any, parent_id: str | None = None) -> str:
        del type, payload, parent_id
        return "entry"


def _build_api(tmp_path: Path, *, session_id: str = "sess-semconv") -> _ExtensionAPIImpl:
    scope = build_extension_api_scope(
        bus=EventBus(),
        cwd=str(tmp_path),
        session_id=session_id,
        scenario="unit_semconv",
        session=_SessionView(),
        tools=[],
        commands={},
        providers={},
        renderers={},
        pending_user_messages=[],
        model_getter=lambda: None,
        provider_getter=lambda: None,
    )
    return _ExtensionAPIImpl(scope)


def _read_otlp_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _spans_in(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        for scope in line.get("scopeSpans", []) or []:
            out.extend(scope.get("spans", []) or [])
    return out


def _log_records_in(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        for scope in line.get("scopeLogs", []) or []:
            out.extend(scope.get("logRecords", []) or [])
    return out


def _span_attrs(span: dict[str, Any]) -> dict[str, Any]:
    """Flatten a span's OTLP attribute list into a plain dict.

    OTLP attribute values are tagged unions; we collapse to the most common
    representation (stringValue / intValue / boolValue) so tests can assert
    on plain Python types.
    """
    out: dict[str, Any] = {}
    for attr in span.get("attributes", []) or []:
        key = attr["key"]
        value_wrapper = attr["value"]
        for tag in ("stringValue", "boolValue"):
            if tag in value_wrapper:
                out[key] = value_wrapper[tag]
                break
        else:
            for tag in ("intValue", "doubleValue"):
                if tag in value_wrapper:
                    # OTLP encodes int as string-of-int in proto JSON.
                    raw = value_wrapper[tag]
                    try:
                        out[key] = int(raw) if tag == "intValue" else float(raw)
                    except (TypeError, ValueError):
                        out[key] = raw
                    break
    return out


def _log_record_attrs(record: dict[str, Any]) -> dict[str, Any]:
    return _span_attrs(record)  # same OTLP attribute encoding


def _log_record_body(record: dict[str, Any]) -> Any:
    body = record.get("body")
    if not isinstance(body, dict):
        return body
    if "stringValue" in body:
        return body["stringValue"]
    if "kvlistValue" in body:
        return _kvlist_to_dict(body["kvlistValue"].get("values", []))
    return body


def _kvlist_to_dict(values: list[dict[str, Any]]) -> Any:
    out: dict[str, Any] = {}
    for item in values:
        key = item.get("key")
        value_wrapper = item.get("value") or {}
        if "stringValue" in value_wrapper:
            out[key] = value_wrapper["stringValue"]
        elif "intValue" in value_wrapper:
            try:
                out[key] = int(value_wrapper["intValue"])
            except (TypeError, ValueError):
                out[key] = value_wrapper["intValue"]
        elif "boolValue" in value_wrapper:
            out[key] = value_wrapper["boolValue"]
        elif "doubleValue" in value_wrapper:
            try:
                out[key] = float(value_wrapper["doubleValue"])
            except (TypeError, ValueError):
                out[key] = value_wrapper["doubleValue"]
        elif "kvlistValue" in value_wrapper:
            out[key] = _kvlist_to_dict(value_wrapper["kvlistValue"].get("values", []))
        elif "arrayValue" in value_wrapper:
            arr = []
            for elem in value_wrapper["arrayValue"].get("values", []):
                arr.append(_unwrap_value(elem))
            out[key] = arr
        else:
            out[key] = value_wrapper
    return out


def _unwrap_value(value_wrapper: dict[str, Any]) -> Any:
    if "stringValue" in value_wrapper:
        return value_wrapper["stringValue"]
    if "intValue" in value_wrapper:
        try:
            return int(value_wrapper["intValue"])
        except (TypeError, ValueError):
            return value_wrapper["intValue"]
    if "boolValue" in value_wrapper:
        return value_wrapper["boolValue"]
    if "doubleValue" in value_wrapper:
        return float(value_wrapper["doubleValue"])
    if "kvlistValue" in value_wrapper:
        return _kvlist_to_dict(value_wrapper["kvlistValue"].get("values", []))
    if "arrayValue" in value_wrapper:
        return [_unwrap_value(v) for v in value_wrapper["arrayValue"].get("values", [])]
    return value_wrapper


def _log_event_name(record: dict[str, Any]) -> str:
    return str(record.get("eventName", ""))


def _trace_path(api: _ExtensionAPIImpl) -> Path:
    return Path(api.cwd) / ".agentm" / "observability" / f"{api.session_id}.jsonl"


# --------------------------------------------------------------------------
# The actual gate tests


@pytest.mark.asyncio
async def test_writer_emits_invoke_agent_chat_execute_tool_spans(
    tmp_path: Path,
) -> None:
    """Drive one synthetic turn: BeforeAgentStart -> LlmRequestStart/End ->
    ToolCall -> ToolResult -> TurnEnd -> AgentEnd. The trace must contain
    exactly one of each gen_ai span family with the right semconv attrs.
    """
    api = _build_api(tmp_path)
    observability.install(api, {"include_handler_records": False})

    # BeforeAgentStart -> open invoke_agent span
    await api.events.emit(
        BeforeAgentStartEvent.CHANNEL,
        BeforeAgentStartEvent(messages=[], system="sys"),
    )

    # LlmRequestStart -> open chat span
    await api.events.emit(
        LlmRequestStartEvent.CHANNEL,
        LlmRequestStartEvent(
            turn_index=0,
            message_count=1,
            tool_count=1,
            system_chars=3,
            model_id="m-stub",
            turn_id=42,
        ),
    )
    # LlmRequestEnd -> close chat span
    await api.events.emit(
        LlmRequestEndEvent.CHANNEL,
        LlmRequestEndEvent(
            turn_index=0,
            chunk_count=2,
            duration_ns=1_000_000,
            error=None,
            turn_id=42,
        ),
    )

    # ToolCall -> open execute_tool span; ToolResult -> close
    tool_call_id = "tc-1"
    await api.events.emit(
        ToolCallEvent.CHANNEL,
        ToolCallEvent(
            tool_call_id=tool_call_id,
            tool_name="read",
            args={"path": "/x"},
        ),
    )
    await api.events.emit(
        ToolResultEvent.CHANNEL,
        ToolResultEvent(
            tool_call_id=tool_call_id,
            tool_name="read",
            result=ToolResult(
                content=[TextContent(type="text", text="ok")], is_error=False
            ),
        ),
    )

    # AgentEnd -> close invoke_agent span
    msg = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        timestamp=0.0,
        stop_reason="end_turn",
    )
    await api.events.emit(
        AgentEndEvent.CHANNEL, AgentEndEvent(messages=[msg], cause=ModelEndTurn())
    )

    # Shutdown drains the SDK batch processor.
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=api.cwd)
    )

    lines = _read_otlp_lines(_trace_path(api))
    spans = _spans_in(lines)

    chat_spans = [s for s in spans if s.get("name", "").startswith("chat ")]
    invoke_spans = [s for s in spans if s.get("name", "").startswith("invoke_agent ")]
    tool_spans = [s for s in spans if s.get("name", "").startswith("execute_tool ")]

    assert len(chat_spans) == 1, [s["name"] for s in spans]
    assert len(invoke_spans) == 1
    assert len(tool_spans) == 1

    chat_attrs = _span_attrs(chat_spans[0])
    assert chat_attrs["gen_ai.operation.name"] == "chat"
    assert chat_attrs["gen_ai.request.model"] == "m-stub"
    assert chat_attrs["gen_ai.conversation.id"] == api.session_id

    invoke_attrs = _span_attrs(invoke_spans[0])
    assert invoke_attrs["gen_ai.operation.name"] == "invoke_agent"
    assert invoke_attrs["gen_ai.agent.name"] == "unit_semconv"
    assert invoke_attrs["gen_ai.conversation.id"] == api.session_id

    tool_attrs = _span_attrs(tool_spans[0])
    assert tool_attrs["gen_ai.operation.name"] == "execute_tool"
    assert tool_attrs["gen_ai.tool.name"] == "read"
    assert tool_attrs["gen_ai.tool.call.id"] == "tc-1"


@pytest.mark.asyncio
async def test_writer_emits_session_header_and_message_appended_logs(
    tmp_path: Path,
) -> None:
    """SessionHeader + MessageAppended events become agentm.session.header /
    agentm.message.appended log records with the SessionEntry dict verbatim
    in the body. This is the contract SessionManager._load reads from.
    """
    api = _build_api(tmp_path)
    observability.install(api, {"include_handler_records": False})

    header = SessionHeader(
        type="session",
        version=CURRENT_SESSION_VERSION,
        id=api.session_id,
        timestamp=1.0,
        cwd=api.cwd,
        parent_session=None,
    )
    await api.events.emit(
        SessionHeaderEmittedEvent.CHANNEL,
        SessionHeaderEmittedEvent(record=_header_to_record(header)),
    )

    entry = message_entry(text_message("hello"), parent_id=None)
    await api.events.emit(
        MessageAppendedEvent.CHANNEL,
        MessageAppendedEvent(record=_entry_to_record(entry)),
    )

    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=api.cwd)
    )

    lines = _read_otlp_lines(_trace_path(api))
    records = _log_records_in(lines)

    headers = [r for r in records if _log_event_name(r) == "agentm.session.header"]
    msgs = [r for r in records if _log_event_name(r) == "agentm.message.appended"]
    assert len(headers) == 1
    assert len(msgs) == 1

    header_body = _log_record_body(headers[0])
    assert isinstance(header_body, dict)
    assert header_body["id"] == api.session_id
    assert header_body["version"] == CURRENT_SESSION_VERSION

    msg_body = _log_record_body(msgs[0])
    assert isinstance(msg_body, dict)
    assert msg_body["id"] == entry.id
    assert msg_body["type"] == entry.type
    payload = msg_body["payload"]
    assert isinstance(payload, dict)
    assert payload["role"] == "user"


@pytest.mark.asyncio
async def test_writer_emits_turn_summary_with_usage_attrs(tmp_path: Path) -> None:
    """TurnEndEvent produces agentm.turn.summary with gen_ai.usage.* attrs
    and tool counts. tool_guard_watch + catalog/indexer depend on these.
    """
    api = _build_api(tmp_path)
    observability.install(api, {"include_handler_records": False})

    await api.events.emit(
        TurnStartEvent.CHANNEL, TurnStartEvent(turn_index=0, turn_id=1)
    )
    await api.events.emit(
        TurnEndEvent.CHANNEL,
        TurnEndEvent(
            turn_index=0,
            turn_id=1,
            message=AssistantMessage(
                role="assistant",
                content=[
                    TextContent(type="text", text="ok"),
                    ToolCallBlock(
                        type="tool_call", id="t-1", name="read", arguments={}
                    ),
                ],
                timestamp=0.0,
                stop_reason="end_turn",
                usage=Usage(
                    input_tokens=10,
                    output_tokens=20,
                    cache_read=0,
                    cache_write=0,
                ),
            ),
        ),
    )
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=api.cwd)
    )

    lines = _read_otlp_lines(_trace_path(api))
    records = _log_records_in(lines)
    summaries = [r for r in records if _log_event_name(r) == "agentm.turn.summary"]
    assert len(summaries) == 1
    attrs = _log_record_attrs(summaries[0])
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 20
    assert attrs["agentm.turn.tool_call_count"] == 1
    assert attrs["agentm.turn.stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_writer_emits_fingerprint_log_record(tmp_path: Path) -> None:
    """SessionReadyEvent produces agentm.session.fingerprint with task_meta
    and atom hashes. tool_query_traces and the rca grader depend on this.
    """
    api = _build_api(tmp_path)
    observability.install(api, {"include_handler_records": False})

    await api.events.emit(
        SessionReadyEvent.CHANNEL,
        SessionReadyEvent(
            cwd=api.cwd,
            session_id=api.session_id,
            tool_names=(),
            command_names=(),
            extension_module_paths=(),
            model=None,
            root_session_id=api.root_session_id,
            task_id=None,
            persona=None,
            task_class="rca_baseline",
            eval_run_id="er_test",
            eval_task_id="01_mysql",
        ),
    )
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=api.cwd)
    )

    lines = _read_otlp_lines(_trace_path(api))
    records = _log_records_in(lines)
    fps = [r for r in records if _log_event_name(r) == "agentm.session.fingerprint"]
    assert len(fps) == 1
    attrs = _log_record_attrs(fps[0])
    assert attrs["agentm.task.class"] == "rca_baseline"
    assert attrs["agentm.task.eval_run_id"] == "er_test"
    assert attrs["agentm.task.eval_task_id"] == "01_mysql"


@pytest.mark.asyncio
async def test_dispatch_id_links_dispatch_and_handler_records(tmp_path: Path) -> None:
    """A single emit produces one agentm.event.dispatch record and one
    agentm.handler.invoke per handler, all sharing the same dispatch_id.
    This is the join key consumers need to reconstruct the fanout.
    """
    api = _build_api(tmp_path)
    observability.install(
        api, {"include_handler_records": True, "include_mutation_diff": False}
    )

    # Use a real Event subclass on a bespoke channel so the bus-assigned
    # dispatch_id lands on the field; observability reads it off the event
    # in scope and stamps it onto both records.
    @dataclass(slots=True)
    class _ProbeEvent(Event):
        CHANNEL: ClassVar[str] = "custom"
        value: int = 0

    seen: list[str] = []

    def handler(event: Any) -> None:
        seen.append(event.dispatch_id)

    api.events.on("custom", handler)
    await api.events.emit("custom", _ProbeEvent(value=1))
    # Drain telemetry by emitting the session-shutdown event; the
    # substrate-installed POST handler flushes the OTel pipeline.
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=api.cwd)
    )

    assert len(seen) == 1 and seen[0]
    expected_id = seen[0]

    lines = _read_otlp_lines(_trace_path(api))
    records = _log_records_in(lines)
    dispatches = [
        r
        for r in records
        if _log_event_name(r) == "agentm.event.dispatch"
        and _log_record_attrs(r).get("agentm.event.channel") == "custom"
    ]
    invokes = [
        r
        for r in records
        if _log_event_name(r) == "agentm.handler.invoke"
        and _log_record_attrs(r).get("agentm.event.channel") == "custom"
    ]
    assert len(dispatches) == 1
    assert len(invokes) == 1
    assert _log_record_attrs(dispatches[0])["agentm.event.dispatch_id"] == expected_id
    assert _log_record_attrs(invokes[0])["agentm.event.dispatch_id"] == expected_id
