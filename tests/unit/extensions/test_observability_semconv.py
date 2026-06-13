"""Semconv contract for the rewritten observability atom.

These tests exist to lock the on-disk shape readers (session_manager,
catalog indexer, query_traces, llmharness CLI, rca graders) depend
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
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    EventBus,
    text_message,
)
from agentm.core.abi import (
    MessageAppendedEvent,
    SessionHeaderEmittedEvent,
    SessionShutdownEvent,
)
from agentm.core.runtime.extension import (
    _ExtensionAPIImpl,
    build_extension_api_scope,
)
from agentm.core.runtime.session_inbox import SessionInbox
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
        inbox=SessionInbox(),
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
    from agentm.core.observability.otlp import iter_spans

    out: list[dict[str, Any]] = []
    for line in lines:
        out.extend(iter_spans(line))
    return out


def _log_records_in(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from agentm.core.observability.otlp import iter_log_records

    out: list[dict[str, Any]] = []
    for line in lines:
        out.extend(iter_log_records(line))
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
async def test_writer_emits_session_header_and_message_appended_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SessionHeader + MessageAppended events become agentm.session.header /
    agentm.message.appended log records with the SessionEntry dict verbatim
    in the body. This is the contract SessionManager._load reads from.
    """
    monkeypatch.setenv("AGENTM_OBSERVABILITY_DIR", str(tmp_path / ".agentm" / "observability"))
    api = _build_api(tmp_path)
    observability.install(api, observability.ObservabilityConfig(include_handler_records=False))

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






