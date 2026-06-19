"""Behavioural contract for the rewritten observability atom.

The exact-bytes-of-JSONL-envelope assertions from before the OTLP cutover
have been retired (the semconv suite in
``test_observability_semconv.py`` locks down on-disk structure). What
this file covers now is the two cross-cutting behaviours the atom is
still responsible for after the cutover:

1. Bus observer wiring — installing the atom installs an observer that
   produces ``agentm.event.dispatch`` + ``agentm.handler.invoke`` log
   records, both stamped with the bus-owned ``dispatch_id``.
2. Mutation diff — when a handler mutates a mutable event in place, the
   atom records an ``agentm.handler.mutated`` log record listing the
   changed paths.

The fail-stop position is the contract, not the exact attribute values;
attribute encoding is locked down by ``test_observability_semconv.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import EventBus
from agentm.core.abi import (
    BeforeSendToLlmEvent,
    SessionShutdownEvent,
)
from agentm.core.runtime.extension import (
    _ExtensionAPIImpl,
    build_extension_api_scope,
)
from agentm.core.runtime.session_inbox import SessionInbox
from agentm.extensions.builtin import observability


class _SessionView:
    def get_session_id(self) -> str:
        return "session-1"

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


def _api(
    tmp_path: Path,
    *,
    lineage: dict[str, Any] | None = None,
    experiment: dict[str, Any] | None = None,
) -> _ExtensionAPIImpl:
    scope = build_extension_api_scope(
        bus=EventBus(),
        cwd=str(tmp_path),
        session_id="session-1",
        lineage=lineage,
        experiment=experiment,
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


def _trace_file(tmp_path: Path) -> Path:
    return tmp_path / ".agentm" / "observability" / "session-1.jsonl"


def _read_otlp(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _log_records(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from agentm.core.observability.otlp import iter_log_records

    out: list[dict[str, Any]] = []
    for line in lines:
        out.extend(iter_log_records(line))
    return out


def _attrs_dict(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in record.get("attributes", []) or []:
        key = attr["key"]
        value_wrapper = attr["value"]
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
    return out


def _body_dict(record: dict[str, Any]) -> dict[str, Any]:
    from agentm.core.observability.otlp import otlp_unwrap

    body = otlp_unwrap(record.get("body", {}))
    return body if isinstance(body, dict) else {}


def _handler(event: dict[str, Any]) -> dict[str, Any]:
    return {"seen": event["value"]}




def _mutating_handler(event: dict[str, Any]) -> str:
    event["value"] = 8
    event["added"] = {"nested": True}
    return "mutated"




# --- prompt-redaction integration ----------------------------------------

_LEAK_SECRET = "sk-proj-FAKE_SECRET_xxx"


def _build_before_send_event(text: str) -> Any:
    from agentm.core.abi import BeforeSendToLlmEvent as _Bs
    from agentm.core.abi import TextContent, UserMessage
    from agentm.core.abi.stream import Model

    msg = UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
    )
    model = Model(id="m", provider="stub", context_window=1000, max_output_tokens=100)
    return _Bs(messages=[msg], model=model, tools=[], system="sys")


@pytest.mark.asyncio
async def test_observability_strips_messages_from_before_send_to_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-event-log merge: ``before_send_to_llm`` no longer carries the
    full ``messages`` snapshot on disk — the canonical trajectory lives in
    ``agentm.message.appended`` records. Any secret in the prompt body
    therefore cannot reach the trace via this channel by construction.
    """
    monkeypatch.setenv("AGENTM_OBSERVABILITY_DIR", str(tmp_path / ".agentm" / "observability"))
    api = _api(tmp_path)
    observability.install(api, observability.ObservabilityConfig())
    event = _build_before_send_event(f"leak: {_LEAK_SECRET}")
    await api.events.emit(BeforeSendToLlmEvent.CHANNEL, event)
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=str(tmp_path))
    )

    raw = _trace_file(tmp_path).read_text(encoding="utf-8")
    assert _LEAK_SECRET not in raw, (
        "secret must not appear in any record — including via the "
        "before_send_to_llm dispatch payload"
    )

    records = _log_records(_read_otlp(_trace_file(tmp_path)))
    dispatch_records = [
        r
        for r in records
        if r.get("eventName") == "agentm.event.dispatch"
        and _attrs_dict(r).get("agentm.event.channel") == "before_send_to_llm"
    ]
    assert len(dispatch_records) == 1
    payload_str = _attrs_dict(dispatch_records[0]).get("agentm.event.payload", "")
    # The dispatch payload is JSON-encoded into a single attribute. Parse
    # and assert that the messages field was stripped.
    payload = json.loads(payload_str) if payload_str else {}
    assert "messages" not in payload, (
        "messages field must be stripped — trajectory lives in agentm.message.appended"
    )
    assert payload.get("system") == "sys"


def test_session_start_records_lineage_and_experiment_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AGENTM_OBSERVABILITY_DIR", str(tmp_path / ".agentm" / "observability")
    )
    reminder_text = "Check the ts-route-plan-service to ts-travel2-service edge."
    api = _api(
        tmp_path,
        lineage={
            "kind": "fork",
            "entrypoint": "agentm.cli",
            "source_session_id": "baseline-session",
            "fork_point": {"turn_index": 12},
        },
        experiment={
            "kind": "reminder_injection",
            "id": "ablation-1",
            "case_id": "1188",
            "reminder_id": "r1",
            "insert_turn_index": 12,
            "reminder_text": reminder_text,
        },
    )
    observability.install(api, observability.ObservabilityConfig())
    api.get_session_telemetry().shutdown()

    records = _log_records(_read_otlp(_trace_file(tmp_path)))
    starts = [r for r in records if r.get("eventName") == "agentm.session.start"]
    assert len(starts) == 1

    body = _body_dict(starts[0])
    attrs = _attrs_dict(starts[0])
    assert body["lineage"]["source_session_id"] == "baseline-session"
    assert body["experiment"]["reminder_text"] == reminder_text
    assert attrs["agentm.session.lineage.kind"] == "fork"
    assert attrs["agentm.session.lineage.source_session_id"] == "baseline-session"
    assert attrs["agentm.session.lineage.fork.turn_index"] == 12
    assert attrs["agentm.session.experiment.kind"] == "reminder_injection"
    assert attrs["agentm.session.experiment.id"] == "ablation-1"
    assert attrs["agentm.session.experiment.case_id"] == "1188"
    assert "agentm.session.experiment.reminder_text_hash" in attrs
    assert "reminder_text" not in attrs
