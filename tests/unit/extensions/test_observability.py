from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import EventBus
from agentm.extensions.builtin import observability
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.runtime.extension import _ExtensionAPIImpl


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


def _api(tmp_path: Path) -> _ExtensionAPIImpl:
    from agentm.core.runtime.extension import build_extension_api_scope

    scope = build_extension_api_scope(
        bus=EventBus(),
        cwd=str(tmp_path),
        session_id="session-1",
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


def _handler(event: dict[str, Any]) -> dict[str, Any]:
    return {"seen": event["value"]}


@pytest.mark.asyncio
async def test_observability_trace_snapshot_uses_add_observer(
    tmp_path: Path,
) -> None:
    """Structural smoke: observability subscribes via ``add_observer`` (no
    EventBus monkeypatch) and writes a session-start / event.dispatch /
    handler.invoke / session-end quartet for a single ``emit`` round-trip.

    Previously asserted on exact JSONL bytes; that shape was fragile to any
    extra allocation on the emit path (e.g. the bus-owned ``dispatch_id``
    landing in Commit 1 of the single-event-log cutover). The behavioural
    contract — observer is wired, dispatches produce dispatch + invoke
    records — is what we actually want to lock down; the exact integer
    timestamps were never the point. The whole trace shape is rewritten in
    Commit 2 of the cutover anyway, at which point this test is replaced
    by the OTLP-semconv suite.
    """

    api = _api(tmp_path)
    observability.install(api, {"path": "trace.jsonl", "include_handler_records": True})
    api.on("alpha", _handler)

    await api.events.emit("alpha", {"value": 7})
    await api.events.emit(
        SessionShutdownEvent.CHANNEL,
        SessionShutdownEvent(cwd=str(tmp_path)),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = [r["kind"] for r in records]
    assert "session.start" in kinds
    assert "event.dispatch" in kinds
    assert "handler.invoke" in kinds
    assert "session.end" in kinds

    dispatch = next(r for r in records if r["kind"] == "event.dispatch")
    assert dispatch["attributes"]["channel"] == "alpha"
    assert dispatch["attributes"]["event"] == {"value": 7}
    assert dispatch["attributes"]["handler_count"] == 1

    invoke = next(r for r in records if r["kind"] == "handler.invoke")
    assert invoke["attributes"]["channel"] == "alpha"
    assert invoke["attributes"]["result"] == {"seen": 7}


def _mutating_handler(event: dict[str, Any]) -> str:
    event["value"] = 8
    event["added"] = {"nested": True}
    return "mutated"


@pytest.mark.asyncio
async def test_observability_records_mutation_diff_without_api_on_patch(
    tmp_path: Path,
) -> None:
    api = _api(tmp_path)
    observability.install(
        api,
        {
            "path": "trace.jsonl",
            "include_handler_records": True,
            "include_mutation_diff": True,
        },
    )
    api.on("context", _mutating_handler)

    await api.events.emit("context", {"value": 7})
    await api.events.emit(
        SessionShutdownEvent.CHANNEL,
        SessionShutdownEvent(cwd=str(tmp_path)),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    mutated = [record for record in records if record["kind"] == "handler.mutated"]
    assert len(mutated) == 1
    assert mutated[0]["attributes"]["handler"] == (
        "tests.unit.extensions.test_observability._mutating_handler"
    )
    assert {item["path"] for item in mutated[0]["attributes"]["mutations"]} == {
        "value",
        "added",
    }


# --- prompt-redaction integration ----------------------------------------

_LEAK_SECRET = "sk-proj-FAKE_SECRET_xxx"


def _trace_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _build_before_send_event(text: str) -> Any:
    """Construct a real ``BeforeSendToLlmEvent`` so the test exercises the
    same serialization path the production loop uses."""

    from agentm.core.abi.events import BeforeSendToLlmEvent
    from agentm.core.abi.messages import TextContent, UserMessage
    from agentm.core.abi.stream import Model

    msg = UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
    )
    model = Model(
        id="m",
        provider="stub",
        context_window=1000,
        max_output_tokens=100,
    )
    return BeforeSendToLlmEvent(
        messages=[msg],
        model=model,
        tools=[],
        system="sys",
    )


@pytest.mark.asyncio
async def test_observability_strips_messages_from_before_send_to_llm(
    tmp_path: Path,
) -> None:
    """Single-event-log merge (.claude/designs/single-event-log.md): the
    ``before_send_to_llm`` event no longer carries a ``messages`` snapshot
    on disk — the full trajectory lives in ``message.appended`` rows of
    the same merged log. Any leaked secret in the prompt body therefore
    cannot reach the trace via this channel by construction (the field is
    absent), making the old redaction layer redundant."""

    from agentm.core.abi.events import (
        BeforeSendToLlmEvent,
        SessionShutdownEvent as _Shut,
    )

    api = _api(tmp_path)
    observability.install(api, {"path": "trace.jsonl"})
    event = _build_before_send_event(f"leak: {_LEAK_SECRET}")
    await api.events.emit(BeforeSendToLlmEvent.CHANNEL, event)
    await api.events.emit(_Shut.CHANNEL, _Shut(cwd=str(tmp_path)))

    raw = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    assert _LEAK_SECRET not in raw, "secret must not appear via before_send_to_llm"

    dispatch_records = [
        r
        for r in _trace_records(tmp_path / "trace.jsonl")
        if r["kind"] == "event.dispatch" and r["attributes"]["channel"] == "before_send_to_llm"
    ]
    assert len(dispatch_records) == 1
    event_payload = dispatch_records[0]["attributes"]["event"]
    assert "messages" not in event_payload, (
        "messages field must be stripped — trajectory lives in message.appended"
    )
    # System prompt remains untouched on this channel (it is not a
    # messages-array snapshot; the spec only drops the duplicated
    # trajectory field).
    assert event_payload["system"] == "sys"
