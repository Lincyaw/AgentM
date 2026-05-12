from __future__ import annotations

import itertools
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

import agentm.core.abi.events as events_mod
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = itertools.count(1)
    monkeypatch.setattr(observability, "_now_ns", lambda: next(counter) * 1000)
    monkeypatch.setattr(observability.uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))
    monkeypatch.setattr(events_mod.time, "perf_counter_ns", lambda: next(counter) * 100)

    api = _api(tmp_path)
    observability.install(api, {"path": "trace.jsonl", "include_handler_records": True})
    api.on("alpha", _handler)

    await api.events.emit("alpha", {"value": 7})
    await api.events.emit(
        SessionShutdownEvent.CHANNEL,
        SessionShutdownEvent(cwd=str(tmp_path)),
    )

    expected = "\n".join(
        [
            '{"schema": "otel/span/v0", "kind": "session.start", "trace_id": "session-1", "span_id": "session-1", "parent_span_id": null, "name": "session.start", "start_time_unix_nano": 1000, "attributes": {"session_id": "session-1", "root_session_id": "session-1", "parent_session_id": null, "purpose": "root", "scenario": null, "cwd": "'
            + str(tmp_path)
            + '", "log_path": "'
            + str(tmp_path / "trace.jsonl")
            + '"}, "status": {"code": "OK"}}',
            '{"schema": "otel/span/v0", "kind": "handler.invoke", "trace_id": "session-1", "span_id": "0000000000000000", "parent_span_id": "0000000000000000", "name": "handler:alpha", "start_time_unix_nano": 5900, "end_time_unix_nano": 6000, "attributes": {"channel": "alpha", "handler": "tests.unit.extensions.test_observability._handler", "extension": null, "result": {"seen": 7}, "duration_ns": 100}, "status": {"code": "OK", "message": null, "traceback": null}}',
            '{"schema": "otel/span/v0", "kind": "event.dispatch", "trace_id": "session-1", "span_id": "0000000000000000", "parent_span_id": "session-1", "name": "emit:alpha", "start_time_unix_nano": 3000, "end_time_unix_nano": 8000, "attributes": {"channel": "alpha", "event": {"value": 7}, "handler_count": 1}, "status": {"code": "OK"}}',
            '{"schema": "otel/span/v0", "kind": "session.end", "trace_id": "session-1", "span_id": "session-1", "parent_span_id": null, "name": "session.end", "start_time_unix_nano": 1000, "end_time_unix_nano": 12000, "attributes": {"session_id": "session-1", "root_session_id": "session-1", "parent_session_id": null, "purpose": "root", "scenario": null}, "status": {"code": "OK"}}',
        ]
    ) + "\n"
    assert (tmp_path / "trace.jsonl").read_text(encoding="utf-8") == expected


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
