"""Fail-stop: wire_driver atom event -> outbound translation (§4).

wire_driver is the only new code in channels v2; it is the single bridge
from a session's events to the chat surface. If a turn_end produces no
assistant-text outbound, or a write-class tool_call is not gated through
the approval manager, the gateway is mute or unsafe.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi.events import (
    DiagnosticEvent,
    ExtensionReloadEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from agentm.core.abi.messages import AssistantMessage, TextContent, Usage
from agentm.core.abi.stream import TextDelta, ThinkingDelta
from agentm.core.abi.tool import ToolResult

_WIRE_DRIVER = (
    Path(__file__).resolve().parents[3]
    / "src/agentm/extensions/builtin/wire_driver.py"
)


def _load_wire_driver() -> Any:
    spec = importlib.util.spec_from_file_location("_wire_driver_test", _WIRE_DRIVER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeAPI:
    """Records ``@api.on`` handlers and serves ``get_service`` lookups.

    Keeps EVERY handler per channel (a channel may carry more than one — e.g.
    ``tool_call`` has the projector then the approval gate), so a test can pick
    the handler it means instead of depending on registration order silently
    overwriting one. Convention: ``handlers[ch][0]`` is the projector;
    ``handlers[ch][-1]`` is the approval gate on ``tool_call``.
    """

    def __init__(self, services: dict[str, Any]) -> None:
        self._services = services
        self.handlers: dict[str, list[Any]] = {}

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    def on(self, channel: str, handler: Any, **_kw: Any) -> None:
        self.handlers.setdefault(channel, []).append(handler)


class _RecordingApproval:
    def __init__(self, *, requires: set[str], decision: bool) -> None:
        self._requires = requires
        self._decision = decision
        self.requested: list[str] = []

    def requires(self, name: str) -> bool:
        return name in self._requires

    async def request(self, *, tool_name: str, **_kw: Any) -> bool:
        self.requested.append(tool_name)
        return self._decision


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
    )


def _install(services: dict[str, Any]) -> _FakeAPI:
    mod = _load_wire_driver()
    api = _FakeAPI(services)
    mod.install(api, {})
    return api


@pytest.mark.asyncio
async def test_turn_end_emits_assistant_text_outbound() -> None:
    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install(
        {
            "wire_outbound": sink,
            "session_key": "terminal:t1",
            "turn_context": {"channel": "terminal", "chat_id": "t1", "thread_id": None},
        }
    )
    await api.handlers[TurnEndEvent.CHANNEL][0](
        TurnEndEvent(turn_index=0, message=_assistant("the answer is 42"))
    )
    assert len(out) == 1
    body = out[0]
    assert body["content"] == "the answer is 42"
    assert body["metadata"]["kind"] == "assistant_text"
    assert body["channel"] == "terminal"
    assert body["chat_id"] == "t1"
    # Outbound must carry _session_key so the gateway sink stamps it onto
    # the envelope (§2.5/§3.3) — without it a multi-surface client cannot
    # attribute the message to its conversation.
    assert body["_session_key"] == "terminal:t1"


@pytest.mark.asyncio
async def test_empty_turn_end_emits_nothing() -> None:
    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install(
        {
            "wire_outbound": sink,
            "session_key": "terminal:t1",
            "turn_context": {"channel": "terminal", "chat_id": "t1", "thread_id": None},
        }
    )
    await api.handlers[TurnEndEvent.CHANNEL][0](
        TurnEndEvent(turn_index=0, message=_assistant("   "))
    )
    assert out == []


@pytest.mark.asyncio
async def test_tool_call_gated_through_approval_when_required() -> None:
    approval = _RecordingApproval(requires={"bash"}, decision=False)
    api = _install(
        {
            "wire_outbound": lambda *_a, **_k: None,
            "session_key": "terminal:t1",
            "turn_context": {"channel": "terminal", "chat_id": "t1", "thread_id": None, "sender_id": "u1"},
            "approval_manager": approval,
        }
    )
    # The approval gate is the LAST handler registered on tool_call (the
    # projector is [0]); pick it explicitly rather than by overwrite luck.
    result = await api.handlers[ToolCallEvent.CHANNEL][-1](
        ToolCallEvent(tool_call_id="tc1", tool_name="bash", args={"cmd": "ls"})
    )
    assert approval.requested == ["bash"]
    # Denied -> the handler blocks the tool.
    assert result == {
        "block": True,
        "reason": "tool 'bash' was denied via the chat approval gate",
    }


@pytest.mark.asyncio
async def test_tool_call_not_gated_when_policy_allows() -> None:
    approval = _RecordingApproval(requires=set(), decision=True)
    api = _install(
        {
            "wire_outbound": lambda *_a, **_k: None,
            "session_key": "terminal:t1",
            "turn_context": {"channel": "terminal", "chat_id": "t1", "thread_id": None, "sender_id": "u1"},
            "approval_manager": approval,
        }
    )
    result = await api.handlers[ToolCallEvent.CHANNEL][-1](
        ToolCallEvent(tool_call_id="tc1", tool_name="read", args={})
    )
    assert approval.requested == []
    assert result is None


@pytest.mark.asyncio
async def test_diagnostic_error_emits_outbound() -> None:
    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install(
        {
            "wire_outbound": sink,
            "session_key": "terminal:t1",
            "turn_context": {"channel": "terminal", "chat_id": "t1", "thread_id": None},
        }
    )
    # diagnostic is a SYNC handler (it can be emit_sync-dispatched, where an
    # async handler is silently skipped); it schedules the sink on the loop.
    handler = api.handlers[DiagnosticEvent.CHANNEL][0]
    assert not asyncio.iscoroutinefunction(handler)
    handler(DiagnosticEvent(level="error", source="loop", message="boom"))
    await asyncio.sleep(0)
    assert out[0]["metadata"]["kind"] == "diagnostic_error"
    assert out[0]["content"] == "boom"


def test_install_without_services_fails_fast() -> None:
    mod = _load_wire_driver()
    api = _FakeAPI({})  # no wire_outbound / session_key
    with pytest.raises(RuntimeError):
        mod.install(api, {})


# --- projector-table forwarding (the live event surface) -------------------


def _install_multi(sink: Any) -> _FakeAPI:
    return _install(
        {
            "wire_outbound": sink,
            "session_key": "terminal:t1",
            "turn_context": {
                "channel": "terminal",
                "chat_id": "t1",
                "thread_id": None,
            },
        }
    )


@pytest.mark.asyncio
async def test_stream_text_delta_projected() -> None:
    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install_multi(sink)
    await api.handlers[StreamDeltaEvent.CHANNEL][0](
        StreamDeltaEvent(turn_index=0, delta=TextDelta(text="hel"), turn_id=7)
    )
    await api.handlers[StreamDeltaEvent.CHANNEL][0](
        StreamDeltaEvent(turn_index=0, delta=ThinkingDelta(text="hmm"), turn_id=7)
    )
    # Empty deltas produce no frame.
    await api.handlers[StreamDeltaEvent.CHANNEL][0](
        StreamDeltaEvent(turn_index=0, delta=TextDelta(text=""), turn_id=7)
    )
    kinds = [(b["metadata"]["kind"], b["content"], b["metadata"]["turn_id"]) for b in out]
    assert kinds == [("stream_text", "hel", 7), ("stream_thinking", "hmm", 7)]


@pytest.mark.asyncio
async def test_turn_end_also_emits_usage() -> None:
    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install_multi(sink)
    msg = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        timestamp=0.0,
        usage=Usage(input_tokens=12, output_tokens=3, cache_read=4),
    )
    await api.handlers[TurnEndEvent.CHANNEL][0](
        TurnEndEvent(turn_index=0, message=msg, turn_id=5)
    )
    kinds = {b["metadata"]["kind"]: b for b in out}
    assert kinds["assistant_text"]["content"] == "done"
    usage = kinds["usage"]["metadata"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["cache_read"]) == (
        12,
        3,
        4,
    )


@pytest.mark.asyncio
async def test_tool_call_and_result_projected() -> None:
    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install_multi(sink)
    # handlers[tool_call] = [projector, approval_gate]; the projector is [0].
    await api.handlers[ToolCallEvent.CHANNEL][0](
        ToolCallEvent(tool_call_id="tc1", tool_name="bash", args={"cmd": "ls"})
    )
    await api.handlers[ToolResultEvent.CHANNEL][0](
        ToolResultEvent(
            tool_call_id="tc1",
            tool_name="bash",
            result=ToolResult(
                content=[TextContent(type="text", text="file.txt")], is_error=False
            ),
        )
    )
    await api.handlers[ToolResultEvent.CHANNEL][0](
        ToolResultEvent(
            tool_call_id="tc2",
            tool_name="bash",
            result=ToolResult(
                content=[TextContent(type="text", text="boom")], is_error=True
            ),
        )
    )
    call, ok_res, err_res = out
    assert call["metadata"]["kind"] == "tool_call"
    assert call["metadata"]["name"] == "bash"
    assert call["metadata"]["args"] == {"cmd": "ls"}
    assert ok_res["metadata"]["kind"] == "tool_result"
    assert ok_res["metadata"]["ok"] is True
    assert ok_res["content"] == "file.txt"
    assert err_res["metadata"]["ok"] is False


@pytest.mark.asyncio
async def test_sync_emitted_control_event_is_forwarded() -> None:
    """extension_reload is dispatched via bus.emit_sync, where an async
    handler would be silently skipped. wire_driver registers a SYNC handler
    that schedules the async sink on the loop — without it, self-modification
    (the headline behavior the TUI exists to surface) would be invisible.
    """
    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install_multi(sink)
    handler = api.handlers[ExtensionReloadEvent.CHANNEL][0]
    # The handler MUST be a plain (sync) function so emit_sync runs it.
    assert not asyncio.iscoroutinefunction(handler)
    handler(
        ExtensionReloadEvent(
            name="cost_budget",
            old_hash="a",
            new_hash="b",
            trigger="agent",
            tier=2,
            is_self_modify=True,
        )
    )
    # The sink runs on a scheduled task; let it drain.
    await asyncio.sleep(0)
    assert len(out) == 1
    meta = out[0]["metadata"]
    assert meta["kind"] == "extension_reload"
    assert meta["is_self_modify"] is True
    assert meta["trigger"] == "agent"
