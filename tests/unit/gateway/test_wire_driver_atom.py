"""Fail-stop: wire_driver atom event -> outbound translation (§4).

wire_driver is the only new code in channels v2; it is the single bridge
from a session's events to the chat surface. If a turn_end produces no
assistant-text outbound, or a write-class tool_call is not gated through
the approval manager, the gateway is mute or unsafe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi.events import DiagnosticEvent, ToolCallEvent, TurnEndEvent
from agentm.core.abi.messages import AssistantMessage, TextContent

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
    """Records ``@api.on`` handlers and serves ``get_service`` lookups."""

    def __init__(self, services: dict[str, Any]) -> None:
        self._services = services
        self.handlers: dict[str, Any] = {}

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    def on(self, channel: str, handler: Any, **_kw: Any) -> None:
        self.handlers[channel] = handler


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
    await api.handlers[TurnEndEvent.CHANNEL](
        TurnEndEvent(turn_index=0, message=_assistant("the answer is 42"))
    )
    assert len(out) == 1
    body = out[0]
    assert body["content"] == "the answer is 42"
    assert body["metadata"]["kind"] == "assistant_text"
    assert body["channel"] == "terminal"
    assert body["chat_id"] == "t1"


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
    await api.handlers[TurnEndEvent.CHANNEL](
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
    result = await api.handlers[ToolCallEvent.CHANNEL](
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
    result = await api.handlers[ToolCallEvent.CHANNEL](
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
    await api.handlers[DiagnosticEvent.CHANNEL](
        DiagnosticEvent(level="error", source="loop", message="boom")
    )
    assert out[0]["metadata"]["kind"] == "diagnostic_error"
    assert out[0]["content"] == "boom"


def test_install_without_services_fails_fast() -> None:
    mod = _load_wire_driver()
    api = _FakeAPI({})  # no wire_outbound / session_key
    with pytest.raises(RuntimeError):
        mod.install(api, {})
