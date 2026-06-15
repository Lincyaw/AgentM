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

from agentm.core.abi import (
    DiagnosticEvent,
    ExtensionReloadEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from agentm.core.abi import AssistantMessage, TextContent, Usage
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
        self.registered: dict[str, Any] = {}

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    def set_service(self, name: str, obj: Any) -> None:
        self.registered[name] = obj
        self._services[name] = obj

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


# --- session_ready carries the model-profile list --------------------------


def test_session_ready_carries_models() -> None:
    """The session_ready frame must advertise the configured model-profile
    names (same source as the gateway /model command) so a chat client can
    populate a model switcher without a second round trip."""
    from agentm.core.abi import Model
    from agentm.core.abi.events import SessionReadyEvent

    mod = _load_wire_driver()
    # The gateway seeds the model-profile names via the ``model_names`` service;
    # install() binds them into the session_ready projector (the atom does not
    # read user config itself — §11.4.6).
    project = mod._make_session_ready_projector(["doubao", "glm"])

    ev = SessionReadyEvent(
        cwd="/tmp",
        session_id="s1",
        tool_names=("read",),
        command_names=("/help",),
        extension_module_paths=(),
        model=Model(id="doubao-seed", provider="openai", context_window=1, max_output_tokens=1),
        root_session_id="r1",
    )
    body = project(ev)
    assert body["kind"] == "session_ready"
    assert body["model"] == "doubao-seed"
    assert body["models"] == ["doubao", "glm"]


# --- child-session trajectory forwarding -----------------------------------


class _FakeChild:
    """Minimal child-session stand-in: a real EventBus + a session_id."""

    def __init__(self, bus: Any, session_id: str) -> None:
        self.bus = bus
        self.session_id = session_id


@pytest.mark.asyncio
async def test_child_trajectory_forwarded_with_child_id() -> None:
    """A spawned child's own stream/tool/turn events must reach the PARENT's
    wire_outbound, each stamped metadata.child_id; parent frames carry none."""
    from agentm.core.abi import EventBus

    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install_multi(sink)
    forwarder = api.registered["child_wire_forwarder"]
    assert forwarder is not None

    child_bus = EventBus()
    forwarder(_FakeChild(child_bus, "child-abc"))

    # A parent frame first — it must NOT carry child_id.
    await api.handlers[StreamDeltaEvent.CHANNEL][0](
        StreamDeltaEvent(turn_index=0, delta=TextDelta(text="parent"), turn_id=1)
    )
    # Now the child's own bus emits its trajectory.
    await child_bus.emit(
        StreamDeltaEvent.CHANNEL,
        StreamDeltaEvent(turn_index=0, delta=TextDelta(text="child-tok"), turn_id=2),
    )
    await child_bus.emit(
        ToolCallEvent.CHANNEL,
        ToolCallEvent(tool_call_id="tc1", tool_name="bash", args={"cmd": "ls"}),
    )

    parent_frame = out[0]
    assert parent_frame["content"] == "parent"
    assert "child_id" not in parent_frame["metadata"]

    child_frames = [b for b in out if b["metadata"].get("child_id") == "child-abc"]
    kinds = {b["metadata"]["kind"] for b in child_frames}
    assert kinds == {"stream_text", "tool_call"}
    stream = next(b for b in child_frames if b["metadata"]["kind"] == "stream_text")
    assert stream["content"] == "child-tok"
    # Child frames inherit the parent turn_context routing.
    assert stream["channel"] == "terminal"
    assert stream["_session_key"] == "terminal:t1"


@pytest.mark.asyncio
async def test_child_lifecycle_markers_not_double_emitted() -> None:
    """child_session_start/end belong to the parent bus; forwarding the child
    bus must not re-emit them (they'd double up)."""
    from agentm.core.abi import ChildSessionStartEvent, EventBus

    out: list[dict] = []

    async def sink(body: dict) -> None:
        out.append(body)

    api = _install_multi(sink)
    child_bus = EventBus()
    api.registered["child_wire_forwarder"](_FakeChild(child_bus, "child-xyz"))

    await child_bus.emit(
        ChildSessionStartEvent.CHANNEL,
        ChildSessionStartEvent(
            child_session_id="grandchild",
            parent_session_id="child-xyz",
            purpose="nested",
        ),
    )
    assert out == []


def test_child_forwarder_skips_handle_without_bus() -> None:
    out: list[dict] = []

    async def sink(body: dict) -> None:  # pragma: no cover - never called
        out.append(body)

    api = _install_multi(sink)
    # A child object missing bus/session_id is silently skipped.
    api.registered["child_wire_forwarder"](object())
    assert out == []
