"""End-to-end gateway test using StubChannel + a fake AgentSession.

Drives the full flow: stub.push() → MessageBus.inbound → gateway →
session.prompt() (which emits TurnEnd / ToolCall / ToolResult on its
own EventBus) → MessageBus.outbound → ChannelManager → StubChannel.send.
No lark-oapi or real LLM in the loop — proves the architectural seams.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    EventBus,
    TextContent,
    ToolCallEvent,
    ToolResultEvent,
)
from agentm.core.abi.events import TurnEndEvent
from agentm.core.abi.tool import ToolResult

from agentm_channels.approval import ApprovalPolicy
from agentm_channels.bus import MessageBus
from agentm_channels.gateway import Gateway, GatewayConfig
from agentm_channels.manager import ChannelManager


class _FakeSM:
    def get_session_id(self) -> str:
        return f"fake-{int(time.time()*1000)}"


class _FakeSession:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self.session_manager = _FakeSM()

    async def prompt(self, text: str) -> None:
        if text.strip().lower() == "run-tool":
            tc = ToolCallEvent(
                tool_call_id="t1", tool_name="bash", args={"cmd": "echo hi"}
            )
            results = await self._bus.emit(ToolCallEvent.CHANNEL, tc)
            blocked = next(
                (r for r in results if isinstance(r, dict) and r.get("block")), None
            )
            if blocked is None:
                tr = ToolResult(
                    content=[TextContent(type="text", text="hi\n")], is_error=False
                )
                await self._bus.emit(
                    ToolResultEvent.CHANNEL,
                    ToolResultEvent(tool_call_id="t1", tool_name="bash", result=tr),
                )
                reply = "ran bash"
            else:
                reply = f"blocked: {blocked['reason']}"
        else:
            reply = f"echo: {text}"
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=reply)],
            timestamp=time.time(),
        )
        await self._bus.emit(
            TurnEndEvent.CHANNEL,
            TurnEndEvent(turn_index=0, message=msg, messages=()),
        )

    async def shutdown(self) -> None:
        pass


async def _factory(_cwd: str, bus: EventBus, _resume: str | None) -> Any:
    return _FakeSession(bus)


async def _build(tmp_path: Path, **kwargs: Any) -> tuple[MessageBus, ChannelManager, Gateway]:
    bus = MessageBus()
    mgr = ChannelManager({"stub": {"enabled": True, "allow_from": ["*"]}}, bus)
    gw = Gateway(
        bus=bus,
        config=GatewayConfig(cwd=str(tmp_path), state_dir=tmp_path / "state", **kwargs),
        session_factory=_factory,
    )
    await mgr.start()
    await gw.start()
    return bus, mgr, gw


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


@pytest.mark.asyncio
async def test_inbound_message_yields_assistant_reply(tmp_path: Path) -> None:
    bus, mgr, gw = await _build(tmp_path)
    try:
        stub = mgr.channels["stub"]
        await stub.push(sender_id="u1", chat_id="c1", content="hello")  # type: ignore[attr-defined]
        await _wait_for(lambda: any(o.content == "echo: hello" for o in stub.outbox))  # type: ignore[attr-defined]
        delivered = stub.outbox  # type: ignore[attr-defined]
        assert delivered[0].channel == "stub"
        assert delivered[0].chat_id == "c1"
    finally:
        await gw.stop()
        await mgr.stop()


@pytest.mark.asyncio
async def test_session_id_persisted_after_first_message(tmp_path: Path) -> None:
    bus, mgr, gw = await _build(tmp_path)
    try:
        stub = mgr.channels["stub"]
        await stub.push(sender_id="u1", chat_id="c1", content="hi")  # type: ignore[attr-defined]
        await _wait_for(lambda: bool(stub.outbox))  # type: ignore[attr-defined]
        persisted = (tmp_path / "state" / "session_map.json").read_text()
        assert "stub:c1" in persisted
        assert "fake-" in persisted
    finally:
        await gw.stop()
        await mgr.stop()


@pytest.mark.asyncio
async def test_blocked_tool_call_is_reported(tmp_path: Path) -> None:
    bus, mgr, gw = await _build(
        tmp_path,
        approval_policy=ApprovalPolicy(always_block=frozenset({"bash"})),
    )
    try:
        stub = mgr.channels["stub"]
        await stub.push(sender_id="u1", chat_id="c1", content="run-tool")  # type: ignore[attr-defined]
        await _wait_for(
            lambda: any(  # noqa: E501
                "blocked: tool 'bash'" in o.content for o in stub.outbox  # type: ignore[attr-defined]
            )
        )
    finally:
        await gw.stop()
        await mgr.stop()


@pytest.mark.asyncio
async def test_approval_card_round_trip_lets_tool_through(tmp_path: Path) -> None:
    bus, mgr, gw = await _build(
        tmp_path,
        approval_policy=ApprovalPolicy(
            require_approval=frozenset({"bash"}), timeout_seconds=2.0
        ),
    )
    try:
        stub = mgr.channels["stub"]

        async def auto_approver() -> None:
            for _ in range(400):
                for entry in list(stub.outbox):  # type: ignore[attr-defined]
                    if entry.metadata.get("kind") == "approval_request":
                        approve_value = entry.buttons[0].value
                        await stub.push(  # type: ignore[attr-defined]
                            sender_id="u1",
                            chat_id="c1",
                            content="(approve)",
                            button_value=approve_value,
                        )
                        return
                await asyncio.sleep(0.01)

        await stub.push(sender_id="u1", chat_id="c1", content="run-tool")  # type: ignore[attr-defined]
        approver = asyncio.create_task(auto_approver())
        await _wait_for(
            lambda: any("ran bash" in o.content for o in stub.outbox),  # type: ignore[attr-defined]
            timeout=3.0,
        )
        await approver
    finally:
        await gw.stop()
        await mgr.stop()


@pytest.mark.asyncio
async def test_tool_call_streams_intermediate_outbound(tmp_path: Path) -> None:
    """Fail-stop: tool calls produce live stream frames mid-turn.

    Before this feature the user only saw the assistant's final
    TurnEnd text — there was no on-screen signal that a tool was
    running. The contract this test locks down:

    * a ``🔧 <tool>(…)`` frame is published BEFORE the tool result
      (i.e. mid-turn, not buffered until TurnEnd)
    * tool-call frame and result frame share the same ``stream_id``
      so stream-aware channels patch one card
    * the final TurnEnd frame carries ``final=True``
    """
    bus, mgr, gw = await _build(tmp_path)
    try:
        stub = mgr.channels["stub"]
        await stub.push(sender_id="u1", chat_id="c1", content="run-tool")  # type: ignore[attr-defined]
        await _wait_for(
            lambda: any(o.final for o in stub.outbox),  # type: ignore[attr-defined]
            timeout=2.0,
        )
        streamed = [
            o for o in stub.outbox if o.stream_id is not None  # type: ignore[attr-defined]
        ]
        assert streamed, "no streamed outbound frames were published"
        stream_ids = {o.stream_id for o in streamed}
        assert len(stream_ids) == 1, (
            f"tool call and final frame must share one stream_id; got {stream_ids}"
        )
        tool_frames = [o for o in streamed if "🔧 bash" in o.content]
        assert tool_frames, "no 🔧 tool-call frame was emitted"
        # The first 🔧 frame must precede the final frame on the queue,
        # proving the user sees the intent mid-turn rather than
        # post-hoc on TurnEnd.
        first_tool_idx = stub.outbox.index(tool_frames[0])  # type: ignore[attr-defined]
        final_frames = [o for o in streamed if o.final]
        assert final_frames, "TurnEnd frame missing final=True"
        last_final_idx = stub.outbox.index(final_frames[-1])  # type: ignore[attr-defined]
        assert first_tool_idx < last_final_idx
        # The final frame should reflect both the tool result (✓ bash …)
        # and the assistant's reply text, all in one rendered card.
        last_final = final_frames[-1]
        assert "✓ bash" in last_final.content
        assert "ran bash" in last_final.content
    finally:
        await gw.stop()
        await mgr.stop()
