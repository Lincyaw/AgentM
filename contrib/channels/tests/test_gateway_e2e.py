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
                    content=[
                        TextContent(
                            type="text",
                            text="line one\nline two\nline three\n",
                        )
                    ],
                    is_error=False,
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




