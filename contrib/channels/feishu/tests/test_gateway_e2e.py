"""Stub-driven gateway E2E.

Exercises the full FeishuGateway.run loop without lark-oapi or a real
LLM. The session_factory hands back a :class:`_FakeSession` whose
``prompt(text)`` synthesizes a turn by emitting events on the bus —
the same wire shape the real kernel uses — so the gateway's renderers
fire and we can assert the outbound message stream against the stub
chat source.
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

from agentm_feishu.approval import ApprovalPolicy
from agentm_feishu.chat_source import CardActionEvent, InboundMessage, StubChatSource
from agentm_feishu.gateway import FeishuGateway, GatewayConfig


class _FakeSession:
    """Minimal stand-in for ``AgentSession`` used by the gateway tests.

    Each ``prompt(text)`` invocation simulates one turn by emitting:
        TurnEndEvent(AssistantMessage(text=...))
    optionally preceded by ToolCall/ToolResult events when ``text`` is
    a tool-call directive (``"!tool <name> <args_json>"``).
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self.session_manager = _FakeSessionManager()
        self.prompts: list[str] = []

    async def prompt(self, text: str) -> None:
        self.prompts.append(text)
        if text.startswith("!tool "):
            _, tool_name, *rest = text.split(" ", 2)
            args = {"argv": rest[0] if rest else ""}
            tc = ToolCallEvent(tool_call_id="t1", tool_name=tool_name, args=args)
            results = await self._bus.emit(ToolCallEvent.CHANNEL, tc)
            blocked = next(
                (r for r in results if isinstance(r, dict) and r.get("block")),
                None,
            )
            if blocked is None:
                tr = ToolResult(
                    content=[TextContent(type="text", text="ok")],
                    is_error=False,
                )
                await self._bus.emit(
                    ToolResultEvent.CHANNEL,
                    ToolResultEvent(tool_call_id="t1", tool_name=tool_name, result=tr),
                )
                reply = f"called {tool_name}"
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
        return None


class _FakeSessionManager:
    def get_session_id(self) -> str:
        return "fake-session-1"


async def _factory(_cwd: str, bus: EventBus, _resume: str | None) -> Any:
    return _FakeSession(bus)


def _make_gateway(tmp_path: Path, **kwargs: Any) -> tuple[FeishuGateway, StubChatSource]:
    src = StubChatSource()
    config = GatewayConfig(
        cwd=str(tmp_path),
        scenario=None,
        state_dir=tmp_path / "state",
        **kwargs,
    )
    gateway = FeishuGateway(source=src, config=config, session_factory=_factory)
    return gateway, src


async def _run_until_outbox(
    gateway: FeishuGateway,
    src: StubChatSource,
    *,
    expect_kinds: list[str],
    timeout: float = 2.0,
) -> None:
    """Wait until the outbox contains entries matching ``expect_kinds`` (in order)."""

    async def watch() -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            kinds = [e["kind"] for e in src.outbox]
            if kinds[: len(expect_kinds)] == expect_kinds:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(
            f"timeout waiting for outbox kinds {expect_kinds!r}; got {[e['kind'] for e in src.outbox]!r}"
        )

    runner = asyncio.create_task(gateway.run())
    try:
        await watch()
    finally:
        await src.close()
        try:
            await asyncio.wait_for(runner, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            runner.cancel()


@pytest.mark.asyncio
async def test_inbound_message_yields_assistant_reply(tmp_path: Path) -> None:
    gateway, src = _make_gateway(tmp_path)
    src.push_message(InboundMessage(chat_id="c1", user_id="u1", text="hello"))
    await _run_until_outbox(gateway, src, expect_kinds=["text"])
    assert src.outbox[0]["chat_id"] == "c1"
    assert src.outbox[0]["text"] == "echo: hello"


@pytest.mark.asyncio
async def test_chat_session_id_persisted_after_first_message(tmp_path: Path) -> None:
    gateway, src = _make_gateway(tmp_path)
    src.push_message(InboundMessage(chat_id="c1", user_id="u1", text="hi"))
    await _run_until_outbox(gateway, src, expect_kinds=["text"])
    persisted = (tmp_path / "state" / "chat_sessions.json").read_text()
    assert "fake-session-1" in persisted


@pytest.mark.asyncio
async def test_unapproved_tool_call_is_blocked(tmp_path: Path) -> None:
    gateway, src = _make_gateway(
        tmp_path,
        approval_policy=ApprovalPolicy(
            always_block=frozenset({"bash"}),
        ),
    )
    src.push_message(
        InboundMessage(chat_id="c1", user_id="u1", text="!tool bash echo")
    )
    await _run_until_outbox(gateway, src, expect_kinds=["text"])
    # The approval bridge blocked synchronously (always_block); the only
    # outbound render is the assistant's wrap-up text (the fake session
    # echoes the block reason as its assistant reply).
    assert "blocked: tool 'bash' is denied" in src.outbox[0]["text"]


@pytest.mark.asyncio
async def test_approve_flow_lets_tool_through(tmp_path: Path) -> None:
    gateway, src = _make_gateway(
        tmp_path,
        approval_policy=ApprovalPolicy(
            require_approval=frozenset({"bash"}),
            timeout_seconds=2.0,
        ),
    )

    async def auto_approver() -> None:
        # Wait for the approval card, click approve.
        for _ in range(200):
            for entry in src.outbox:
                if entry["kind"] == "card":
                    approve = entry["card"]["body"]["elements"][1]["actions"][0]
                    src.push_card_action(
                        CardActionEvent(
                            card_id=approve["value"]["card_id"],
                            user_id="u1",
                            action="approve",
                        )
                    )
                    return
            await asyncio.sleep(0.01)

    src.push_message(
        InboundMessage(chat_id="c1", user_id="u1", text="!tool bash echo")
    )
    auto = asyncio.create_task(auto_approver())
    # Expected outbox: approval card → resolved card update → tool_result
    # render (`↩ bash → ok`) → assistant turn-end text (`called bash`).
    await _run_until_outbox(
        gateway,
        src,
        expect_kinds=["card", "update_card", "text", "text"],
        timeout=3.0,
    )
    await auto
    # Final assistant text should report success, not block.
    assistant_texts = [e for e in src.outbox if e["kind"] == "text"]
    assert any("called bash" in t["text"] for t in assistant_texts)
