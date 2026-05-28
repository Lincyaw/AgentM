"""ui: Pilot smoke of the TUI render path — wire frame -> widget mutation.

Fail-stop for the router: if a streamed turn does not produce exactly one
AssistantTurn keyed to the stream, or a tool_call+tool_result does not produce
a ToolBlock with the right status glyph, the live surface is broken. Marked
``ui`` (opt-in: run with ``-m ui``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentm_terminal.frontends.tui.app import AgentMTui
from agentm_terminal.frontends.tui.theme import TOOL_OK
from agentm_terminal.frontends.tui.widgets import AssistantTurn, ToolBlock, UserTurn

pytestmark = pytest.mark.ui


class _FakeClient:
    """Duck-typed stand-in for TerminalClient: a controllable outbound stream
    + a record of inbound sends."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []

    async def connect(self) -> None:
        return None

    async def send_inbound(self, body: dict[str, Any]) -> None:
        self.sent.append(body)

    async def outbound(self) -> Any:
        while True:
            body = await self._q.get()
            if body is None:
                return
            yield body

    async def close(self) -> None:
        self._q.put_nowait(None)

    def push(self, body: dict[str, Any]) -> None:
        self._q.put_nowait(body)


def _frame(kind: str, *, content: str = "", **meta: Any) -> dict[str, Any]:
    return {
        "channel": "terminal",
        "content": content,
        "metadata": {"kind": kind, **meta},
    }


async def test_streamed_turn_and_tool_block_render() -> None:
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        client.push(_frame("turn_start", turn_id=1, turn_index=0))
        client.push(_frame("stream_text", content="Hello ", turn_id=1))
        client.push(_frame("stream_text", content="world", turn_id=1))
        client.push(_frame("tool_call", tool_call_id="t1", name="bash", args={"cmd": "ls"}))
        client.push(_frame("tool_result", tool_call_id="t1", ok=True, content="file.txt"))
        client.push(_frame("assistant_text", content="Hello world"))
        await pilot.pause(0.2)

        assert len(app.query(AssistantTurn)) == 1
        blocks = list(app.query(ToolBlock))
        assert len(blocks) == 1
        assert TOOL_OK in str(blocks[0].title)


async def test_submit_echoes_user_turn_and_sends_inbound() -> None:
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#prompt-input")
        inp.text = "hi there"  # type: ignore[attr-defined]
        inp.action_submit()  # type: ignore[attr-defined]
        await pilot.pause(0.1)

        assert any(b.get("content") == "hi there" for b in client.sent)
        assert len(app.query(UserTurn)) == 1
