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


async def test_esc_interrupts_an_in_flight_turn() -> None:
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#prompt-input")
        inp.text = "long task"  # type: ignore[attr-defined]
        inp.action_submit()  # type: ignore[attr-defined]
        await pilot.pause(0.1)
        # A turn is now in flight; Esc must send an out-of-band interrupt.
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert any(b.get("control") == "interrupt" for b in client.sent)

        # Once the loop ends (agent_end), Esc no longer interrupts.
        client.push(_frame("agent_end", cause="SignalAborted"))
        await pilot.pause(0.1)
        before = len([b for b in client.sent if b.get("control") == "interrupt"])
        await pilot.press("escape")
        await pilot.pause(0.1)
        after = len([b for b in client.sent if b.get("control") == "interrupt"])
        assert after == before  # idle Esc does not send another interrupt


async def test_command_palette_lists_local_and_gateway_commands() -> None:
    from agentm_terminal.frontends.tui.modals import CommandPalette

    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        client.push(
            _frame(
                "session_ready",
                tool_names=["bash", "read"],
                command_names=["new", "status"],
                model="m",
            )
        )
        await pilot.pause(0.1)
        assert app.catalog.tools == ["bash", "read"]
        assert "/new" in app.catalog.commands and "/status" in app.catalog.commands

        await pilot.press("ctrl+r")
        await pilot.pause(0.1)
        assert isinstance(app.screen, CommandPalette)
        # built-ins + the gateway commands surfaced from session_ready.
        assert "/help" in app.screen._all and "/new" in app.screen._all


async def test_tools_slash_opens_info_modal() -> None:
    from agentm_terminal.frontends.tui.modals import InfoModal

    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        client.push(_frame("session_ready", tool_names=["bash"], model="m"))
        await pilot.pause(0.1)
        inp = app.query_one("#prompt-input")
        inp.text = "/tools"  # type: ignore[attr-defined]
        inp.action_submit()  # type: ignore[attr-defined]
        await pilot.pause(0.1)
        assert isinstance(app.screen, InfoModal)
        assert "bash" in app.screen._body
        # A local slash command is NOT forwarded to the gateway.
        assert all(b.get("content") != "/tools" for b in client.sent)


async def test_input_history_up_down_cycles_prior_inputs() -> None:
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#prompt-input")
        for text in ("first", "second"):
            inp.text = text  # type: ignore[attr-defined]
            inp.action_submit()  # type: ignore[attr-defined]
            await pilot.pause(0.05)
        # Input is empty; Up walks back (newest first), Down walks forward.
        await pilot.press("up")
        await pilot.pause(0.05)
        assert inp.text == "second"  # type: ignore[attr-defined]
        await pilot.press("up")
        await pilot.pause(0.05)
        assert inp.text == "first"  # type: ignore[attr-defined]
        await pilot.press("down")
        await pilot.pause(0.05)
        assert inp.text == "second"  # type: ignore[attr-defined]
        await pilot.press("down")
        await pilot.pause(0.05)
        assert inp.text == ""  # type: ignore[attr-defined]


async def test_slash_on_empty_line_opens_palette() -> None:
    from agentm_terminal.frontends.tui.modals import CommandPalette

    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#prompt-input")
        inp.text = "/"  # type: ignore[attr-defined]  # simulate typing a slash
        await pilot.pause(0.1)
        assert isinstance(app.screen, CommandPalette)
        assert inp.text == ""  # type: ignore[attr-defined]  # the slash was consumed


async def test_tool_block_title_shows_arg_summary() -> None:
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        client.push(_frame("turn_start", turn_id=1))
        client.push(
            _frame(
                "tool_call",
                tool_call_id="t1",
                name="bash",
                args={"command": "ls -la"},
            )
        )
        await pilot.pause(0.15)
        blocks = list(app.query(ToolBlock))
        assert len(blocks) == 1
        assert "bash(ls -la)" in str(blocks[0].title)
