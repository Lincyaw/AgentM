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


async def test_short_turn_does_not_fill_viewport() -> None:
    # Regression: AssistantTurn/SystemTurn extend Textual's Vertical, which
    # defaults to height:1fr — a single short turn would stretch to fill the
    # whole transcript. The .turn { height: auto } rule keeps it content-sized.
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        client.push(_frame("turn_start", turn_id=1, turn_index=0))
        client.push(_frame("stream_text", content="Hi", turn_id=1))
        client.push(_frame("assistant_text", content="Hi"))
        await pilot.pause(0.2)
        turn = app.query_one(AssistantTurn)
        # Attribution label + one text line, not the full viewport height.
        assert turn.size.height <= 4


async def test_idle_flush_is_a_noop_between_deltas() -> None:
    # Regression: the 20Hz flush timer must not repaint when nothing arrived.
    # flush() returns False on an idle tick so the app skips scroll_end (an
    # unconditional scroll forces a layout/repaint every tick during a slow
    # thinking phase).
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        client.push(_frame("turn_start", turn_id=1))
        client.push(_frame("stream_thinking", content="pondering", turn_id=1))
        await pilot.pause(0.15)
        turn = app.query_one(AssistantTurn)
        assert await turn.flush() is False  # delta already applied; idle now
        turn.append_thinking(" more")
        assert await turn.flush() is True  # new delta -> one repaint


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


async def test_typing_slash_shows_inline_suggestions_from_registry() -> None:
    from agentm_terminal.frontends.tui.widgets import CommandSuggestions

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
        assert "/new" in app.catalog.commands and "/status" in app.catalog.commands

        inp = app.query_one("#prompt-input")
        sug = app.query_one("#suggestions", CommandSuggestions)
        assert sug.display is False  # hidden until a slash is typed
        inp.text = "/"  # type: ignore[attr-defined]
        await pilot.pause(0.05)
        # Inline panel (not a modal) lists /clear (client-local) + registered.
        shown = [
            str(sug.get_option_at_index(i).prompt) for i in range(sug.option_count)
        ]
        assert sug.display is True
        assert {"/clear", "/new", "/status"} <= set(shown)

        # Prefix filtering as you type.
        inp.text = "/st"  # type: ignore[attr-defined]
        await pilot.pause(0.05)
        shown = [
            str(sug.get_option_at_index(i).prompt) for i in range(sug.option_count)
        ]
        assert shown == ["/status"]


async def test_tab_completes_highlighted_suggestion() -> None:
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        client.push(_frame("session_ready", command_names=["status"], model="m"))
        await pilot.pause(0.1)
        inp = app.query_one("#prompt-input")
        inp.text = "/st"  # type: ignore[attr-defined]
        await pilot.pause(0.05)
        await pilot.press("tab")
        await pilot.pause(0.05)
        assert inp.text == "/status "  # type: ignore[attr-defined]  # completed + space
        # Completing past the command hides the popup.
        from agentm_terminal.frontends.tui.widgets import CommandSuggestions

        assert app.query_one("#suggestions", CommandSuggestions).display is False


async def test_non_clear_slash_is_forwarded_to_gateway() -> None:
    # Only /clear is handled locally now; everything else (registered atom /
    # gateway commands) is forwarded verbatim to the gateway command router.
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#prompt-input")
        inp.text = "/status"  # type: ignore[attr-defined]
        inp.action_submit()  # type: ignore[attr-defined]
        await pilot.pause(0.1)
        assert any(b.get("content") == "/status" for b in client.sent)


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


async def test_clear_wipes_transcript_and_cold_resets_gateway() -> None:
    # /clear must clear records AND reset the session: wiping the UI alone
    # leaves the model's context intact on the gateway. It forwards /end.
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#prompt-input")
        inp.text = "hello"  # type: ignore[attr-defined]
        inp.action_submit()  # type: ignore[attr-defined]
        await pilot.pause(0.1)
        assert len(app.transcript.children) == 1
        inp.text = "/clear"  # type: ignore[attr-defined]
        inp.action_submit()  # type: ignore[attr-defined]
        await pilot.pause(0.1)
        assert len(app.transcript.children) == 0  # transcript wiped
        assert any(b.get("content") == "/end" for b in client.sent)  # session reset


async def test_slash_stays_in_box_no_modal_screen() -> None:
    # The slash is typed literally (the box keeps "/"); suggestions render
    # inline, never as a separate modal screen pushed over the transcript.
    client = _FakeClient()
    app = AgentMTui(client=client, sender_id="local", chat_id="terminal")
    async with app.run_test(size=(100, 30)) as pilot:
        inp = app.query_one("#prompt-input")
        inp.text = "/"  # type: ignore[attr-defined]
        await pilot.pause(0.05)
        assert inp.text == "/"  # type: ignore[attr-defined]
        assert app.screen is app.screen_stack[0]  # no modal pushed


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
