"""TerminalChannel smoke test.

Drives the channel against piped stdin so a real-process invocation
(``echo /help | agentm-gateway --terminal``) is covered without
spawning a subprocess. We rebind ``sys.stdin`` to an in-memory
StringIO; the read loop's ``run_in_executor(None, sys.stdin.readline)``
goes through whatever ``sys.stdin`` resolves to at call time.
"""

from __future__ import annotations

import asyncio
import io
import sys

import pytest

from agentm_channels.bus import MessageBus, OutboundKind, OutboundMessage
from agentm_channels.channels.terminal import TerminalChannel
from agentm_channels.registry import discover_all


def test_terminal_channel_is_discoverable() -> None:
    found = discover_all()
    assert "terminal" in found
    assert found["terminal"] is TerminalChannel


@pytest.mark.asyncio
async def test_terminal_channel_pipes_inbound_messages_to_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = MessageBus()
    stdin = io.StringIO("hello\n=approval-deadbeef:approve\n")
    monkeypatch.setattr(sys, "stdin", stdin)
    chan = TerminalChannel(
        {"enabled": True, "allow_from": ["*"], "color": False}, bus
    )
    task = asyncio.create_task(chan.start())
    try:
        # First line → plain inbound.
        msg1 = await asyncio.wait_for(bus.consume_inbound(), timeout=2.0)
        assert msg1.channel == "terminal"
        assert msg1.content == "hello"
        assert msg1.button_value is None
        # Second line starts with '=' → button-click round-trip.
        msg2 = await asyncio.wait_for(bus.consume_inbound(), timeout=2.0)
        assert msg2.button_value == "approval-deadbeef:approve"
        # Third read returns "" (EOF) and the channel stops itself.
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        await chan.stop()


@pytest.mark.asyncio
async def test_terminal_channel_send_renders_buttons(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentm_channels.bus import Button

    bus = MessageBus()
    # No stdin needed — we don't start() the channel, just call send().
    chan = TerminalChannel(
        {"enabled": True, "allow_from": ["*"], "color": False}, bus
    )
    await chan.send(
        OutboundMessage(
            channel="terminal",
            chat_id="terminal",
            content="approve `bash`?",
            buttons=[
                Button(label="Approve", value="ap-1:approve", style="primary"),
                Button(label="Deny", value="ap-1:deny", style="danger"),
            ],
        )
    )
    captured = capsys.readouterr().out
    assert "approve `bash`?" in captured
    assert "[1] Approve" in captured
    assert "[2] Deny" in captured
    assert "ap-1:approve" in captured  # value visible so user can copy/paste


@pytest.mark.asyncio
async def test_turn_complete_kind_does_not_print_empty_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = MessageBus()
    chan = TerminalChannel(
        {"enabled": True, "allow_from": ["*"], "color": False}, bus
    )
    await chan.send(
        OutboundMessage(
            channel="terminal",
            chat_id="terminal",
            kind=OutboundKind.TURN_COMPLETE,
        )
    )
    captured = capsys.readouterr().out
    assert "turn complete" in captured
