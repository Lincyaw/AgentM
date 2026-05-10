from __future__ import annotations

import asyncio

import pytest

from agentm_channels.bus import MessageBus, OutboundMessage
from agentm_channels.manager import ChannelManager
from agentm_channels.registry import discover_all


def test_registry_discovers_stub_and_feishu() -> None:
    found = discover_all()
    assert "stub" in found
    assert "feishu" in found  # bundled built-in


@pytest.mark.asyncio
async def test_manager_skips_disabled_channel() -> None:
    bus = MessageBus()
    mgr = ChannelManager({"stub": {"enabled": False}}, bus)
    assert mgr.channels == {}


@pytest.mark.asyncio
async def test_manager_dispatches_outbound_to_stub() -> None:
    bus = MessageBus()
    mgr = ChannelManager({"stub": {"enabled": True, "allow_from": ["*"]}}, bus)
    assert "stub" in mgr.channels
    await mgr.start()
    try:
        await bus.publish_outbound(
            OutboundMessage(channel="stub", chat_id="c", content="hi")
        )
        # Give the dispatch loop a tick.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if mgr.channels["stub"].outbox:  # type: ignore[attr-defined]
                break
        outbox = mgr.channels["stub"].outbox  # type: ignore[attr-defined]
        assert len(outbox) == 1
        assert outbox[0].content == "hi"
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_manager_drops_outbound_for_unknown_channel() -> None:
    bus = MessageBus()
    mgr = ChannelManager({"stub": {"enabled": True, "allow_from": ["*"]}}, bus)
    await mgr.start()
    try:
        await bus.publish_outbound(
            OutboundMessage(channel="not-a-channel", chat_id="c", content="hi")
        )
        # Brief settle window — message should be silently dropped, not crash.
        await asyncio.sleep(0.1)
        assert mgr.channels["stub"].outbox == []  # type: ignore[attr-defined]
    finally:
        await mgr.stop()
