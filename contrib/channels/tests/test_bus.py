from __future__ import annotations

import pytest

from agentm_channels.bus import InboundMessage, MessageBus, OutboundMessage


@pytest.mark.asyncio
async def test_session_key_defaults_to_channel_chat() -> None:
    msg = InboundMessage(channel="feishu", sender_id="u", chat_id="c", content="hi")
    assert msg.session_key == "feishu:c"


@pytest.mark.asyncio
async def test_session_key_override_wins() -> None:
    msg = InboundMessage(
        channel="feishu",
        sender_id="u",
        chat_id="c",
        content="hi",
        session_key_override="feishu:c::omt_thread1",
    )
    assert msg.session_key == "feishu:c::omt_thread1"


@pytest.mark.asyncio
async def test_bus_round_trip() -> None:
    bus = MessageBus()
    inbound = InboundMessage(channel="stub", sender_id="u", chat_id="c", content="hi")
    await bus.publish_inbound(inbound)
    got = await bus.consume_inbound()
    assert got is inbound
    outbound = OutboundMessage(channel="stub", chat_id="c", content="bye")
    await bus.publish_outbound(outbound)
    assert (await bus.consume_outbound()) is outbound
