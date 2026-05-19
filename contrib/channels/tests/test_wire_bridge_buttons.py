"""Phase 4 fail-stop: WireBridge must serialize buttons / metadata on
outbound and forward ``button_value`` on inbound.

Without this, the approval round-trip (the entire reason for typed
:class:`Button`s on :class:`OutboundMessage`) is broken across the wire
— the gateway publishes buttons, the client never sees them, the user
can't approve, the agent hangs. The two assertions here are the
shortest possible reproduction of the bug Phase 4 closes.

These are unit-shaped: an in-memory :class:`OutboxStore` records what
the synthetic ``_WireChannel`` would have shipped, so the test stays
fast and independent of the Unix-socket transport.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentm_channels.bus import (
    Button,
    InboundMessage,
    MessageBus,
    OutboundKind,
    OutboundMessage,
)
from agentm_channels.manager import ChannelManager
from agentm_channels.peer import PeerSession
from agentm_channels.wire import KIND_INBOUND, WIRE_VERSION, Envelope
from agentm_channels.wire_bridge import WireBridge, _WireChannel


@dataclass
class _RecordedOutbox:
    """Drop-in stand-in for :class:`SqliteOutbox` capturing
    ``enqueue(peer_id, env)`` calls.

    Only the surface ``_WireChannel`` touches is implemented — the
    delivery worker / lease machinery is out of scope here.
    """

    items: list[tuple[str, Envelope]] = field(default_factory=list)

    def enqueue(self, peer_id: str, env: Envelope) -> None:
        self.items.append((peer_id, env))


def _new_bus() -> MessageBus:
    # MessageBus binds its home loop on first construction inside a
    # running loop; the fixture below is async so this is safe.
    return MessageBus()


@pytest.mark.asyncio
async def test_wirebridge_serializes_buttons_and_metadata() -> None:
    bus = _new_bus()
    manager = ChannelManager({}, bus)
    outbox = _RecordedOutbox()
    ch = _WireChannel(
        {"enabled": True, "allow_from": ["*"]},
        bus,
        peer_id="peer-1",
        channel_name="terminal",
        outbox=outbox,  # type: ignore[arg-type]
    )
    manager.inject_channel("terminal", ch)

    msg = OutboundMessage(
        channel="terminal",
        chat_id="user:42",
        content="Approve the bash call?",
        buttons=[
            Button(label="Approve", value="appr-1:approve", style="primary"),
            Button(label="Deny", value="appr-1:deny", style="danger"),
        ],
        metadata={"correlation_id": "appr-1", "kind": "approval_request"},
        kind=OutboundKind.MESSAGE,
    )
    await ch.send(msg)

    assert len(outbox.items) == 1
    peer_id, env = outbox.items[0]
    assert peer_id == "peer-1"
    body = env.body
    assert isinstance(body, dict)
    assert body["channel"] == "terminal"
    assert body["chat_id"] == "user:42"
    assert body["content"] == "Approve the bash call?"
    assert body["kind"] == "message"
    assert body["buttons"] == [
        {"label": "Approve", "value": "appr-1:approve", "style": "primary"},
        {"label": "Deny", "value": "appr-1:deny", "style": "danger"},
    ]
    assert body["metadata"] == {
        "correlation_id": "appr-1",
        "kind": "approval_request",
    }

    # Sanity: when buttons / metadata are empty, they must NOT appear
    # in the body (avoid noisy null fields on the wire).
    await ch.send(
        OutboundMessage(channel="terminal", chat_id="user:42", content="hi")
    )
    plain_body = outbox.items[1][1].body
    assert isinstance(plain_body, dict)
    assert "buttons" not in plain_body
    assert "metadata" not in plain_body

    await manager.stop()


@pytest.mark.asyncio
async def test_wirebridge_forwards_button_value_inbound() -> None:
    bus = _new_bus()
    manager = ChannelManager({}, bus)
    outbox = _RecordedOutbox()
    bridge = WireBridge(
        bus=bus, manager=manager, outbox=outbox  # type: ignore[arg-type]
    )

    # Minimal PeerSession — only ``peer_id`` is read on the inbound path.
    peer = PeerSession(
        peer_id="peer-2",
        peer_kind="chat_client",
        transport_writer=None,  # type: ignore[arg-type]
    )

    env = Envelope(
        v=WIRE_VERSION,
        id="in-1",
        kind=KIND_INBOUND,
        ts=time.time(),
        body={
            "channel": "terminal",
            "chat_id": "user:42",
            # sender_id is bound to peer_id by the spoof guard (issue
            # #1); use peer-2 so the envelope is accepted.
            "sender_id": "peer-2",
            "content": "[card click: appr-1:approve]",
            "button_value": "appr-1:approve",
        },
    )
    await bridge.handle_inbound(peer, env)

    # Pull the just-published inbound off the bus and confirm the
    # button_value rode through unchanged.
    received: InboundMessage = await asyncio.wait_for(
        bus.consume_inbound(), timeout=1.0
    )
    assert received.channel == "terminal"
    assert received.chat_id == "user:42"
    assert received.button_value == "appr-1:approve"

    # When the envelope has no button_value, the inbound's field stays
    # None — the approval bridge uses None vs str to gate routing.
    env_plain = Envelope(
        v=WIRE_VERSION,
        id="in-2",
        kind=KIND_INBOUND,
        ts=time.time(),
        body={
            "channel": "terminal",
            "chat_id": "user:42",
            "sender_id": "peer-2",
            "content": "hello",
        },
    )
    await bridge.handle_inbound(peer, env_plain)
    received2 = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
    assert received2.button_value is None

    await manager.stop()


__all__: list[str] = []
# Silence "unused" lint complaints — these imports are part of the
# tested API surface.
_ = (Any,)
