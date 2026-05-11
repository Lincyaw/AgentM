"""Inbound → server handler enqueues outbound → client receives it."""

from __future__ import annotations

import time

from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.wire import WIRE_VERSION, Envelope

from .conftest import wait_for


async def test_echo_inbound_to_outbound(socket_path: str, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    received: list[Envelope] = []

    async def on_inbound(session: PeerSession, env: Envelope) -> None:
        echo = Envelope(
            v=WIRE_VERSION,
            id=f"echo-of-{env.id}",
            kind="outbound",
            ts=time.time(),
            body={"echo_of": env.body},
        )
        outbox.enqueue(session.peer_id, echo)

    server = WireServer(socket_path, outbox, inbox, on_inbound)
    await server.start()

    async def collect(env: Envelope) -> None:
        received.append(env)

    try:
        from agentm_channels.client import WireClient

        client = WireClient(
            socket_path, peer_id="A", peer_kind="chat_client", on_outbound=collect
        )
        await client.connect()
        await client.send_inbound({"text": "hi"}, env_id="m1")
        ok = await wait_for(lambda: len(received) == 1)
        assert ok
        assert received[0].kind == "outbound"
        assert received[0].body["echo_of"] == {"text": "hi"}
        await client.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
