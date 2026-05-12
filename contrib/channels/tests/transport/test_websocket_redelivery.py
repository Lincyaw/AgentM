"""Outbox redelivery over WS reconnect.

Mirrors ``tests/server/test_durability_restart.py`` for the WebSocket
transport: drop a peer mid-flight, restart the server with the same
sqlite outbox, reconnect, and verify the in-flight envelopes get
redelivered (exactly once across both sessions when combined with the
ones already drained).
"""

from __future__ import annotations

import asyncio
import os
import time

from agentm_channels.client import WireClient
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.transport import (
    WebSocketClientTransport,
    WebSocketServerTransport,
)
from agentm_channels.wire import WIRE_VERSION, Envelope


async def _noop(_s: PeerSession, _e: Envelope) -> None:
    return None


def _outbound(env_id: str) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id,
        kind="outbound",
        ts=time.time(),
        body={"seq": env_id},
    )


async def _wait(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


async def test_ws_redelivery_across_reconnect(tmp_dir: str, free_port: int) -> None:
    db_path = os.path.join(tmp_dir, "outbox.sqlite")

    seen_session1: list[str] = []
    seen_session2: list[str] = []

    # --- session 1: connect peer, drop before anything is enqueued --
    outbox = SqliteOutbox(db_path, lease_ttl=0.3)
    inbox = SqliteInbox(db_path)
    server = WireServer(
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop,
        transport=WebSocketServerTransport("127.0.0.1", free_port),
        delivery_batch_max=1,
    )
    await server.start()

    async def on_outbound_1(env: Envelope) -> None:
        seen_session1.append(env.id)

    try:
        client = WireClient(
            transport=WebSocketClientTransport(f"ws://127.0.0.1:{free_port}/"),
            peer_id="A",
            peer_kind="chat_client",
            on_outbound=on_outbound_1,
        )
        await client.connect()
        await _wait(lambda: "A" in server.registry, timeout=2.0)
        # Hard-close the client so the connection drops; envelopes
        # enqueued AFTER this point are not delivered to session 1.
        if client._read_task is not None:  # type: ignore[attr-defined]
            client._read_task.cancel()  # type: ignore[attr-defined]
        # Wait for the server to notice the disconnect.
        await _wait(lambda: "A" not in server.registry, timeout=2.0)
        # Now enqueue 3 envelopes — they sit pending in the outbox.
        for i in range(3):
            outbox.enqueue("A", _outbound(f"m{i}"))
    finally:
        await server.stop()
        outbox.close()
        inbox.close()

    # Let leases expire so any pending leased rows are eligible again.
    await asyncio.sleep(0.5)

    # --- session 2: fresh server, same sqlite, same port ------------
    outbox = SqliteOutbox(db_path, lease_ttl=0.3)
    inbox = SqliteInbox(db_path)
    server = WireServer(
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop,
        transport=WebSocketServerTransport("127.0.0.1", free_port),
        delivery_batch_max=8,
    )
    await server.start()

    async def on_outbound_2(env: Envelope) -> None:
        seen_session2.append(env.id)

    try:
        client2 = WireClient(
            transport=WebSocketClientTransport(f"ws://127.0.0.1:{free_port}/"),
            peer_id="A",
            peer_kind="chat_client",
            on_outbound=on_outbound_2,
        )
        await client2.connect()
        ok = await _wait(lambda: len(seen_session2) >= 3, timeout=5.0)
        assert ok
        # Each pending envelope arrives exactly once on session 2;
        # session 1 saw none of them (it was already dropped).
        assert sorted(seen_session2) == ["m0", "m1", "m2"]
        assert seen_session1 == []
        await client2.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
