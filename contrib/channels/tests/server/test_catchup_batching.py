"""Catch-up batching: M=50 queued → batched delivery, not 50 individual frames."""

from __future__ import annotations

import asyncio
import time
from math import ceil

from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.wire import (
    KIND_DELIVERY_BATCH,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
    decode_stream,
)



async def _noop(_session: PeerSession, _env: Envelope) -> None:
    return None


async def test_50_messages_arrive_as_batches(socket_path: str, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    for i in range(50):
        outbox.enqueue(
            "B",
            Envelope(
                v=WIRE_VERSION,
                id=f"b{i}",
                kind="outbound",
                ts=time.time(),
                body={"i": i},
            ),
        )

    BATCH_MAX = 32
    server = WireServer(
        socket_path, outbox, inbox, _noop, delivery_batch_max=BATCH_MAX
    )
    await server.start()

    # We intercept at the *wire* level by speaking the socket
    # directly — the client lib hides batch frames behind per-item
    # dispatch. Count raw frame kinds.
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        from agentm_channels.wire import KIND_HELLO, encode

        hello = Envelope(
            v=WIRE_VERSION,
            id="h",
            kind=KIND_HELLO,
            ts=time.time(),
            body={"peer_id": "B", "peer_kind": "chat_client"},
        )
        writer.write(encode(hello))
        await writer.drain()

        envs: list[Envelope] = []
        buf = b""
        deadline = asyncio.get_running_loop().time() + 5.0
        item_count = 0
        while asyncio.get_running_loop().time() < deadline and item_count < 50:
            chunk = await asyncio.wait_for(reader.read(65536), timeout=2.0)
            if not chunk:
                break
            buf += chunk
            ready, buf = decode_stream(buf)
            for env in ready:
                envs.append(env)
                if env.kind == KIND_DELIVERY_BATCH:
                    item_count += len(env.body.get("items", []))
                elif env.kind == KIND_OUTBOUND:
                    item_count += 1
        writer.close()

        # Drop the welcome.
        delivery_envs = [e for e in envs if e.kind in (KIND_OUTBOUND, KIND_DELIVERY_BATCH)]
        delivered_total = 0
        for e in delivery_envs:
            if e.kind == KIND_DELIVERY_BATCH:
                delivered_total += len(e.body["items"])
            else:
                delivered_total += 1
        assert delivered_total == 50
        # Server must use batching: at most ceil(50 / batch_max) batch envelopes
        assert len(delivery_envs) <= ceil(50 / BATCH_MAX) + 1  # tolerate one trailing single
        # And we should NOT have 50 single-frame outbound envelopes.
        single_count = sum(1 for e in delivery_envs if e.kind == KIND_OUTBOUND)
        assert single_count < 50, (
            f"expected batched delivery, got {single_count} single outbound frames"
        )
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
