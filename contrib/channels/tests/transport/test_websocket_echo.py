"""WebSocket round-trip: one framed envelope decodes byte-identical
versus a Unix-socket round-trip.

Drives the wire framing directly through the adapter (no full server)
so we isolate the transport seam.
"""

from __future__ import annotations

import asyncio
import time

from agentm_channels.transport import (
    WebSocketClientTransport,
    WebSocketServerTransport,
)
from agentm_channels.wire import WIRE_VERSION, Envelope, encode

# Reuse the same helpers the wire server uses.
from agentm_channels.server import _read_one_frame, _send


def _make_env() -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id="m1",
        kind="inbound",
        ts=time.time(),
        body={"hello": "world", "n": 42},
    )


async def test_ws_envelope_roundtrip(free_port: int) -> None:
    received: list[Envelope] = []
    server_done = asyncio.Event()

    async def handle(reader, writer) -> None:  # type: ignore[no-untyped-def]
        env = await _read_one_frame(reader)
        if env is not None:
            received.append(env)
        # Echo it back.
        if env is not None:
            await _send(writer, env)
        server_done.set()

    server = WebSocketServerTransport("127.0.0.1", free_port)
    await server.serve(handle)
    try:
        client = WebSocketClientTransport(f"ws://127.0.0.1:{free_port}/")
        reader, writer = await client.connect()
        sent = _make_env()
        await _send(writer, sent)
        echoed = await asyncio.wait_for(_read_one_frame(reader), timeout=3.0)
        await asyncio.wait_for(server_done.wait(), timeout=3.0)

        assert echoed is not None
        # Byte-identical decode invariant: re-encoding both must match.
        assert encode(echoed) == encode(sent)
        assert received and encode(received[0]) == encode(sent)
    finally:
        await server.close()
