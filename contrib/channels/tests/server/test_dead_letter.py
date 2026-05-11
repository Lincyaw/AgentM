"""Dead-letter path: write keeps failing → after MAX_ATTEMPTS, row → DLQ.

The server exits the delivery loop after one socket failure (the
connection is dead), but on each peer *reconnect* the row is leased
afresh and attempts increments. So getting a row dead-lettered needs
``max_delivery_attempts`` reconnects. Tests run with
``max_delivery_attempts=1`` so a single attempt suffices.
"""

from __future__ import annotations

import asyncio
import time

from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.wire import WIRE_VERSION, Envelope

from .conftest import wait_for


async def _noop(_session: PeerSession, _env: Envelope) -> None:
    return None


class _BadWriter:
    """Wraps a real :class:`asyncio.StreamWriter` and raises on ``write``."""

    def __init__(self, real: asyncio.StreamWriter) -> None:
        self._real = real
        self.is_closing = real.is_closing  # type: ignore[assignment]

    def write(self, data: bytes) -> None:
        raise BrokenPipeError("simulated write failure")

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._real.close()

    async def wait_closed(self) -> None:
        await self._real.wait_closed()


async def test_dead_letter_after_max_attempts(socket_path: str, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    outbox.enqueue(
        "D",
        Envelope(
            v=WIRE_VERSION,
            id="dead-1",
            kind="outbound",
            ts=time.time(),
            body={"k": "v"},
        ),
    )

    # Use a very small backoff so retries happen fast. We don't expose
    # the policy fn as config — instead, monkey-patch the policy module
    # so attempts re-fire promptly.
    import agentm_channels.server as server_mod

    real_backoff = server_mod.exponential_backoff
    server_mod.exponential_backoff = lambda attempts: 0.0  # type: ignore[assignment]

    # max_delivery_attempts=1 → first write failure dead-letters,
    # exercising the same code path that fires at MAX_DELIVERY_ATTEMPTS
    # in production. Simulating 5 reconnect cycles for one test would
    # bloat the suite without adding behavioural coverage.
    server = WireServer(
        socket_path,
        outbox,
        inbox,
        _noop,
        delivery_batch_max=1,
        max_delivery_attempts=1,
    )
    await server.start()

    # Replace the writer on register to one that always fails.
    real_register = server._registry.register

    def patched_register(session: PeerSession) -> None:  # type: ignore[no-redef]
        session.transport_writer = _BadWriter(session.transport_writer)  # type: ignore[assignment]
        real_register(session)

    server._registry.register = patched_register  # type: ignore[assignment]

    try:
        from agentm_channels.client import WireClient

        client = WireClient(socket_path, peer_id="D", peer_kind="chat_client")
        await client.connect()
        ok = await wait_for(
            lambda: outbox.dead_letter_count("D") >= 1, timeout=5.0
        )
        assert ok
        assert outbox.pending_count("D") == 0
        await client.close()
    finally:
        server_mod.exponential_backoff = real_backoff  # type: ignore[assignment]
        await server.stop()
        outbox.close()
        inbox.close()
