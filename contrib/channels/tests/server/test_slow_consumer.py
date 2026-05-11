"""Slow-consumer detection: high-water tripped → diagnostic logged, no loss."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from agentm_channels.client import WireClient
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.wire import WIRE_VERSION, Envelope

from .conftest import wait_for


async def _noop(_session: PeerSession, _env: Envelope) -> None:
    return None


async def test_slow_consumer_logs_and_does_not_lose(
    socket_path: str, db_path: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="agentm_channels.server")
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    for i in range(30):
        outbox.enqueue(
            "C",
            Envelope(
                v=WIRE_VERSION,
                id=f"c{i}",
                kind="outbound",
                ts=time.time(),
                body={"i": i},
            ),
        )

    server = WireServer(
        socket_path,
        outbox,
        inbox,
        _noop,
        delivery_batch_max=1,  # force a slow drip so high-water trips before drain
        slow_consumer_high_water=10,
    )
    await server.start()

    received: list[str] = []

    async def slow_handler(env: Envelope) -> None:
        received.append(env.id)
        await asyncio.sleep(0.05)

    try:
        client = WireClient(
            socket_path, peer_id="C", peer_kind="chat_client", on_outbound=slow_handler
        )
        await client.connect()
        await wait_for(lambda: len(received) >= 30, timeout=10.0)
        assert len(received) == 30
        assert {r for r in received} == {f"c{i}" for i in range(30)}
        # Diagnostic was logged.
        slow_logs = [r for r in caplog.records if "slow_consumer" in r.getMessage()]
        assert slow_logs, "expected slow_consumer warning log"
        await client.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
