"""Catch-up batching: M=50 queued → batched delivery, not 50 individual frames."""

from __future__ import annotations

import asyncio
import time
from math import ceil

from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import BATCH_REASON_RECONNECT_CATCHUP, WireServer
from agentm_channels.wire import (
    KIND_DELIVERY_BATCH,
    KIND_HELLO,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
    decode_stream,
    encode,
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
        socket_path=socket_path,
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop, delivery_batch_max=BATCH_MAX
    )
    await server.start()

    # We intercept at the *wire* level by speaking the socket
    # directly — the client lib hides batch frames behind per-item
    # dispatch. Count raw frame kinds.
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)

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


async def _read_envelopes_until(
    reader: asyncio.StreamReader,
    predicate,
    *,
    timeout: float,
) -> list[Envelope]:
    envs: list[Envelope] = []
    buf = b""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        chunk = await asyncio.wait_for(reader.read(65536), timeout=2.0)
        if not chunk:
            break
        buf += chunk
        ready, buf = decode_stream(buf)
        for env in ready:
            envs.append(env)
        if predicate(envs):
            return envs
    return envs


async def test_batch_envelope_carries_reason_and_session_key(
    socket_path: str, db_path: str
) -> None:
    """Pre-connect catch-up batch includes ``reason`` and (when all
    records share one) ``session_key`` per §4.5.3."""
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    for i in range(3):
        outbox.enqueue(
            "P",
            Envelope(
                v=WIRE_VERSION,
                id=f"q{i}",
                kind="outbound",
                ts=time.time(),
                body={"i": i},
                root_session_key="feishu:chat-123",
            ),
        )

    server = WireServer(
        socket_path=socket_path,
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop, delivery_batch_max=32)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(
            encode(
                Envelope(
                    v=WIRE_VERSION,
                    id="h",
                    kind=KIND_HELLO,
                    ts=time.time(),
                    body={"peer_id": "P", "peer_kind": "chat_client"},
                )
            )
        )
        await writer.drain()
        envs = await _read_envelopes_until(
            reader,
            lambda es: any(e.kind == KIND_DELIVERY_BATCH for e in es),
            timeout=3.0,
        )
        batches = [e for e in envs if e.kind == KIND_DELIVERY_BATCH]
        assert batches, f"expected a delivery_batch, got {[e.kind for e in envs]}"
        batch = batches[0]
        assert batch.body.get("reason") == BATCH_REASON_RECONNECT_CATCHUP
        assert batch.body.get("session_key") == "feishu:chat-123"
        assert len(batch.body.get("items") or []) == 3
        writer.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_batch_session_key_none_when_records_diverge(
    socket_path: str, db_path: str
) -> None:
    """If records in the same drain have different root_session_keys,
    ``session_key`` is omitted (set to ``None``)."""
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    outbox.enqueue(
        "P",
        Envelope(
            v=WIRE_VERSION, id="a", kind="outbound", ts=time.time(),
            body={}, root_session_key="feishu:chat-1",
        ),
    )
    outbox.enqueue(
        "P",
        Envelope(
            v=WIRE_VERSION, id="b", kind="outbound", ts=time.time(),
            body={}, root_session_key="feishu:chat-2",
        ),
    )

    server = WireServer(
        socket_path=socket_path,
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop, delivery_batch_max=32)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(
            encode(
                Envelope(
                    v=WIRE_VERSION, id="h", kind=KIND_HELLO, ts=time.time(),
                    body={"peer_id": "P", "peer_kind": "chat_client"},
                )
            )
        )
        await writer.drain()
        envs = await _read_envelopes_until(
            reader,
            lambda es: any(e.kind == KIND_DELIVERY_BATCH for e in es),
            timeout=3.0,
        )
        batches = [e for e in envs if e.kind == KIND_DELIVERY_BATCH]
        assert batches
        assert batches[0].body.get("session_key") is None
        writer.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_steady_state_single_outbound_not_batched(
    socket_path: str, db_path: str
) -> None:
    """After a peer connects and drains, subsequent single enqueues
    arrive as ``KIND_OUTBOUND`` not ``KIND_DELIVERY_BATCH``."""
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)

    server = WireServer(
        socket_path=socket_path,
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop, delivery_batch_max=32)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(
            encode(
                Envelope(
                    v=WIRE_VERSION, id="h", kind=KIND_HELLO, ts=time.time(),
                    body={"peer_id": "P", "peer_kind": "chat_client"},
                )
            )
        )
        await writer.drain()

        # Wait for welcome to land so we know the peer is registered
        # (and the first drain pass has completed against an empty
        # outbox, advancing the worker out of "first drain" state).
        envs = await _read_envelopes_until(
            reader,
            lambda es: any(e.kind == "welcome" for e in es),
            timeout=3.0,
        )
        # Tiny grace for the worker to enter its wait state.
        await asyncio.sleep(0.05)

        # Now enqueue a single outbound and read what arrives.
        outbox.enqueue(
            "P",
            Envelope(
                v=WIRE_VERSION,
                id="solo",
                kind="outbound",
                ts=time.time(),
                body={"content": "hi"},
            ),
        )
        envs = await _read_envelopes_until(
            reader,
            lambda es: any(e.kind == KIND_OUTBOUND for e in es),
            timeout=3.0,
        )
        delivery = [e for e in envs if e.kind in (KIND_OUTBOUND, KIND_DELIVERY_BATCH)]
        assert delivery, "expected one outbound delivery"
        assert delivery[-1].kind == KIND_OUTBOUND, (
            f"steady-state single enqueue should arrive as outbound, "
            f"got {delivery[-1].kind}"
        )
        writer.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
