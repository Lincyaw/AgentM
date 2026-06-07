"""Fail-stop: WireClient auto-reconnect survives a gateway restart (§2.6).

The gateway can be restarted/updated (e.g. by systemd) under a live peer.
The opt-in reconnect supervisor (:meth:`WireClient.run_reconnecting`) must
re-dial transparently with the *same* ``peer_name`` so the server's outbox
replays every durable frame the gap produced. If this breaks, a peer that
rides through a gateway restart silently loses the answers enqueued while
it was disconnected — the whole point of Phase-1 reconnect.

This drives the real client + real ``WireServer`` over a Unix socket: it
stands a server, connects a reconnecting client, drops the gateway, enqueues
a durable ``outbound`` for the peer into the shared outbox during the gap,
then starts a fresh server on the same socket and asserts the client
reconnects under the same ``peer_name`` AND the gap frame is delivered via
reconnect replay. Assertions are on observable behaviour (delivered frames,
hello peer_name seen server-side), not private state.

NOTE ON THE DROP: a real gateway restart (systemd SIGTERM/SIGKILL) tears the
process down, so the OS closes the listening socket *and* every live peer fd
at once and the client sees EOF. ``WireServer.stop()`` alone does not model
that — its ``transport.close()`` awaits ``asyncio.Server.wait_closed()``,
which blocks until still-open client connections close, so stopping under a
live peer would hang. We therefore abort the live peer connection(s) first
(EOF to the client), then stop the listener — an abrupt drop, which is what
the reconnect path must survive.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from agentm.gateway.auth import AllowAllAuthenticator
from agentm.gateway.client import WireClient
from agentm.gateway.outbox import SqliteInbox, SqliteOutbox
from agentm.gateway.peer import PeerSession
from agentm.gateway.server import WireServer
from agentm.gateway.transport import UnixClientTransport, UnixServerTransport
from agentm.gateway.wire import WIRE_VERSION, Envelope


async def _make_server(
    sock: str, outbox: SqliteOutbox, inbox: SqliteInbox, seen_peers: list[str]
) -> WireServer:
    async def on_inbound(peer: PeerSession, env: Envelope) -> None:
        # Record which peer_name the server saw connect (handshake identity).
        if peer.peer_id not in seen_peers:
            seen_peers.append(peer.peer_id)

    server = WireServer(
        transport=UnixServerTransport(sock),
        outbox=outbox,
        inbox=inbox,
        on_inbound=on_inbound,
        authenticator=AllowAllAuthenticator(),
    )
    await server.start()
    return server


async def _drop_gateway(server: WireServer) -> None:
    """Abruptly drop the gateway under a live peer (systemd-restart model).

    Abort every live peer connection first so the client sees EOF, then stop
    the listener — otherwise ``stop()`` blocks on ``wait_closed()`` waiting
    for the still-open peer connection to close gracefully.
    """
    for peer in list(server.registry):
        with contextlib.suppress(Exception):
            peer.transport_writer.close()
    await asyncio.wait_for(server.stop(), timeout=5)


@pytest.mark.asyncio
async def test_reconnecting_client_survives_gateway_restart(tmp_path: Path) -> None:
    sock = str(tmp_path / "gw.sock")
    # One durable outbox shared across both server incarnations (it is the
    # peer's persistent mailbox — survives the gateway process restart).
    outbox = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    inbox = SqliteInbox(str(tmp_path / "ib.sqlite"))
    delivered: list[Envelope] = []
    seen_peers: list[str] = []
    peer_name = "feishu-stable1"

    async def on_outbound(env: Envelope) -> None:
        delivered.append(env)

    server = await _make_server(sock, outbox, inbox, seen_peers)
    client = WireClient(
        transport=UnixClientTransport(sock),
        peer_name=peer_name,
        on_outbound=on_outbound,
        backoff_base=0.05,
        backoff_cap=0.2,
    )
    supervisor = asyncio.create_task(client.run_reconnecting())
    try:
        # Establish the first connection (and force a handshake the server
        # observes by sending one inbound).
        for _ in range(100):
            if client.welcome() is not None:
                break
            await asyncio.sleep(0.02)
        assert client.welcome() is not None, "initial connect never completed"
        await client.send_inbound(
            {"channel": "terminal", "chat_id": "t1", "content": "hi"},
            session_key="terminal:t1",
        )
        for _ in range(100):
            if seen_peers:
                break
            await asyncio.sleep(0.02)
        assert seen_peers == [peer_name]

        # --- Gateway restart: abruptly drop the connection. ---
        await _drop_gateway(server)

        # While the client is in its reconnect gap, a durable answer is
        # enqueued for this peer (what an in-flight session reply would do).
        gap_frame = Envelope(
            v=WIRE_VERSION,
            id="gap-answer",
            kind="outbound",
            ts=1.0,
            session_key="terminal:t1",
            body={"channel": "terminal", "chat_id": "t1", "content": "delayed-answer"},
        )
        await asyncio.to_thread(outbox.enqueue, peer_name, gap_frame)

        # Bring the gateway back on the same socket.
        server = await _make_server(sock, outbox, inbox, seen_peers)

        # (a) The client transparently reconnects with the same peer_name AND
        # (b) the gap frame is delivered via outbox replay.
        for _ in range(200):
            if any(e.id == "gap-answer" for e in delivered):
                break
            await asyncio.sleep(0.02)
        assert any(e.body.get("content") == "delayed-answer" for e in delivered), (
            "reconnecting client never received the durable frame enqueued "
            f"during the gateway-restart gap; delivered={[e.id for e in delivered]!r}"
        )
        # Same-peer-name reuse is proven *transitively* by assertion (b): the
        # outbox is keyed by peer_name, so the gap frame replays only if the
        # reconnect hello reused ``peer_name``. This extra check just confirms
        # no *other* peer identity leaked into the inbound path along the way.
        assert seen_peers == [peer_name], (
            f"unexpected extra peer identity on the inbound path: {seen_peers!r}"
        )
        # The replayed durable row drained (acked on successful delivery).
        assert await asyncio.to_thread(outbox.pending_count, peer_name) == 0
    finally:
        await client.close()
        supervisor.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await supervisor
        await _drop_gateway(server)
        outbox.close()
        inbox.close()
