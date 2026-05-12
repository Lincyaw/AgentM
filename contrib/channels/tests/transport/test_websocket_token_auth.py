"""Token auth gating over WebSocket.

- bad token → server rejects the hello; peer never enters the registry.
- good token → handshake completes and the peer registers.
"""

from __future__ import annotations

import asyncio

import pytest

from agentm_channels.auth import TokenAuthenticator
from agentm_channels.client import AuthError, WireClient
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.transport import (
    WebSocketClientTransport,
    WebSocketServerTransport,
)
from agentm_channels.wire import Envelope


async def _noop(_s: PeerSession, _e: Envelope) -> None:
    return None


async def test_ws_token_auth_rejects_bad_token(free_port: int, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    server = WireServer(
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop,
        transport=WebSocketServerTransport("127.0.0.1", free_port),
        authenticator=TokenAuthenticator({"good"}),
    )
    await server.start()
    try:
        client = WireClient(
            transport=WebSocketClientTransport(f"ws://127.0.0.1:{free_port}/"),
            peer_id="bad-peer",
            peer_kind="chat_client",
            token="bad",
        )
        with pytest.raises(AuthError):
            await client.connect()

        # Give the server a moment to finalise; the peer must NOT be in
        # the registry.
        await asyncio.sleep(0.05)
        assert "bad-peer" not in server.registry
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_ws_token_auth_accepts_good_token(free_port: int, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    server = WireServer(
        outbox=outbox,
        inbox=inbox,
        on_inbound=_noop,
        transport=WebSocketServerTransport("127.0.0.1", free_port),
        authenticator=TokenAuthenticator({"good"}),
    )
    await server.start()
    try:
        client = WireClient(
            transport=WebSocketClientTransport(f"ws://127.0.0.1:{free_port}/"),
            peer_id="good-peer",
            peer_kind="chat_client",
            token="good",
        )
        await client.connect()
        # Wait briefly for the registry to register.
        for _ in range(50):
            if "good-peer" in server.registry:
                break
            await asyncio.sleep(0.02)
        assert "good-peer" in server.registry
        await client.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
