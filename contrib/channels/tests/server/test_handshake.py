"""Hello/welcome handshake + authenticator-reject path."""

from __future__ import annotations

import asyncio

import pytest

from agentm_channels.client import AuthError, WireClient
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.server import WireServer


async def _noop_inbound(_session: object, _env: object) -> None:
    return None


async def test_hello_welcome_roundtrip(socket_path: str, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    server = WireServer(socket_path, outbox, inbox, _noop_inbound)
    await server.start()
    try:
        client = WireClient(socket_path, peer_id="P1", peer_kind="chat_client")
        await client.connect()
        welcome = client.welcome()
        assert welcome is not None
        assert welcome.kind == "welcome"
        assert welcome.body["peer_id_echo"] == "P1"
        assert "server_version" in welcome.body
        await client.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


class RejectingAuth:
    async def authenticate(
        self,
        peer_kind: str,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,
    ) -> bool:
        return False


async def test_authenticator_reject_closes_connection(
    socket_path: str, db_path: str
) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    server = WireServer(socket_path, outbox, inbox, _noop_inbound, authenticator=RejectingAuth())
    await server.start()
    try:
        client = WireClient(socket_path, peer_id="banned", peer_kind="chat_client")
        with pytest.raises(AuthError) as exc_info:
            await client.connect()
        assert exc_info.value.code == "auth_failed"
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
