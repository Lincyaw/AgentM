"""Authenticator that rejects → client receives error and connect() raises."""

from __future__ import annotations

import asyncio

import pytest

from agentm_channels.client import AuthError, WireClient
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.wire import Envelope


async def _noop(_session: PeerSession, _env: Envelope) -> None:
    return None


class TokenAuth:
    def __init__(self, expected_token: str, expected_peer: str) -> None:
        self.expected_token = expected_token
        self.expected_peer = expected_peer

    async def authenticate(
        self,
        peer_kind: str,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,
    ) -> bool:
        if peer_id == self.expected_peer and token == self.expected_token:
            return True
        return False


async def test_wrong_token_rejected(socket_path: str, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    server = WireServer(
        socket_path, outbox, inbox, _noop, authenticator=TokenAuth("good", "trusted")
    )
    await server.start()
    try:
        evil = WireClient(
            socket_path, peer_id="evil", peer_kind="chat_client", token="wrong"
        )
        with pytest.raises(AuthError) as exc_info:
            await evil.connect()
        assert exc_info.value.code == "auth_failed"
        # Confirm the *good* peer still works.
        good = WireClient(
            socket_path, peer_id="trusted", peer_kind="chat_client", token="good"
        )
        await good.connect()
        assert good.welcome() is not None
        await good.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
