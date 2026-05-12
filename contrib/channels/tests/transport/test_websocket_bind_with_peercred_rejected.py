"""Refuse the (peer-cred auth, WS transport) pairing at construction.

Peer-cred reads AF_UNIX kernel credentials — those don't exist on a
WebSocket transport. Failing fast in ``WireServer.__init__`` is the
explicit guard rail (see ``.claude/plans/2026-05-12-gateway-websocket-transport.md``).
"""

from __future__ import annotations

import pytest

from agentm_channels.auth import UnixPeerCredAuthenticator
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.transport import WebSocketServerTransport
from agentm_channels.wire import Envelope


async def _noop(_s: PeerSession, _e: Envelope) -> None:
    return None


async def test_ws_with_peercred_auth_rejected(free_port: int, db_path: str) -> None:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    try:
        with pytest.raises(ValueError, match="UnixPeerCred"):
            WireServer(
                outbox=outbox,
                inbox=inbox,
                on_inbound=_noop,
                transport=WebSocketServerTransport("127.0.0.1", free_port),
                authenticator=UnixPeerCredAuthenticator(),
            )
    finally:
        outbox.close()
        inbox.close()
