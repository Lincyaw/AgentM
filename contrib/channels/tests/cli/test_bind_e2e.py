"""Phase 1.4 fail-stop: a wire peer's ``inbound`` reaches the v0
Gateway dispatch path, and the gateway's reply round-trips back to
the peer via the outbox.

Drives :class:`WireBridge` + :class:`WireServer` + :class:`Gateway`
with a fake :class:`AgentSession` (echo). No subprocess, no real LLM.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    EventBus,
    TextContent,
)
from agentm.core.abi.events import TurnEndEvent

from agentm_channels.auth import UnixPeerCredAuthenticator
from agentm_channels.client import WireClient
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.server import WireServer
from agentm_channels.wire import Envelope


class _FakeSM:
    def get_session_id(self) -> str:
        return f"fake-{int(time.time() * 1000)}"


class _FakeSession:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self.session_manager = _FakeSM()

    async def prompt(self, text: str) -> None:
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=f"echo: {text}")],
            timestamp=time.time(),
        )
        await self._bus.emit(
            TurnEndEvent.CHANNEL,
            TurnEndEvent(turn_index=0, message=msg, messages=()),
        )

    async def shutdown(self) -> None:
        pass


async def _factory(_cwd: str, bus: EventBus, _resume: str | None) -> Any:
    return _FakeSession(bus)


async def _wait_for(predicate: Any, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")




async def test_wire_peer_rejected_by_uid_policy() -> None:
    """A peer-cred allow-list that excludes the test process's uid
    refuses the connection. This is the fail-stop for ``--bind-allow-uid``.
    """
    pytest.importorskip("socket")
    with tempfile.TemporaryDirectory(prefix="agentm-bind-e2e-") as d:
        sock = os.path.join(d, "g.sock")
        outbox = SqliteOutbox(os.path.join(d, "ob.sqlite"))
        inbox = SqliteInbox(os.path.join(d, "ib.sqlite"))

        async def _noop(_p: Any, _e: Envelope) -> None:
            return None

        server = WireServer(
            socket_path=sock,
            outbox=outbox,
            inbox=inbox,
            on_inbound=_noop,
            authenticator=UnixPeerCredAuthenticator(
                allowed_uids={os.geteuid() + 99999}
            ),
        )
        await server.start()
        try:
            from agentm_channels.client import AuthError

            client = WireClient(
        socket_path=sock, peer_id="p1", peer_kind="chat_client")
            with pytest.raises(AuthError) as exc:
                await client.connect()
            assert exc.value.code == "auth_failed"
        finally:
            await server.stop()
            outbox.close()
            inbox.close()
