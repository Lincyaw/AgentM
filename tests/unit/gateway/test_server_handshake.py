"""Fail-stop: WireServer handshake — auth + wire-version negotiation (§2.2, §6).

The hello/welcome handshake is the trust + compatibility boundary. A
rejected token must not get a welcome; a v1 envelope must be rejected
with ``unsupported_wire_version``; an accepted peer must round-trip one
inbound -> outbound.
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from agentm.gateway.auth import AllowAllAuthenticator, TokenAuthenticator
from agentm.gateway.client import AuthError, WireClient
from agentm.gateway.outbox import SqliteInbox, SqliteOutbox
from agentm.gateway.peer import PeerSession
from agentm.gateway.server import WireServer
from agentm.gateway.wire import WIRE_VERSION, Envelope, encode
from agentm.gateway.transport import UnixServerTransport, UnixClientTransport


async def _make_server(tmp_path: Path, sock: str, *, authenticator, on_inbound):
    outbox = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    inbox = SqliteInbox(str(tmp_path / "ib.sqlite"))
    server = WireServer(
        transport=UnixServerTransport(sock),
        outbox=outbox,
        inbox=inbox,
        on_inbound=on_inbound,
        authenticator=authenticator,
    )
    await server.start()
    return server, outbox, inbox


@pytest.mark.asyncio
async def test_allow_all_welcomes_and_round_trips(tmp_path: Path) -> None:
    sock = str(tmp_path / "gw.sock")
    received: list[Envelope] = []
    delivered: list[Envelope] = []

    async def on_inbound(peer: PeerSession, env: Envelope) -> None:
        received.append(env)
        # Echo an outbound back to the same peer via the outbox.
        out = Envelope(
            v=WIRE_VERSION,
            id="out1",
            kind="outbound",
            ts=1.0,
            session_key=env.session_key,
            body={"channel": "terminal", "chat_id": "t1", "content": "ack"},
        )
        await asyncio.to_thread(outbox.enqueue, peer.peer_id, out)

    server, outbox, inbox = await _make_server(
        tmp_path, sock, authenticator=AllowAllAuthenticator(), on_inbound=on_inbound
    )
    try:
        async def on_outbound(env: Envelope) -> None:
            delivered.append(env)

        client = WireClient(
            transport=UnixClientTransport(sock),
            peer_name="terminal-1",
            on_outbound=on_outbound,
        )
        await client.connect()
        assert client.welcome() is not None
        assert client.welcome().body["wire_version"] == WIRE_VERSION
        await client.send_inbound(
            {"channel": "terminal", "chat_id": "t1", "content": "hi"},
            session_key="terminal:t1",
        )
        # Wait for the echo to be delivered.
        for _ in range(100):
            if delivered:
                break
            await asyncio.sleep(0.02)
        assert received and received[0].session_key == "terminal:t1"
        assert delivered and delivered[0].body["content"] == "ack"
        await client.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


@pytest.mark.asyncio
async def test_token_mismatch_rejected(tmp_path: Path) -> None:
    sock = str(tmp_path / "gw.sock")

    async def on_inbound(peer: PeerSession, env: Envelope) -> None:
        return None

    server, outbox, inbox = await _make_server(
        tmp_path,
        sock,
        authenticator=TokenAuthenticator(allowed_tokens={"goodtoken"}),
        on_inbound=on_inbound,
    )
    try:
        client = WireClient(
            transport=UnixClientTransport(sock),
            peer_name="terminal-1",
            token="wrongtoken",
        )
        with pytest.raises(AuthError) as exc:
            await client.connect()
        assert exc.value.code == "auth_failed"
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


@pytest.mark.asyncio
async def test_token_match_accepted(tmp_path: Path) -> None:
    sock = str(tmp_path / "gw.sock")

    async def on_inbound(peer: PeerSession, env: Envelope) -> None:
        return None

    server, outbox, inbox = await _make_server(
        tmp_path,
        sock,
        authenticator=TokenAuthenticator(allowed_tokens={"goodtoken"}),
        on_inbound=on_inbound,
    )
    try:
        client = WireClient(
            transport=UnixClientTransport(sock),
            peer_name="terminal-1",
            token="goodtoken",
        )
        await client.connect()
        assert client.welcome() is not None
        await client.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


@pytest.mark.asyncio
async def test_v1_hello_rejected_with_unsupported_wire_version(tmp_path: Path) -> None:
    """A raw v1 hello frame (v=1) must be rejected at the handshake.

    We hand-craft a v1-shaped frame because the v2 Envelope refuses to
    construct with v=1 — the server must answer ``unsupported_wire_version``.
    """
    sock = str(tmp_path / "gw.sock")

    async def on_inbound(peer: PeerSession, env: Envelope) -> None:
        return None

    server, outbox, inbox = await _make_server(
        tmp_path, sock, authenticator=AllowAllAuthenticator(), on_inbound=on_inbound
    )
    try:
        reader, writer = await UnixClientTransport(sock).connect()
        import json

        v1_hello = {
            "v": 1,
            "id": "hello-old",
            "kind": "hello",
            "ts": 1.0,
            "body": {"peer_id": "legacy", "peer_kind": "chat_client"},
            "hops": 0,
        }
        payload = json.dumps(v1_hello).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()
        # Read the server's reply frame.
        header = await reader.readexactly(4)
        length = int.from_bytes(header, "big")
        body = await reader.readexactly(length)
        reply = json.loads(body.decode("utf-8"))
        assert reply["kind"] == "error"
        assert reply["body"]["code"] == "unsupported_wire_version"
        writer.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


def test_encode_v2_frame_helper_roundtrips() -> None:
    # Guards the framing helper the test above relies on.
    env = Envelope(v=WIRE_VERSION, id="x", kind="ping", ts=1.0, body={})
    assert encode(env).startswith(struct.pack(">I", len(encode(env)) - 4))
