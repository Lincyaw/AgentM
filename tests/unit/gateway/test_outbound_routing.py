"""Fail-stop: outbound routes to the peer serving its channel (§3.2).

The bug this guards against: with a Feishu peer and a terminal peer
connected to one gateway, a reply for the Feishu chat must NOT be
enqueued to the terminal peer (and vice versa). The earlier code
broadcast every outbound to every peer, so a second connected client of
a different platform received traffic it did not own. ``_route_targets``
is the pure selection that fixes it.
"""

from __future__ import annotations

from typing import Any, cast, get_args

import pytest

from agentm.gateway.cli import _GatewayRuntime, _route_targets
from agentm.gateway.peer import PeerSession
from agentm.gateway.wire import (
    DURABLE_OUTBOUND_KINDS,
    EPHEMERAL_OUTBOUND_KINDS,
    KIND_OUTBOUND,
    OutboundMetaKind,
    decode_stream,
)


def test_delivery_class_partition_matches_the_kind_literal() -> None:
    """The durable/ephemeral sets are the single source the sink routes by;
    they must partition OutboundMetaKind exactly — disjoint and exhaustive —
    so a new kind cannot be added to the Literal without being classified, and
    a kind cannot drift between the wire vocabulary and the delivery policy."""
    literal = set(get_args(OutboundMetaKind))
    assert DURABLE_OUTBOUND_KINDS & EPHEMERAL_OUTBOUND_KINDS == set()
    assert DURABLE_OUTBOUND_KINDS | EPHEMERAL_OUTBOUND_KINDS == literal


def _peer(peer_id: str) -> PeerSession:
    # _route_targets only reads .peer_id; the writer is never touched here.
    return PeerSession(peer_id=peer_id, transport_writer=cast(Any, None))


def test_routes_only_to_peer_serving_the_channel() -> None:
    feishu = _peer("feishu-1")
    terminal = _peer("terminal-1")
    peers = [feishu, terminal]
    channels = {"feishu-1": "feishu", "terminal-1": "terminal"}

    assert _route_targets(peers, channels, "feishu") == [feishu]
    assert _route_targets(peers, channels, "terminal") == [terminal]


def test_excludes_peers_with_unknown_channel_when_a_match_exists() -> None:
    feishu = _peer("feishu-1")
    fresh = _peer("just-connected")  # no inbound yet -> channel unknown
    peers = [feishu, fresh]
    channels = {"feishu-1": "feishu"}

    # The fresh peer must not receive Feishu traffic just because it is
    # connected — only the known Feishu peer does.
    assert _route_targets(peers, channels, "feishu") == [feishu]


def test_falls_back_to_all_when_no_peer_serves_the_channel() -> None:
    a = _peer("a")
    b = _peer("b")
    peers = [a, b]

    # Nothing learned yet (single-client-before-first-inbound) -> deliver
    # to everyone so the degenerate case still works.
    assert _route_targets(peers, {}, "feishu") == peers
    # Empty target channel (proactive / degenerate) -> same fallback.
    assert _route_targets(peers, {"a": "feishu"}, "") == peers


def test_routes_to_all_same_channel_peers_mirror_case() -> None:
    t1 = _peer("terminal-1")
    t2 = _peer("terminal-2")
    peers = [t1, t2]
    channels = {"terminal-1": "terminal", "terminal-2": "terminal"}

    # Two terminals on the same channel both mirror the conversation.
    assert _route_targets(peers, channels, "terminal") == [t1, t2]


# --- delivery-class split (durable outbox vs ephemeral direct write) -------


class _RecordingOutbox:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, Any]] = []

    def enqueue(self, peer_id: str, env: Any) -> None:
        self.enqueued.append((peer_id, env))


class _RecordingWriter:
    """Captures bytes written straight to the peer (the ephemeral path)."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        return None


class _FakeServer:
    def __init__(self, peers: list[PeerSession]) -> None:
        self._peers = peers

    @property
    def registry(self) -> list[PeerSession]:
        return self._peers


def _runtime_with(peer: PeerSession, outbox: _RecordingOutbox) -> _GatewayRuntime:
    # Bypass the heavy __init__ (builds SqliteOutbox/SessionManager/...); we
    # only exercise _emit_outbound, which reads _server/_outbox/_peer_channels.
    rt = object.__new__(_GatewayRuntime)
    rt._server = cast(Any, _FakeServer([peer]))
    rt._outbox = cast(Any, outbox)
    rt._peer_channels = {peer.peer_id: "terminal"}
    return rt


def _body(kind: str) -> dict[str, Any]:
    return {
        "channel": "terminal",
        "chat_id": "t1",
        "content": "x",
        "metadata": {"kind": kind},
        "_session_key": "terminal:t1",
    }


@pytest.mark.asyncio
async def test_durable_kind_enqueues_and_does_not_write_directly() -> None:
    outbox = _RecordingOutbox()
    writer = _RecordingWriter()
    peer = PeerSession(peer_id="terminal-1", transport_writer=cast(Any, writer))
    rt = _runtime_with(peer, outbox)

    await rt._emit_outbound(_body("assistant_text"))

    assert len(outbox.enqueued) == 1  # durable -> outbox
    assert bytes(writer.buf) == b""  # never written straight to the peer


@pytest.mark.asyncio
async def test_ephemeral_kind_writes_directly_and_skips_the_outbox() -> None:
    outbox = _RecordingOutbox()
    writer = _RecordingWriter()
    peer = PeerSession(peer_id="terminal-1", transport_writer=cast(Any, writer))
    rt = _runtime_with(peer, outbox)

    await rt._emit_outbound(_body("stream_text"))

    # Streaming deltas MUST bypass the durable outbox (a turn streams ~50/s;
    # otherwise the SQLite queue explodes and replays stale frames).
    assert outbox.enqueued == []
    # ... and land as one valid `outbound` envelope on the live socket.
    envelopes, rest = decode_stream(bytes(writer.buf))
    assert rest == b""
    assert len(envelopes) == 1
    assert envelopes[0].kind == KIND_OUTBOUND
    assert envelopes[0].session_key == "terminal:t1"
    assert envelopes[0].body["metadata"]["kind"] == "stream_text"
