"""Fail-stop: outbound routes to the peer serving its channel (§3.2).

The bug this guards against: with a Feishu peer and a terminal peer
connected to one gateway, a reply for the Feishu chat must NOT be
enqueued to the terminal peer (and vice versa). The earlier code
broadcast every outbound to every peer, so a second connected client of
a different platform received traffic it did not own. ``route_targets``
is the pure selection that fixes it.
"""

from __future__ import annotations

from typing import Any, cast, get_args

import pytest

from agentm.gateway.runtime import GatewayRuntime, route_targets
from agentm.gateway.peer import PeerSession
from agentm.gateway.wire import (
    DURABLE_OUTBOUND_KINDS,
    EPHEMERAL_OUTBOUND_KINDS,
    OutboundMetaKind,
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
    # route_targets only reads .peer_id; the writer is never touched here.
    return PeerSession(peer_id=peer_id, transport_writer=cast(Any, None))


def test_routes_only_to_peer_serving_the_channel() -> None:
    feishu = _peer("feishu-1")
    terminal = _peer("terminal-1")
    peers = [feishu, terminal]
    channels = {"feishu-1": "feishu", "terminal-1": "terminal"}

    assert route_targets(peers, channels, "feishu") == [feishu]
    assert route_targets(peers, channels, "terminal") == [terminal]


def test_excludes_peers_with_unknown_channel_when_a_match_exists() -> None:
    feishu = _peer("feishu-1")
    fresh = _peer("just-connected")  # no inbound yet -> channel unknown
    peers = [feishu, fresh]
    channels = {"feishu-1": "feishu"}

    # The fresh peer must not receive Feishu traffic just because it is
    # connected — only the known Feishu peer does.
    assert route_targets(peers, channels, "feishu") == [feishu]


def test_falls_back_to_all_when_no_peer_serves_the_channel() -> None:
    a = _peer("a")
    b = _peer("b")
    peers = [a, b]

    # Nothing learned yet (single-client-before-first-inbound) -> deliver
    # to everyone so the degenerate case still works.
    assert route_targets(peers, {}, "feishu") == peers
    # Empty target channel (proactive / degenerate) -> same fallback.
    assert route_targets(peers, {"a": "feishu"}, "") == peers


def test_routes_to_all_same_channel_peers_mirror_case() -> None:
    t1 = _peer("terminal-1")
    t2 = _peer("terminal-2")
    peers = [t1, t2]
    channels = {"terminal-1": "terminal", "terminal-2": "terminal"}

    # Two terminals on the same channel both mirror the conversation.
    assert route_targets(peers, channels, "terminal") == [t1, t2]


# --- delivery-class split (durable persisted + queued, ephemeral queued) ----
# Unified ordered delivery (§2.6): _emit_outbound enqueues BOTH classes onto
# the peer's single send queue (the sender is the only socket writer). The
# classes differ only in persistence — durable rides the outbox (replay on
# reconnect) carrying its row id; ephemeral carries no row id.


class _RecordingOutbox:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, Any]] = []
        self._next_id = 0

    def enqueue(self, peer_id: str, env: Any) -> int:
        self._next_id += 1
        self.enqueued.append((peer_id, env))
        return self._next_id


class _RecordingWriter:
    """Captures any bytes written straight to the peer.

    _emit_outbound must NOT write here — the per-peer sender is the only
    socket writer now — so this stays empty in these unit tests.
    """

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


def _runtime_with(peer: PeerSession, outbox: _RecordingOutbox) -> GatewayRuntime:
    # Bypass the heavy __init__ (builds SqliteOutbox/SessionManager/...); we
    # only exercise _emit_outbound, which reads _server/_outbox/_peer_channels.
    rt = object.__new__(GatewayRuntime)
    rt._server = cast(Any, _FakeServer([peer]))
    rt._outbox = cast(Any, outbox)
    rt._session_routes: dict[str, tuple[str, str, str | None]] = {}
    rt._snapshots: dict[str, Any] = {}
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
async def test_durable_kind_persists_to_outbox_and_queues_with_row_id() -> None:
    outbox = _RecordingOutbox()
    writer = _RecordingWriter()
    peer = PeerSession(peer_id="terminal-1", transport_writer=cast(Any, writer))
    rt = _runtime_with(peer, outbox)

    await rt._emit_outbound(_body("assistant_text"))

    # durable -> persisted to outbox (so it replays on reconnect) ...
    assert len(outbox.enqueued) == 1
    # ... and queued for ordered delivery carrying its outbox row id.
    assert len(peer.send_q) == 1
    item = await peer.send_q.get()
    assert item.outbox_id == 1
    assert item.envelope.body["metadata"]["kind"] == "assistant_text"
    # _emit_outbound never writes the socket directly — the sender does.
    assert bytes(writer.buf) == b""


@pytest.mark.asyncio
async def test_ephemeral_kind_queues_without_persisting() -> None:
    outbox = _RecordingOutbox()
    writer = _RecordingWriter()
    peer = PeerSession(peer_id="terminal-1", transport_writer=cast(Any, writer))
    rt = _runtime_with(peer, outbox)

    await rt._emit_outbound(_body("stream_text"))

    # Streaming deltas MUST bypass the durable outbox (a turn streams ~50/s;
    # otherwise the SQLite queue explodes and replays stale frames) ...
    assert outbox.enqueued == []
    # ... and queue with no row id (best-effort, droppable under backpressure).
    assert len(peer.send_q) == 1
    item = await peer.send_q.get()
    assert item.outbox_id is None
    assert item.envelope.body["metadata"]["kind"] == "stream_text"
    assert bytes(writer.buf) == b""


@pytest.mark.asyncio
async def test_request_ack_is_durable() -> None:
    outbox = _RecordingOutbox()
    writer = _RecordingWriter()
    peer = PeerSession(peer_id="terminal-1", transport_writer=cast(Any, writer))
    rt = _runtime_with(peer, outbox)

    await rt._emit_outbound(_body("request_ack"))

    assert len(outbox.enqueued) == 1
    item = await peer.send_q.get()
    assert item.outbox_id == 1
    assert item.envelope.body["metadata"]["kind"] == "request_ack"


@pytest.mark.asyncio
async def test_mixed_kinds_queue_in_emit_order() -> None:
    """The ordering-fix regression lock: consecutive durable/ephemeral emits
    land on the ONE send queue in call order. The old split routed durable to
    the async outbox and ephemeral straight to the socket, so a later
    ephemeral (agent_end) overtook an earlier durable (assistant_text). Now a
    single FIFO carries both — delivery order == emit order."""
    outbox = _RecordingOutbox()
    writer = _RecordingWriter()
    peer = PeerSession(peer_id="terminal-1", transport_writer=cast(Any, writer))
    rt = _runtime_with(peer, outbox)

    # A realistic turn tail: tool_call (eph), assistant_text (durable),
    # agent_end (eph) — the exact sequence that used to reorder.
    for kind in ("tool_call", "assistant_text", "agent_end"):
        await rt._emit_outbound(_body(kind))

    kinds = []
    while len(peer.send_q):
        kinds.append((await peer.send_q.get()).envelope.body["metadata"]["kind"])
    assert kinds == ["tool_call", "assistant_text", "agent_end"]
