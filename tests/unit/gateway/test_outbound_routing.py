"""Fail-stop: outbound routes to the peer serving its channel (§3.2).

The bug this guards against: with a Feishu peer and a terminal peer
connected to one gateway, a reply for the Feishu chat must NOT be
enqueued to the terminal peer (and vice versa). The earlier code
broadcast every outbound to every peer, so a second connected client of
a different platform received traffic it did not own. ``_route_targets``
is the pure selection that fixes it.
"""

from __future__ import annotations

from typing import Any, cast

from agentm.gateway.cli import _route_targets
from agentm.gateway.peer import PeerSession


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
