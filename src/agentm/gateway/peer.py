"""Per-peer connection state for the wire server (§3.1).

Tiny module — mechanism only, no policy. Only one peer kind exists in v2
(``chat_client``), so there is no ``peer_kind`` field.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field

from agentm.gateway.send_queue import SendQueue


@dataclass(slots=True)
class PeerSession:
    """One connected chat-client peer's bookkeeping.

    ``transport_writer`` is the live :class:`asyncio.StreamWriter`.
    ``pending_count_hint`` is the last sampled depth of :attr:`send_q`
    used by the slow-consumer gate; it is a hint, not a source of truth.
    """

    peer_id: str
    transport_writer: asyncio.StreamWriter
    cwd: str | None = None
    last_seen: float = 0.0
    pending_count_hint: int = 0
    # Set when the sender trips the slow-consumer high-water mark on the
    # send queue. Cleared when it drains below high_water / 2.
    backpressure: bool = False
    capabilities: dict[str, object] = field(default_factory=dict)
    # Serialises ALL writes to ``transport_writer``. The per-peer sender is
    # the usual writer, but handshake/pong frames (``server._send``) also
    # write here; the WebSocket adapter coalesces buffered writes into ONE
    # ``ws.send`` per ``drain()``, so two concurrent ``write()+drain()``
    # pairs would merge two wire frames into a single WS message and corrupt
    # the stream. Every writer acquires this lock around its write+drain so
    # each frame maps to exactly one flush.
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Unified per-peer ordered send queue (§2.6): every outbound — durable
    # and ephemeral — is enqueued here and drained by the single sender
    # task, which is what guarantees delivery order. The server replaces
    # this with one sized to its slow-consumer high-water mark.
    send_q: SendQueue = field(default_factory=SendQueue)


class PeerRegistry:
    """Single-loop in-memory registry of connected peers.

    No locking — asyncio is single-threaded. Connection rejection on
    duplicate ``peer_id`` is the caller's policy decision.
    """

    def __init__(self) -> None:
        self._peers: dict[str, PeerSession] = {}

    def register(self, session: PeerSession) -> None:
        self._peers[session.peer_id] = session

    def deregister(self, peer_id: str) -> None:
        self._peers.pop(peer_id, None)

    def get(self, peer_id: str) -> PeerSession | None:
        return self._peers.get(peer_id)

    def __contains__(self, peer_id: object) -> bool:
        return isinstance(peer_id, str) and peer_id in self._peers

    def __iter__(self) -> "Iterator[PeerSession]":
        return iter(list(self._peers.values()))

    def __len__(self) -> int:
        return len(self._peers)


__all__ = ["PeerRegistry", "PeerSession"]
