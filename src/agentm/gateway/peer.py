"""Per-peer connection state for the wire server (§3.1).

Tiny module — mechanism only, no policy. Only one peer kind exists in v2
(``chat_client``), so there is no ``peer_kind`` field.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class PeerSession:
    """One connected chat-client peer's bookkeeping.

    ``transport_writer`` is the live :class:`asyncio.StreamWriter`.
    ``pending_count_hint`` is the last sample of ``outbox.pending_count``
    used by the slow-consumer gate; it is a hint, not a source of truth.
    """

    peer_id: str
    transport_writer: asyncio.StreamWriter
    last_seen: float = 0.0
    pending_count_hint: int = 0
    # Set when the per-peer delivery worker trips the slow-consumer
    # high-water mark. Cleared when drained below high_water / 2.
    backpressure: bool = False
    capabilities: dict[str, object] = field(default_factory=dict)


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
