"""Per-peer connection state for the asyncio wire server.

See ``.claude/designs/client-server-architecture.md`` §3 (Topology)
and §4.4 (hello/welcome). Tiny module — mechanism only, no policy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class PeerSession:
    """One connected peer's bookkeeping.

    ``transport_writer`` is the live :class:`asyncio.StreamWriter`.
    ``pending_count_hint`` is the last sample of ``outbox.pending_count``
    used by the slow-consumer gate; it is a hint, not a source of truth.
    """

    peer_id: str
    peer_kind: str
    transport_writer: asyncio.StreamWriter
    last_seen: float = 0.0
    pending_count_hint: int = 0
    # Set when the per-peer delivery worker should pause pulling new
    # leases (slow-consumer high-water tripped). Cleared when drained
    # below high_water / 2.
    backpressure: bool = False
    capabilities: dict[str, object] = field(default_factory=dict)
    # Connection identity bound to the peer at connect time. The wire
    # bridge uses it as the **fallback author** for inbound bodies that
    # omit ``sender_id`` (single-user transports carry no per-message
    # identity). It is NOT compared against a client's reported
    # ``sender_id`` — a chat client relays many humans, so the
    # per-message author is the client's to assert; peer impersonation
    # is prevented by channel binding, not by author equality. ``None``
    # means "derive from peer_id" — the current default for every
    # transport (unix/peercred/ws/token). A future authenticator that
    # learns a distinct identity (e.g. an OIDC subject claim) may set
    # this directly so the fallback picks it up without code changes.
    principal: str | None = None

    @property
    def bound_principal(self) -> str:
        """The fallback author for inbounds that omit ``sender_id``.

        Defaults to :attr:`peer_id` when no transport-level principal
        was set. Centralised so callers don't reimplement the fallback.
        """
        return self.principal if self.principal is not None else self.peer_id


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

    def __iter__(self) -> "object":
        return iter(list(self._peers.values()))

    def __len__(self) -> int:
        return len(self._peers)


__all__ = ["PeerRegistry", "PeerSession"]
