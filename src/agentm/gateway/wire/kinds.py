"""Wire envelope kinds (v2).

Seven kinds, half of v1's set. See
``.claude/designs/single-process-gateway.md`` §2.3. Adding a kind
requires a spec amendment.

Peer -> Gateway: ``hello``, ``inbound``, ``ack``, ``pong``.
Gateway -> Peer: ``welcome``, ``outbound``, ``error``, ``ping``.
"""

from __future__ import annotations

KIND_HELLO = "hello"
KIND_WELCOME = "welcome"
KIND_INBOUND = "inbound"
KIND_OUTBOUND = "outbound"
KIND_ACK = "ack"
KIND_PING = "ping"
KIND_PONG = "pong"
KIND_ERROR = "error"

VALID_KINDS: frozenset[str] = frozenset(
    {
        KIND_HELLO,
        KIND_WELCOME,
        KIND_INBOUND,
        KIND_OUTBOUND,
        KIND_ACK,
        KIND_PING,
        KIND_PONG,
        KIND_ERROR,
    }
)

__all__ = [
    "KIND_ACK",
    "KIND_ERROR",
    "KIND_HELLO",
    "KIND_INBOUND",
    "KIND_OUTBOUND",
    "KIND_PING",
    "KIND_PONG",
    "KIND_WELCOME",
    "VALID_KINDS",
]
