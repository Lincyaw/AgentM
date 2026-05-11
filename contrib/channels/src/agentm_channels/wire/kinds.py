"""Wire envelope kinds.

Mirrors designs/client-server-architecture.md §4.3 exactly. Adding a
kind requires a spec amendment — §0 design discipline.
"""

from __future__ import annotations

KIND_HELLO = "hello"
KIND_WELCOME = "welcome"
KIND_INBOUND = "inbound"
KIND_OUTBOUND = "outbound"
KIND_ACK = "ack"
KIND_ACK_BATCH = "ack_batch"
KIND_PING = "ping"
KIND_PONG = "pong"
KIND_ERROR = "error"
KIND_BYE = "bye"
KIND_DELIVERY_BATCH = "delivery_batch"

VALID_KINDS: frozenset[str] = frozenset(
    {
        KIND_HELLO,
        KIND_WELCOME,
        KIND_INBOUND,
        KIND_OUTBOUND,
        KIND_ACK,
        KIND_ACK_BATCH,
        KIND_PING,
        KIND_PONG,
        KIND_ERROR,
        KIND_BYE,
        KIND_DELIVERY_BATCH,
    }
)

__all__ = [
    "KIND_ACK",
    "KIND_ACK_BATCH",
    "KIND_BYE",
    "KIND_DELIVERY_BATCH",
    "KIND_ERROR",
    "KIND_HELLO",
    "KIND_INBOUND",
    "KIND_OUTBOUND",
    "KIND_PING",
    "KIND_PONG",
    "KIND_WELCOME",
    "VALID_KINDS",
]
