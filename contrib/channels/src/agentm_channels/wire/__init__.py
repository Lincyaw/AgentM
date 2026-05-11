"""Wire protocol: envelopes, kinds, length-prefixed framing.

Pure-function foundation for the gateway/client process split. See
``.claude/designs/client-server-architecture.md`` §4. Pure module —
no I/O, no asyncio. Importable in a notebook with no harness.
"""

from __future__ import annotations

from .envelope import WIRE_VERSION, Envelope
from .errors import IncompleteFrame, InvalidEnvelope, WireError
from .framing import HEADER_BYTES, MAX_FRAME_BYTES, decode, decode_stream, encode
from .kinds import (
    KIND_ACK,
    KIND_ACK_BATCH,
    KIND_BYE,
    KIND_DELIVERY_BATCH,
    KIND_ERROR,
    KIND_HELLO,
    KIND_INBOUND,
    KIND_OUTBOUND,
    KIND_PING,
    KIND_PONG,
    KIND_WELCOME,
    VALID_KINDS,
)

__all__ = [
    "HEADER_BYTES",
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
    "MAX_FRAME_BYTES",
    "VALID_KINDS",
    "WIRE_VERSION",
    "Envelope",
    "IncompleteFrame",
    "InvalidEnvelope",
    "WireError",
    "decode",
    "decode_stream",
    "encode",
]
