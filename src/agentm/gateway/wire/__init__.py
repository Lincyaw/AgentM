"""Wire protocol v2: envelopes, kinds, length-prefixed framing, body types.

Pure-function foundation for the single-process gateway. See
``.claude/designs/single-process-gateway.md`` §2. Pure module — no I/O,
no asyncio.
"""

from __future__ import annotations

from .envelope import WIRE_VERSION, Envelope
from .errors import IncompleteFrame, InvalidEnvelope, WireError
from .framing import HEADER_BYTES, MAX_FRAME_BYTES, decode, decode_stream, encode
from .kinds import (
    KIND_ACK,
    KIND_ERROR,
    KIND_HELLO,
    KIND_INBOUND,
    KIND_OUTBOUND,
    KIND_PING,
    KIND_PONG,
    KIND_WELCOME,
    VALID_KINDS,
)
from .types import Button, ButtonStyle, InboundBody, OutboundBody, OutboundMetaKind

__all__ = [
    "HEADER_BYTES",
    "KIND_ACK",
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
    "Button",
    "ButtonStyle",
    "Envelope",
    "InboundBody",
    "IncompleteFrame",
    "InvalidEnvelope",
    "OutboundBody",
    "OutboundMetaKind",
    "WireError",
    "decode",
    "decode_stream",
    "encode",
]
