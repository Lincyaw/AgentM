"""Length-prefixed JSON framing.

Per designs/client-server-architecture.md §4.1: each frame is a 4-byte
big-endian unsigned length followed by UTF-8 JSON. Pure functions —
no sockets, no async, no logging.
"""

from __future__ import annotations

import json
import struct

from .envelope import Envelope
from .errors import IncompleteFrame, InvalidEnvelope, WireError

HEADER_BYTES: int = 4
MAX_FRAME_BYTES: int = 16 * 1024 * 1024  # 16 MiB — §0 discipline, raise via spec only.

_HEADER_STRUCT = struct.Struct(">I")


def encode(env: Envelope) -> bytes:
    """Serialize one envelope to a length-prefixed frame."""
    payload = json.dumps(env.to_dict(), separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise InvalidEnvelope(
            f"encoded envelope is {len(payload)} bytes; exceeds "
            f"MAX_FRAME_BYTES={MAX_FRAME_BYTES}"
        )
    return _HEADER_STRUCT.pack(len(payload)) + payload


def decode(buf: bytes) -> tuple[Envelope, bytes]:
    """Decode the first complete frame from ``buf``.

    Returns ``(envelope, remaining)``. Raises :class:`IncompleteFrame`
    if ``buf`` does not yet hold a full frame. Raises
    :class:`InvalidEnvelope` if the declared length exceeds
    :data:`MAX_FRAME_BYTES` (checked before any allocation of the
    body). Raises :class:`WireError` for malformed JSON.
    """
    if len(buf) < HEADER_BYTES:
        raise IncompleteFrame(
            f"need {HEADER_BYTES} header bytes, have {len(buf)}"
        )
    (length,) = _HEADER_STRUCT.unpack_from(buf, 0)
    if length > MAX_FRAME_BYTES:
        # Reject *before* trying to slice / allocate the body.
        raise InvalidEnvelope(
            f"declared frame length {length} exceeds MAX_FRAME_BYTES={MAX_FRAME_BYTES}"
        )
    end = HEADER_BYTES + length
    if len(buf) < end:
        raise IncompleteFrame(
            f"need {length} body bytes, have {len(buf) - HEADER_BYTES}"
        )
    body_bytes = buf[HEADER_BYTES:end]
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireError(f"frame body is not valid UTF-8 JSON: {exc}") from exc
    env = Envelope.from_dict(payload)
    return env, buf[end:]


def decode_stream(buf: bytes) -> tuple[list[Envelope], bytes]:
    """Drain every complete frame from ``buf``.

    Returns ``(envelopes, trailing)`` where ``trailing`` is the partial
    bytes of the next not-yet-complete frame (possibly empty). Used by
    socket readers that accumulate bytes and want to dispatch all ready
    envelopes in one pass.
    """
    envelopes: list[Envelope] = []
    rest = buf
    while True:
        try:
            env, rest = decode(rest)
        except IncompleteFrame:
            return envelopes, rest
        envelopes.append(env)


__all__ = [
    "HEADER_BYTES",
    "MAX_FRAME_BYTES",
    "decode",
    "decode_stream",
    "encode",
]
