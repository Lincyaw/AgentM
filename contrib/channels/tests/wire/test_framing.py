"""Length-prefixed framing: encode, decode, streaming decode, oversize guard."""

from __future__ import annotations

import struct

import pytest

from agentm_channels.wire import (
    Envelope,
    IncompleteFrame,
    InvalidEnvelope,
    WireError,
    decode,
    decode_stream,
    encode,
)
from agentm_channels.wire.framing import MAX_FRAME_BYTES


def _env(kind: str = "inbound", id_: str = "m1") -> Envelope:
    return Envelope(v=1, id=id_, kind=kind, ts=0.0, body={"k": "v"})






def test_decode_partial_buffer_raises_incomplete() -> None:
    buf = encode(_env())
    # Header truncated.
    with pytest.raises(IncompleteFrame):
        decode(buf[:2])
    # Header complete, body truncated.
    with pytest.raises(IncompleteFrame):
        decode(buf[:6])


def test_decode_stream_drains_back_to_back_frames() -> None:
    a = _env(id_="a")
    b = _env(id_="b")
    c = _env(id_="c")
    buf = encode(a) + encode(b) + encode(c)
    envs, rest = decode_stream(buf)
    assert [e.id for e in envs] == ["a", "b", "c"]
    assert rest == b""






def test_oversized_length_rejected_before_allocation() -> None:
    # Construct a header claiming > MAX_FRAME_BYTES; supply no body.
    bad_len = MAX_FRAME_BYTES + 1
    header = struct.pack(">I", bad_len)
    with pytest.raises(InvalidEnvelope):
        decode(header)
    with pytest.raises(InvalidEnvelope):
        decode_stream(header)


def test_malformed_json_rejected() -> None:
    body = b"not-json"
    buf = struct.pack(">I", len(body)) + body
    with pytest.raises(WireError):
        decode(buf)


def test_decode_rejects_envelope_with_invalid_kind() -> None:
    body = b'{"v":1,"id":"x","kind":"bogus","ts":0,"body":{}}'
    buf = struct.pack(">I", len(body)) + body
    with pytest.raises(InvalidEnvelope):
        decode(buf)
