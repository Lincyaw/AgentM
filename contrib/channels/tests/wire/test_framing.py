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


def test_encode_prepends_4byte_big_endian_length() -> None:
    env = _env()
    buf = encode(env)
    length = struct.unpack(">I", buf[:4])[0]
    assert length == len(buf) - 4
    assert length > 0


def test_round_trip_single_envelope() -> None:
    env = _env()
    buf = encode(env)
    out, rest = decode(buf)
    assert out == env
    assert rest == b""


@pytest.mark.parametrize(
    "env",
    [
        Envelope(v=1, id="a", kind="hello", ts=0.0, body={"peer_kind": "chat_client"}),
        Envelope(v=1, id="b", kind="ack", ts=1.5, body={}),
        Envelope(
            v=1,
            id="c",
            kind="outbound",
            ts=2.0,
            body={"text": "hi", "nested": {"x": [1, 2, 3]}},
            to="peer://xyz",
            correlation_id="cid",
            hops=4,
            root_session_key="feishu:c1",
            peer_kind="agent_worker",
        ),
    ],
)
def test_round_trip_preserves_all_envelope_shapes(env: Envelope) -> None:
    out, rest = decode(encode(env))
    assert out == env
    assert rest == b""


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


def test_decode_stream_returns_trailing_partial() -> None:
    a = _env(id_="a")
    b = _env(id_="b")
    full = encode(a) + encode(b)
    # Cut one byte off the end of b's body.
    truncated = full[:-1]
    envs, rest = decode_stream(truncated)
    assert [e.id for e in envs] == ["a"]
    # The remaining bytes are the partial b-frame: 4-byte header + body-1.
    assert len(rest) == len(encode(b)) - 1


def test_decode_stream_empty_input() -> None:
    envs, rest = decode_stream(b"")
    assert envs == []
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
