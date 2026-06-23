"""Wire protocol: framing roundtrip and envelope validation."""

from __future__ import annotations

import pytest

from agentm.gateway.wire.envelope import Envelope, WIRE_VERSION
from agentm.gateway.wire.errors import IncompleteFrame, InvalidEnvelope
from agentm.gateway.wire.framing import MAX_FRAME_BYTES, decode, decode_stream, encode


def _make_envelope(**overrides: object) -> Envelope:
    defaults: dict[str, object] = dict(
        v=WIRE_VERSION, id="test-1", kind="hello", ts=1.0, body={}
    )
    defaults.update(overrides)
    return Envelope(**defaults)  # type: ignore[arg-type]


# -- Framing roundtrip -------------------------------------------------------


def test_encode_decode_roundtrip() -> None:
    env = _make_envelope(body={"msg": "hi"})
    frame = encode(env)
    decoded, rest = decode(frame)
    assert rest == b""
    assert decoded.v == env.v
    assert decoded.id == env.id
    assert decoded.kind == env.kind
    assert decoded.ts == env.ts
    assert decoded.body == env.body
    assert decoded.session_key is None
    assert decoded.scenario is None


def test_encode_decode_with_optional_fields() -> None:
    env = _make_envelope(
        kind="inbound",
        session_key="terminal:t1",
        scenario="local",
        body={"content": "x"},
    )
    frame = encode(env)
    decoded, rest = decode(frame)
    assert rest == b""
    assert decoded.session_key == "terminal:t1"
    assert decoded.scenario == "local"
    assert decoded.body == {"content": "x"}


def test_decode_stream_multiple_frames() -> None:
    envs = [
        _make_envelope(id=f"m-{i}", kind="ping") for i in range(3)
    ]
    buf = b"".join(encode(e) for e in envs)
    decoded, remainder = decode_stream(buf)
    assert len(decoded) == 3
    assert remainder == b""
    for orig, got in zip(envs, decoded):
        assert got.id == orig.id


def test_decode_incomplete_header() -> None:
    with pytest.raises(IncompleteFrame):
        decode(b"\x00\x00")


def test_decode_incomplete_body() -> None:
    # Header claims 100 bytes, but only 10 body bytes follow.
    header = (100).to_bytes(4, "big")
    buf = header + b"x" * 10
    with pytest.raises(IncompleteFrame):
        decode(buf)


def test_decode_stream_with_trailing_partial() -> None:
    env1 = _make_envelope(id="full-1", kind="ping")
    env2 = _make_envelope(id="full-2", kind="pong")
    full_buf = encode(env1) + encode(env2)
    # Chop 3 bytes off the end so env2's frame is incomplete.
    chopped = full_buf[:-3]
    decoded, trailing = decode_stream(chopped)
    assert len(decoded) == 1
    assert decoded[0].id == "full-1"
    assert len(trailing) > 0


# -- Envelope validation -----------------------------------------------------


def test_envelope_rejects_wrong_version() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope(v=1, id="x", kind="hello", ts=1.0, body={})


def test_envelope_rejects_empty_id() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope(v=WIRE_VERSION, id="", kind="hello", ts=1.0, body={})


def test_envelope_rejects_invalid_kind() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope(v=WIRE_VERSION, id="x", kind="bogus", ts=1.0, body={})


def test_from_dict_rejects_missing_field() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope.from_dict({"v": 2, "id": "x"})


# -- Oversize -----------------------------------------------------------------


def test_encode_rejects_oversize_frame() -> None:
    # A body large enough to exceed MAX_FRAME_BYTES when JSON-encoded.
    huge_body = {"data": "x" * (MAX_FRAME_BYTES + 1)}
    env = _make_envelope(body=huge_body)
    with pytest.raises(InvalidEnvelope, match="MAX_FRAME_BYTES"):
        encode(env)
