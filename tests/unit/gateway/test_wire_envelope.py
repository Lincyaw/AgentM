"""Fail-stop: wire envelope v2 round-trip + version negotiation.

Protects the wire contract (§2). If encode/decode is not idempotent, or
a v1 envelope is silently accepted, every peer's framing assumptions
break and observation can't be attributed to a session.
"""

from __future__ import annotations

import json

import pytest

from agentm.gateway.wire import (
    WIRE_VERSION,
    Envelope,
    InvalidEnvelope,
    decode,
    encode,
)


def test_v2_round_trip_is_idempotent() -> None:
    env = Envelope(
        v=WIRE_VERSION,
        id="abc123",
        kind="inbound",
        ts=1748400000.42,
        session_key="terminal:t1",
        scenario="general_purpose",
        body={"channel": "terminal", "chat_id": "t1", "content": "hi"},
    )
    frame = encode(env)
    decoded, rest = decode(frame)
    assert rest == b""
    assert encode(decoded) == frame
    assert decoded.session_key == "terminal:t1"
    assert decoded.scenario == "general_purpose"


def test_minimal_envelope_omits_none_conditionals() -> None:
    env = Envelope(v=WIRE_VERSION, id="x", kind="ping", ts=1.0)
    payload = json.loads(encode(env)[4:].decode("utf-8"))
    assert "session_key" not in payload
    assert "scenario" not in payload
    # routing primitives deleted in v2 must never appear
    for gone in ("to", "correlation_id", "hops", "root_session_key", "peer_kind"):
        assert gone not in payload


def test_v1_envelope_is_rejected_hard() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope(v=1, id="x", kind="inbound", ts=1.0, body={})


def test_from_dict_rejects_v1_payload() -> None:
    v1_payload = {"v": 1, "id": "x", "kind": "inbound", "ts": 1.0, "body": {}}
    with pytest.raises(InvalidEnvelope):
        Envelope.from_dict(v1_payload)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope(v=WIRE_VERSION, id="x", kind="delivery_batch", ts=1.0, body={})
