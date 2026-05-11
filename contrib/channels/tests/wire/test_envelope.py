"""Envelope validation and round-trip serialization."""

from __future__ import annotations

import pytest

from agentm_channels.wire import Envelope, InvalidEnvelope


def _ok_kwargs() -> dict[str, object]:
    return {
        "v": 1,
        "id": "msg-1",
        "kind": "inbound",
        "ts": 1234567890.0,
        "body": {"text": "hi"},
    }


def test_construction_with_required_fields_sets_defaults() -> None:
    env = Envelope(**_ok_kwargs())  # type: ignore[arg-type]
    assert env.v == 1
    assert env.id == "msg-1"
    assert env.kind == "inbound"
    assert env.ts == 1234567890.0
    assert env.body == {"text": "hi"}
    assert env.to is None
    assert env.correlation_id is None
    assert env.hops == 0
    assert env.root_session_key is None
    assert env.peer_kind is None


def test_construction_with_all_optional_fields() -> None:
    env = Envelope(
        v=1,
        id="m",
        kind="outbound",
        ts=0.0,
        body={},
        to="peer://abc",
        correlation_id="c1",
        hops=2,
        root_session_key="feishu:c1",
        peer_kind="agent_worker",
    )
    assert env.to == "peer://abc"
    assert env.correlation_id == "c1"
    assert env.hops == 2
    assert env.root_session_key == "feishu:c1"
    assert env.peer_kind == "agent_worker"


def test_invalid_kind_rejected() -> None:
    kwargs = _ok_kwargs()
    kwargs["kind"] = "nope"
    with pytest.raises(InvalidEnvelope):
        Envelope(**kwargs)  # type: ignore[arg-type]


def test_invalid_version_rejected() -> None:
    kwargs = _ok_kwargs()
    kwargs["v"] = 2
    with pytest.raises(InvalidEnvelope):
        Envelope(**kwargs)  # type: ignore[arg-type]


def test_empty_id_rejected() -> None:
    kwargs = _ok_kwargs()
    kwargs["id"] = ""
    with pytest.raises(InvalidEnvelope):
        Envelope(**kwargs)  # type: ignore[arg-type]


def test_to_dict_omits_none_optionals_but_keeps_required() -> None:
    env = Envelope(**_ok_kwargs())  # type: ignore[arg-type]
    d = env.to_dict()
    assert d["v"] == 1
    assert d["id"] == "msg-1"
    assert d["kind"] == "inbound"
    assert d["ts"] == 1234567890.0
    assert d["body"] == {"text": "hi"}
    assert d["hops"] == 0
    # None-valued optionals are omitted.
    assert "to" not in d
    assert "correlation_id" not in d
    assert "root_session_key" not in d
    assert "peer_kind" not in d


def test_round_trip_preserves_all_fields() -> None:
    env = Envelope(
        v=1,
        id="m",
        kind="hello",
        ts=42.5,
        body={"peer_kind": "chat_client"},
        to="chat://feishu/oc_x",
        correlation_id="corr-7",
        hops=3,
        root_session_key="feishu:c2",
        peer_kind="chat_client",
    )
    again = Envelope.from_dict(env.to_dict())
    assert again == env


def test_from_dict_missing_required_field_rejected() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope.from_dict({"v": 1, "kind": "ack", "ts": 0.0, "body": {}})


def test_from_dict_non_dict_body_rejected() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope.from_dict(
            {"v": 1, "id": "x", "kind": "ack", "ts": 0.0, "body": "not-a-dict"}
        )
