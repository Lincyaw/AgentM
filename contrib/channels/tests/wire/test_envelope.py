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






def test_from_dict_missing_required_field_rejected() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope.from_dict({"v": 1, "kind": "ack", "ts": 0.0, "body": {}})


def test_from_dict_non_dict_body_rejected() -> None:
    with pytest.raises(InvalidEnvelope):
        Envelope.from_dict(
            {"v": 1, "id": "x", "kind": "ack", "ts": 0.0, "body": "not-a-dict"}
        )
