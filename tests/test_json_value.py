from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentm.core.lib.json_value import json_restore, json_safe


@dataclass(frozen=True, slots=True)
class _Payload:
    data: bytes
    name: str


def test_json_safe_encodes_bytes_inside_dataclasses() -> None:
    assert json_safe(_Payload(data=b"abc", name="payload")) == {
        "data": {"__bytes_hex__": "616263"},
        "name": "payload",
    }


def test_json_restore_rejects_invalid_json_values() -> None:
    with pytest.raises(ValueError, match="encoded JSON numbers must be finite"):
        json_restore(float("nan"))

    with pytest.raises(ValueError, match="encoded value is not JSON-safe"):
        json_restore(object())


def test_json_restore_decodes_bytes_marker() -> None:
    assert json_restore({"__bytes_hex__": "616263"}) == b"abc"
