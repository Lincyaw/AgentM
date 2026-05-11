"""Kinds must match designs/client-server-architecture.md §4.3 exactly."""

from __future__ import annotations

from agentm_channels.wire import kinds


def test_kind_string_values() -> None:
    assert kinds.KIND_HELLO == "hello"
    assert kinds.KIND_WELCOME == "welcome"
    assert kinds.KIND_INBOUND == "inbound"
    assert kinds.KIND_OUTBOUND == "outbound"
    assert kinds.KIND_ACK == "ack"
    assert kinds.KIND_ACK_BATCH == "ack_batch"
    assert kinds.KIND_PING == "ping"
    assert kinds.KIND_PONG == "pong"
    assert kinds.KIND_ERROR == "error"
    assert kinds.KIND_BYE == "bye"
    assert kinds.KIND_DELIVERY_BATCH == "delivery_batch"


def test_valid_kinds_frozenset_exact_membership() -> None:
    expected = {
        "hello",
        "welcome",
        "inbound",
        "outbound",
        "ack",
        "ack_batch",
        "ping",
        "pong",
        "error",
        "bye",
        "delivery_batch",
    }
    assert kinds.VALID_KINDS == frozenset(expected)
    assert len(kinds.VALID_KINDS) == 11
    assert isinstance(kinds.VALID_KINDS, frozenset)
