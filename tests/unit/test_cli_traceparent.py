"""W3C ``TRACEPARENT`` ingestion: AgentM continues a caller-supplied trace.

Pins the contract that lets an upstream dispatcher (workbuddy) link its trace
to AgentM's: a valid ``traceparent`` env var yields ``(trace_id, span_id)`` that
the CLI feeds to ``AgentSessionConfig.root_session_id`` (== OTel trace_id) and
``parent_session_id``; anything malformed/absent falls back to a fresh trace.
"""

from __future__ import annotations

from agentm.cli import _parse_traceparent


VALID = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def test_valid_traceparent() -> None:
    assert _parse_traceparent(VALID) == (
        "0af7651916cd43dd8448eb211c80319c",
        "b7ad6b7169203331",
    )


def test_uppercase_normalized() -> None:
    tid, sid = _parse_traceparent(VALID.upper())  # type: ignore[misc]
    assert tid == "0af7651916cd43dd8448eb211c80319c"
    assert sid == "b7ad6b7169203331"


def test_none_and_empty() -> None:
    assert _parse_traceparent(None) is None
    assert _parse_traceparent("") is None
    assert _parse_traceparent("   ") is None


def test_wrong_field_count() -> None:
    assert _parse_traceparent("00-abc-def") is None
    assert _parse_traceparent("00-trace-span-flags-extra") is None


def test_bad_lengths() -> None:
    assert _parse_traceparent("00-tooshort-b7ad6b7169203331-01") is None
    assert _parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-short-01") is None


def test_non_hex() -> None:
    assert _parse_traceparent("00-zzzz651916cd43dd8448eb211c80319c-b7ad6b7169203331-01") is None


def test_all_zero_rejected() -> None:
    assert _parse_traceparent("00-" + "0" * 32 + "-b7ad6b7169203331-01") is None
    assert _parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-" + "0" * 16 + "-01") is None
