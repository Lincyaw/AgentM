"""Fail-stop tests for the single-event-log merge.

Two positions documented in ``.claude/designs/single-event-log.md``:

1. ``continue_recent`` must rebuild the same in-memory trajectory from a
   merged OTLP/JSON ndjson file containing interleaved
   ``agentm.message.appended`` log records and other OTLP-shaped lines
   (spans, unrelated log records). Wrong → resumes silently truncate or
   drop conversation history.
2. The substrate-installed ``SessionTelemetry`` blocking batch processor
   (PR-A) is the queue contract; backpressure / atexit drain are locked
   down by ``tests/unit/core/test_otel_export.py``. We don't restate
   those properties here — the seam is the SDK, not a bespoke sink.

We exercise the load path through a hand-crafted OTLP/JSON file rather
than driving a real ``observability.install`` because the load contract
(filter log records by ``eventName``, pick out ``agentm.session.header``
+ ``agentm.message.appended``) must hold for any file the bus + sink
ever produce — including hand-edited fixtures from replay tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.core.abi import text_message
from agentm.core.abi.session import (
    CURRENT_SESSION_VERSION,
    SessionHeader,
    message_entry,
)
from agentm.core.runtime.session_manager import SessionManager


_RESOURCE_AGENTM = {
    "attributes": [
        {"key": "service.name", "value": {"stringValue": "agentm"}},
    ]
}


def _kvlist(d: dict[str, Any]) -> dict[str, Any]:
    """Build an OTLP ``kvlistValue`` wrapper for a Python dict."""
    return {
        "kvlistValue": {
            "values": [
                {"key": str(k), "value": _otlp_value(v)} for k, v in d.items()
            ]
        }
    }


def _otlp_value(value: Any) -> dict[str, Any]:
    """Wrap a Python value in the OTLP proto-JSON tagged-union shape."""
    if value is None:
        return {"stringValue": ""}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, dict):
        return _kvlist(value)
    if isinstance(value, list):
        return {"arrayValue": {"values": [_otlp_value(v) for v in value]}}
    return {"stringValue": json.dumps(value, default=str)}


def _wrap_log_record(event_name: str, body: dict[str, Any]) -> dict[str, Any]:
    """One ``ResourceLogs`` element line as PR-A's exporter writes them.

    Each line is a self-contained ``ResourceLogs`` element (no outer
    ``resourceLogs`` wrapper — that's the request envelope, not the
    on-disk shape).
    """
    return {
        "resource": dict(_RESOURCE_AGENTM),
        "scopeLogs": [
            {
                "scope": {"name": "agentm", "version": "0.1.0"},
                "logRecords": [
                    {
                        "timeUnixNano": "0",
                        "observedTimeUnixNano": "0",
                        "severityNumber": "SEVERITY_NUMBER_INFO",
                        "severityText": "INFO",
                        "eventName": event_name,
                        "body": _kvlist(body),
                    }
                ],
            }
        ],
    }


def _wrap_other_span() -> dict[str, Any]:
    """A ``ResourceSpans`` element line ``_load`` must ignore."""
    return {
        "resource": dict(_RESOURCE_AGENTM),
        "scopeSpans": [
            {
                "scope": {"name": "agentm", "version": "0.1.0"},
                "spans": [
                    {
                        "traceId": "AAAA",
                        "spanId": "AAAA",
                        "name": "chat m-stub",
                        "kind": "SPAN_KIND_CLIENT",
                        "startTimeUnixNano": "0",
                        "endTimeUnixNano": "1",
                        "status": {},
                    }
                ],
            }
        ],
    }


def _wrap_unrelated_log() -> dict[str, Any]:
    """A log record with an event_name SessionManager must ignore."""
    return _wrap_log_record(
        "agentm.turn.summary",
        {"turn_index": 0, "tool_call_count": 0},
    )


def test_continue_recent_reads_interleaved_merged_log(tmp_path: Path) -> None:
    """The trajectory rebuilt from the OTLP/JSON merged log must equal the
    trajectory that would have been built from a clean in-memory session.

    We hand-craft a file containing the two SessionManager-relevant log
    records interleaved with unrelated spans and other-event log records;
    ``SessionManager.open`` must pick the header + messages by event_name
    and ignore the rest.
    """

    cwd = tmp_path
    obs_dir = cwd / ".agentm" / "observability"
    obs_dir.mkdir(parents=True)
    sid = "abcd1234"
    log = obs_dir / f"{sid}.jsonl"

    header = SessionHeader(
        type="session",
        version=CURRENT_SESSION_VERSION,
        id=sid,
        timestamp=1.0,
        cwd=str(cwd),
        parent_session=None,
    )
    e1 = message_entry(text_message("hello"), parent_id=None)
    e2 = message_entry(text_message("world"), parent_id=e1.id)

    from agentm.core.runtime.session_manager import _entry_to_record, _header_to_record

    rows = [
        _wrap_other_span(),
        _wrap_log_record("agentm.session.header", _header_to_record(header)),
        _wrap_unrelated_log(),
        _wrap_log_record("agentm.message.appended", _entry_to_record(e1)),
        _wrap_other_span(),
        _wrap_log_record("agentm.message.appended", _entry_to_record(e2)),
        _wrap_other_span(),
    ]
    with log.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    mgr = SessionManager.continue_recent(str(cwd))
    assert mgr.get_session_id() == sid
    assert mgr.get_cwd() == str(cwd)

    entries = mgr.get_entries()
    assert [e.id for e in entries] == [e1.id, e2.id]
    assert mgr.get_leaf_id() == e2.id

    expected = SessionManager.in_memory(cwd=str(cwd))
    expected.new_session(id=sid)
    expected.append(e1)
    expected.append(e2)
    assert [e.id for e in mgr.get_entries()] == [e.id for e in expected.get_entries()]
    assert mgr.get_session_id() == expected.get_session_id()


