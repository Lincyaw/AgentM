"""Lock the ``llmharness dataset`` reconstruction contract.

Disaster guarded: a regression that mis-pairs extractor inputs with their
outputs would silently produce wrong training data. We synthesize a tiny
session JSONL by hand (no real run required) so the test pins down the
windowing rules:

  * extractor input new_turns = messages[prev_cursor.last_turn_index+1 ..
    this_cursor.last_turn_index]; recent_graph = audit_events appended
    BEFORE this firing's events.
  * auditor input graph = audit_events accumulator at firing time;
    recent_verdicts = trailing window of prior verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmharness.distill.export import (
    extractor_records_from_replay,
    legacy_batch_replay_count,
)


def _otlp_value(value: Any) -> dict[str, Any]:
    """Wrap a Python value in the OTLP ``AnyValue`` tagged-union form.

    Mirrors what ``otel_export`` writes on the live path so TraceReader
    can unwrap the body back to its original shape.
    """

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
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(k), "value": _otlp_value(v)}
                    for k, v in value.items()
                ]
            }
        }
    if isinstance(value, list):
        return {"arrayValue": {"values": [_otlp_value(v) for v in value]}}
    return {"stringValue": json.dumps(value, default=str)}


def _otlp_message_log(entry: dict[str, Any]) -> dict[str, Any]:
    """Wrap a SessionEntry dict as an ``agentm.message.appended`` log line.

    Matches what ``MessageAppendedEvent.to_otel`` emits on the live path:
    ``eventName`` is ``agentm.message.appended`` and the body is the
    SessionEntry dict verbatim. TraceReader.load_messages() yields the
    body, so the rest of the CLI keeps reading ``rec["type"]`` /
    ``rec["payload"]`` unchanged.
    """

    return {
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "agentm"}}
            ]
        },
        "scopeLogs": [
            {
                "scope": {"name": "agentm", "version": "0.1.0"},
                "logRecords": [
                    {
                        "timeUnixNano": "0",
                        "observedTimeUnixNano": "0",
                        "severityNumber": "SEVERITY_NUMBER_INFO",
                        "severityText": "INFO",
                        "eventName": "agentm.message.appended",
                        "body": _otlp_value(entry),
                    }
                ],
            }
        ],
    }


def _write_session(path: Path, records: list[dict]) -> None:
    """Write ``records`` as OTLP/JSON ndjson — one ResourceLogs per line.

    Production code reads sessions through :class:`TraceReader`, which
    expects OTLP wire format; bare SessionEntry dicts on disk would be
    skipped. Records typed ``"session"`` are dropped here because the
    live writer persists the session header as its own log record
    (``agentm.session.header``), not as a message; tests don't exercise
    that path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in records:
            if r.get("type") == "session":
                continue
            h.write(json.dumps(_otlp_message_log(r)))
            h.write("\n")


def _msg(role: str, text: str) -> dict:
    return {
        "type": "message",
        "id": f"m-{role}-{text}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "role": role,
            "content": [{"type": "text", "text": text}],
            "timestamp": 0.0,
        },
    }


def _audit_event(eid: int, kind: str, summary: str) -> dict:
    return {
        "type": "llmharness.audit_event",
        "id": f"e-{eid}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "id": eid,
            "kind": kind,
            "summary": summary,
            "refs": [],
            "source_turns": [],
        },
    }


def _cursor(last_turn_index: int) -> dict:
    return {
        "type": "llmharness.extractor_cursor",
        "id": f"c-{last_turn_index}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "last_turn_index": last_turn_index,
            "extraction_run_id": f"run-{last_turn_index}",
        },
    }


def _verdict(surface_reminder: bool, reminder_text: str = "") -> dict:
    return {
        "type": "llmharness.verdict",
        "id": f"v-{surface_reminder}-{reminder_text}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "surface_reminder": surface_reminder,
            "reminder_text": reminder_text,
            "continuation_notes": [],
            "matched_event_ids": [1] if surface_reminder else [],
            "cited_cards": [],
        },
    }






# --- SFT distill export (extractor_records_from_replay) ----------------------


def _ok_extractor_replay_record(
    *,
    raw_assistant_messages: list[dict] | None = None,
) -> dict:
    """One ok extractor replay-record dict; thinking is opt-in per test."""
    rec: dict = {
        "phase": "extractor",
        "status": "ok",
        "root_session_id": "sess-x",
        "turn_index": 4,
        "ts_ns": 1,
        "payload": {"new_turns": [], "recent_graph": []},
        "output": {
            "events": [
                {"id": 1, "kind": "task", "summary": "go", "source_turns": [0]}
            ],
            "edges": [],
            "dropped_edges": [],
        },
    }
    if raw_assistant_messages is not None:
        rec["raw_assistant_messages"] = raw_assistant_messages
    return rec






def test_dataset_export_skips_legacy_batch_replays() -> None:
    """Legacy v18 replays use ``submit_events_batch``; v19 cannot train on them.

    Disaster guarded: training on the old terminal-batch shape under
    the new tool surface would teach the student a deleted contract.
    The export must skip the record AND a separate counter must report
    how many such records were skipped so the operator notices.
    """
    legacy_rec = _ok_extractor_replay_record(
        raw_assistant_messages=[
            {"type": "thinking", "text": "v18 single shot"},
            {
                "type": "tool_call",
                "name": "submit_events_batch",
                "arguments": {"events": [], "done": True},
            },
        ]
    )
    assert list(extractor_records_from_replay([legacy_rec], sample_id="s-1")) == []
    assert legacy_batch_replay_count([legacy_rec]) == 1


