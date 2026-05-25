"""Unit tests for the ``agentm trace`` subcommand.

The CLI is a contract surface — these tests pin the agent-facing pieces:

* Source resolution (``--file`` / ``--session`` / ``--latest`` mutual
  exclusion and missing-source errors with the right exit codes).
* Output goes to stdout, info / errors to stderr (cli-design §3).
* ``--format`` enum is validated; ``ndjson`` / ``json`` / ``text``
  produce the documented shapes.
* Per-verb filters (``messages --role``, ``tools --tool``) project
  correctly.

We synthesise tiny OTLP/JSON fixtures rather than running a real agent
session — same wire shape, deterministic across CI environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentm.cli_trace import app


def _otlp_value(value: Any) -> dict[str, Any]:
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


def _log_line(event_name: str, body: Any, attrs: dict[str, Any] | None = None) -> str:
    record: dict[str, Any] = {
        "timeUnixNano": "0",
        "observedTimeUnixNano": "0",
        "severityNumber": "SEVERITY_NUMBER_INFO",
        "eventName": event_name,
        "body": _otlp_value(body),
    }
    if attrs:
        record["attributes"] = [
            {"key": k, "value": _otlp_value(v)} for k, v in attrs.items()
        ]
    return json.dumps(
        {
            "scopeLogs": [
                {
                    "scope": {"name": "agentm", "version": "0.1.0"},
                    "logRecords": [record],
                }
            ]
        }
    )


def _span_line(name: str, attrs: dict[str, Any]) -> str:
    return json.dumps(
        {
            "scopeSpans": [
                {
                    "scope": {"name": "agentm", "version": "0.1.0"},
                    "spans": [
                        {
                            "name": name,
                            "spanId": "AAAA",
                            "kind": "SPAN_KIND_INTERNAL",
                            "startTimeUnixNano": "1000",
                            "endTimeUnixNano": "2000",
                            "attributes": [
                                {"key": k, "value": _otlp_value(v)}
                                for k, v in attrs.items()
                            ],
                            "status": {},
                        }
                    ],
                }
            ]
        }
    )


def _message(role: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": f"m-{role}-{text}",
        "parent_id": None,
        "timestamp": 0,
        "payload": {
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
    }


@pytest.fixture
def trace_file(tmp_path: Path) -> Path:
    """Synthesise a tiny session JSONL with one of every record shape."""

    path = tmp_path / "session.jsonl"
    lines: list[str] = [
        _log_line(
            "agentm.session.header",
            {"id": "s1", "cwd": str(tmp_path), "scenario": "test"},
        ),
        _log_line(
            "agentm.session.fingerprint",
            {
                "task_meta": {"task_class": "demo", "task_id": "t1"},
                "atoms": {"read_file": "abcd"},
            },
        ),
        _log_line(
            "agentm.message.appended",
            _message("user", "hi"),
            {"agentm.message.type": "message"},
        ),
        _log_line(
            "agentm.message.appended",
            _message("assistant", "hello back"),
            {"agentm.message.type": "message"},
        ),
        _log_line(
            "agentm.turn.summary",
            {
                "turn_index": 0,
                "stop_reason": "stop",
                "tool_calls": [],
                "tool_call_count": 0,
                "tool_error_count": 0,
                "input_tokens": 10,
                "output_tokens": 5,
            },
        ),
        _span_line(
            "chat demo-model",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "demo-model",
                "agentm.turn.index": 0,
                "agentm.llm.message_count": 1,
                "agentm.llm.duration_ns": 1_500_000_000,
            },
        ),
        _span_line(
            "execute_tool write",
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "write",
                "gen_ai.tool.call.arguments": json.dumps({"path": "x.txt"}),
                "gen_ai.tool.call.result": json.dumps({"ok": True}),
            },
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def test_no_source_exits_2() -> None:
    """Missing all of --file / --session / --latest is exit 2."""

    runner = CliRunner()
    result = runner.invoke(app, ["messages"])
    assert result.exit_code == 2
    err = json.loads(result.stderr.splitlines()[0])
    assert err["kind"] == "argument"








# ---------------------------------------------------------------------------
# stdout/stderr separation + format contract
# ---------------------------------------------------------------------------


def test_data_on_stdout_info_on_stderr(trace_file: Path) -> None:
    """Records → stdout; the ``N message(s)`` notice → stderr."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(trace_file), "--format", "ndjson"]
    )
    assert result.exit_code == 0
    stdout_lines = [
        ln for ln in result.stdout.splitlines() if ln.strip()
    ]
    assert len(stdout_lines) == 2
    for ln in stdout_lines:
        parsed = json.loads(ln)
        assert parsed["type"] == "message"
    assert "message(s)" in result.stderr








# ---------------------------------------------------------------------------
# index — directory-granular session topology
# ---------------------------------------------------------------------------


def _session_start_line(
    *,
    session_id: str,
    root_session_id: str,
    parent_session_id: str | None,
    purpose: str | None,
    scenario: str | None,
) -> str:
    """Synthesise an ``agentm.session.start`` record matching the real writer.

    The writer stamps identity into the body as plain keys and mirrors it
    into OTel-prefixed attributes; ``index`` reads the body first.
    """

    body = {
        "session_id": session_id,
        "root_session_id": root_session_id,
        "parent_session_id": parent_session_id,
        "purpose": purpose,
        "scenario": scenario,
    }
    attrs = {
        "agentm.session.id": session_id,
        "agentm.session.root_id": root_session_id,
        "agentm.session.parent_id": parent_session_id,
        "agentm.session.purpose": purpose,
        "agentm.session.scenario": scenario,
    }
    return _log_line("agentm.session.start", body, attrs)


def test_index_emits_one_identity_row_per_session_start(tmp_path: Path) -> None:
    """``index`` maps each session.start file to a trace-tree identity row.

    Fail-stop: this is the only path from a stored ``trace_id`` to its
    session files. If a row goes missing or loses its parent/purpose edge,
    trace-tree reconstruction breaks silently.
    """

    obs = tmp_path / "obs"
    obs.mkdir()
    # Root + one auditor child sharing one trace_id, plus a file with no
    # session.start that must be skipped.
    (obs / "root.jsonl").write_text(
        _session_start_line(
            session_id="root1",
            root_session_id="traceAAA",
            parent_session_id=None,
            purpose="root",
            scenario="rca:harness.sync",
        )
        + "\n",
        encoding="utf-8",
    )
    (obs / "auditor.jsonl").write_text(
        _session_start_line(
            session_id="aud1",
            root_session_id="traceAAA",
            parent_session_id="root1",
            purpose="cognitive_audit_auditor",
            scenario=None,
        )
        + "\n",
        encoding="utf-8",
    )
    (obs / "noise.jsonl").write_text(
        _log_line("agentm.message.appended", _message("user", "hi")) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app, ["index", "--dir", str(obs), "--format", "ndjson"]
    )
    assert result.exit_code == 0
    rows = [json.loads(ln) for ln in result.stdout.splitlines() if ln.strip()]
    by_session = {r["session_id"]: r for r in rows}
    # The session.start-less file is skipped; exactly two identity rows.
    assert set(by_session) == {"root1", "aud1"}

    root = by_session["root1"]
    assert root["trace_id"] == "traceAAA"
    assert root["parent_session_id"] is None
    assert root["purpose"] == "root"
    assert root["scenario"] == "rca:harness.sync"
    assert root["records"] == 1

    aud = by_session["aud1"]
    assert aud["trace_id"] == "traceAAA"
    assert aud["parent_session_id"] == "root1"
    assert aud["purpose"] == "cognitive_audit_auditor"
    # Empty-string scenario (OTLP encoding of None) normalises back to None.
    assert aud["scenario"] is None


def test_index_missing_dir_exits_3(tmp_path: Path) -> None:
    """A non-existent observability dir is a structured exit-3 not_found."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["index", "--dir", str(tmp_path / "nope"), "--format", "ndjson"]
    )
    assert result.exit_code == 3
    err = json.loads(result.stderr.splitlines()[0])
    assert err["kind"] == "not_found"


# ---------------------------------------------------------------------------
# Verb-specific projections
# ---------------------------------------------------------------------------














