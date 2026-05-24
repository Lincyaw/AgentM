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


def test_two_sources_exits_2(trace_file: Path) -> None:
    """--file + --latest is a mutual-exclusion violation → exit 2."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(trace_file), "--latest"]
    )
    assert result.exit_code == 2


def test_missing_file_exits_3(tmp_path: Path) -> None:
    """--file pointing at a non-existent path is exit 3 (not_found)."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(tmp_path / "nope.jsonl")]
    )
    assert result.exit_code == 3


def test_latest_picks_newest(tmp_path: Path) -> None:
    """--latest selects the most-recently-modified *.jsonl in obs dir."""

    obs = tmp_path / ".agentm" / "observability"
    obs.mkdir(parents=True)
    old = obs / "old.jsonl"
    new = obs / "new.jsonl"
    old.write_text(
        _log_line("agentm.session.header", {"id": "old"}) + "\n", encoding="utf-8"
    )
    new.write_text(
        _log_line("agentm.session.header", {"id": "new"}) + "\n", encoding="utf-8"
    )
    import os

    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_800_000_000, 1_800_000_000))

    runner = CliRunner()
    result = runner.invoke(
        app, ["info", "--latest", "--cwd", str(tmp_path), "--what", "header", "--format", "ndjson"]
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(result.stdout.strip())
    # ``info`` emits ``{"header": {...}}`` (or {"fingerprint": ...}) per
    # source; verify --latest picked ``new.jsonl`` by id.
    assert body["header"]["id"] == "new"


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


def test_format_text_renders_human_lines(trace_file: Path) -> None:
    """``--format text`` produces one prefixed line per record."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(trace_file), "--format", "text"]
    )
    assert result.exit_code == 0
    out = result.stdout
    # New format: one header line per message ("[user]" / "[assistant]"),
    # followed by indented content blocks. Both must appear.
    assert "[user]" in out
    assert "[assistant]" in out
    assert "  hi" in out
    assert "  hello back" in out


def test_format_json_emits_array(trace_file: Path) -> None:
    """``--format json`` emits a single JSON array."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(trace_file), "--format", "json"]
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_format_invalid_exits_2(trace_file: Path) -> None:
    """``--format yolo`` is exit 2 with a structured error."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(trace_file), "--format", "yolo"]
    )
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Verb-specific projections
# ---------------------------------------------------------------------------


def test_messages_role_filter(trace_file: Path) -> None:
    """``--role assistant`` projects to that role only."""

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "messages",
            "--file",
            str(trace_file),
            "--role",
            "assistant",
            "--format",
            "ndjson",
        ],
    )
    assert result.exit_code == 0
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["payload"]["role"] == "assistant"


def test_tools_merges_args_and_result(trace_file: Path) -> None:
    """``tools`` joins span + args + result into one record (decoded)."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["tools", "--file", str(trace_file), "--format", "ndjson"]
    )
    assert result.exit_code == 0
    rec = json.loads(result.stdout.strip())
    assert rec["tool"] == "write"
    assert rec["args"] == {"path": "x.txt"}
    assert rec["result"] == {"ok": True}


def test_stats_counts_records(trace_file: Path) -> None:
    """``stats`` returns a histogram of every observed name."""

    runner = CliRunner()
    result = runner.invoke(
        app, ["stats", "--file", str(trace_file), "--format", "json"]
    )
    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["logs"]["agentm.message.appended"] == 2
    assert summary["spans"]["execute_tool write"] == 1


def test_info_missing_metadata_exits_3(tmp_path: Path) -> None:
    """A trace with no header AND no fingerprint returns exit 3."""

    path = tmp_path / "empty.jsonl"
    path.write_text(_log_line("other.event", {}) + "\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["info", "--file", str(path)])
    assert result.exit_code == 3


def test_system_prompt_surfaces_as_message_zero(tmp_path: Path) -> None:
    """When the loop persisted an ``agentm.llm.system_prompt`` log,
    ``messages`` prepends it as a synthetic role=system entry.
    """

    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                _log_line(
                    "agentm.llm.system_prompt",
                    {"turn_index": 0, "turn_id": 0, "text": "be helpful"},
                ),
                _log_line(
                    "agentm.message.appended",
                    _message("user", "hi"),
                    {"agentm.message.type": "message"},
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(path), "--format", "ndjson"]
    )
    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["payload"]["role"] == "system"
    assert first["payload"]["content"][0]["text"] == "be helpful"
    # ``--role assistant`` must still exclude it.
    result2 = runner.invoke(
        app,
        ["messages", "--file", str(path), "--role", "user", "--format", "ndjson"],
    )
    assert result2.exit_code == 0
    lines2 = [ln for ln in result2.stdout.splitlines() if ln.strip()]
    assert len(lines2) == 1
    assert json.loads(lines2[0])["payload"]["role"] == "user"


def test_no_system_prompt_log_means_no_synthetic_entry(trace_file: Path) -> None:
    """Default traces (no ``agentm.llm.system_prompt`` log) must NOT
    inject a synthetic system message — the env-gated record is opt-in.
    """

    runner = CliRunner()
    result = runner.invoke(
        app, ["messages", "--file", str(trace_file), "--format", "ndjson"]
    )
    assert result.exit_code == 0
    roles = [
        json.loads(ln)["payload"]["role"]
        for ln in result.stdout.splitlines()
        if ln.strip()
    ]
    assert "system" not in roles


def test_info_scoped_to_header_only(trace_file: Path) -> None:
    """--what header surfaces only the header body."""

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "info",
            "--file",
            str(trace_file),
            "--what",
            "header",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list) and len(parsed) == 1
    assert "header" in parsed[0]
    assert "fingerprint" not in parsed[0]
