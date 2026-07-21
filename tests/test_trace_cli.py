from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentm.cli import _trace as trace_cli
from agentm.cli._app import app
from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.store import SessionMeta, TrajectoryDiagnostic
from agentm.core.abi.termination import SignalAborted
from agentm.core.abi.trajectory import (
    Outcome,
    ToolRecord,
    TrajectoryHead,
    Turn,
    TurnCheckpoint,
)
from agentm.core.abi.trigger import UserInput
from agentm.storage.trajectory import JsonlTrajectoryStore


@pytest.fixture
def trace_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, JsonlTrajectoryStore]:
    root = tmp_path / "trajectory"
    monkeypatch.setenv("AGENTM_TRAJECTORY_DIR", str(root))
    monkeypatch.delenv("AGENTM_TRAJECTORY_DSN", raising=False)
    monkeypatch.delenv("AGENTM_TRAJECTORY_SCHEMA", raising=False)
    store = JsonlTrajectoryStore(root)
    session_id = "active-session"
    call = ToolCallBlock(
        type="tool_call",
        id="read-1",
        name="read",
        arguments={"path": "src/example.py"},
    )
    result = ToolResultBlock(
        type="tool_result",
        tool_call_id=call.id,
        content=(TextContent(type="text", text="abcdefghij"),),
    )
    turn = Turn(
        index=0,
        id="turn-0",
        run_id="run-0",
        run_step=0,
        trigger=UserInput(content=(TextContent(type="text", text="inspect"),)),
        response=AssistantMessage(
            role="assistant",
            content=(call,),
            timestamp=1.0,
            stop_reason="tool_use",
        ),
        tool_results=(ToolRecord(call=call, result=result),),
        outcome=Outcome(cause=SignalAborted(reason="submit_interrupt")),
        timestamp=2.0,
    )
    store.create_session(
        SessionMeta(id=session_id, created_at=10.0),
        turns=(turn,),
        head=TrajectoryHead(session_id=session_id),
    )
    store.save_checkpoint(
        session_id,
        TurnCheckpoint(
            index=1,
            id="turn-1",
            run_id="run-active",
            run_step=0,
            trigger=UserInput(content=(TextContent(type="text", text="continue"),)),
            response=None,
            tool_results=(),
            updated_at=3.0,
        ),
    )
    store.append_diagnostic(
        TrajectoryDiagnostic(
            id="diagnostic-1",
            session_id=session_id,
            timestamp=4.0,
            level="error",
            source="driver",
            phase="trajectory_commit",
            message="turn abandoned",
            error_type="UniqueViolation",
            error_detail="duplicate key",
            turn_id="turn-1",
            turn_index=1,
            checkpoint_id="turn-1",
        )
    )
    store.create_session(
        SessionMeta(id="idle-session", created_at=20.0),
        head=TrajectoryHead(session_id="idle-session"),
    )
    return session_id, store


def _ndjson(output: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def test_trace_status_wait_and_session_filters(
    trace_store: tuple[str, JsonlTrajectoryStore],
) -> None:
    session_id, _ = trace_store
    runner = CliRunner()

    status = runner.invoke(
        app,
        ["trace", "status", "-s", session_id, "--format", "ndjson"],
    )
    assert status.exit_code == 0, status.output
    assert _ndjson(status.stdout) == [
        {
            "session_id": session_id,
            "state": "active",
            "committed_turns": 1,
            "incomplete_turns": 1,
            "active_checkpoint": True,
            "last_activity_at": 4.0,
            "last_turn_index": 0,
            "last_turn_id": "turn-0",
            "last_turn_cause": "SignalAborted",
            "checkpoint_turn_index": 1,
            "checkpoint_turn_id": "turn-1",
            "checkpoint_run_id": "run-active",
            "checkpoint_run_step": 0,
            "checkpoint_updated_at": 3.0,
            "diagnostic_count": 1,
            "last_diagnostic_id": "diagnostic-1",
        }
    ]

    waited = runner.invoke(
        app,
        [
            "trace",
            "wait",
            "-s",
            session_id,
            "--min-committed-turns",
            "1",
            "--require-active-checkpoint",
            "--timeout",
            "0",
            "--format",
            "ndjson",
        ],
    )
    assert waited.exit_code == 0, waited.output

    timed_out = runner.invoke(
        app,
        [
            "trace",
            "wait",
            "-s",
            session_id,
            "--min-committed-turns",
            "2",
            "--timeout",
            "0",
            "--format",
            "ndjson",
        ],
    )
    assert timed_out.exit_code == 8
    assert _ndjson(timed_out.stdout)[0]["committed_turns"] == 1

    sessions = runner.invoke(
        app,
        ["trace", "sessions", "--active", "--latest", "--format", "ndjson"],
    )
    assert sessions.exit_code == 0, sessions.output
    assert [row["id"] for row in _ndjson(sessions.stdout)] == [session_id]


def test_trace_watch_diagnostics_turn_and_tool_filters(
    trace_store: tuple[str, JsonlTrajectoryStore],
) -> None:
    session_id, _ = trace_store
    runner = CliRunner()

    watched = runner.invoke(
        app,
        [
            "trace",
            "watch",
            "-s",
            session_id,
            "--include-existing",
            "--limit",
            "3",
            "--format",
            "ndjson",
        ],
    )
    assert watched.exit_code == 0, watched.output
    watch_rows = _ndjson(watched.stdout)
    assert [row["type"] for row in watch_rows] == [
        "abort",
        "checkpoint",
        "diagnostic",
    ]
    assert len({row["event_id"] for row in watch_rows}) == 3

    diagnostics = runner.invoke(
        app,
        [
            "trace",
            "diagnostics",
            "-s",
            session_id,
            "--phase",
            "trajectory_commit",
            "--format",
            "ndjson",
        ],
    )
    assert diagnostics.exit_code == 0, diagnostics.output
    assert _ndjson(diagnostics.stdout)[0]["error_type"] == "UniqueViolation"

    turns = runner.invoke(
        app,
        [
            "trace",
            "turns",
            "-s",
            session_id,
            "--status",
            "incomplete",
            "--run-id",
            "run-active",
            "--from-turn",
            "1",
            "--format",
            "ndjson",
        ],
    )
    assert turns.exit_code == 0, turns.output
    assert [row["turn_id"] for row in _ndjson(turns.stdout)] == ["turn-1"]

    tools = runner.invoke(
        app,
        [
            "trace",
            "tools",
            "-s",
            session_id,
            "--result-chars",
            "4",
            "--format",
            "ndjson",
        ],
    )
    assert tools.exit_code == 0, tools.output
    assert _ndjson(tools.stdout) == [
        {
            "turn_index": 0,
            "run_id": "run-0",
            "run_step": 0,
            "tool": "read",
            "args": {"path": "src/example.py"},
            "is_error": False,
            "result": "abcd",
            "result_chars": 10,
            "result_truncated": True,
        }
    ]


def test_trace_watch_emits_only_new_deltas(
    trace_store: tuple[str, JsonlTrajectoryStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, store = trace_store
    runner = CliRunner()
    initialized = threading.Event()
    original_watch_events = trace_cli._watch_events

    def observed_watch_events(
        query: trace_cli.TrajectoryQueryStore,
        observed_session_id: str,
    ) -> list[trace_cli._WatchEvent]:
        events = original_watch_events(query, observed_session_id)
        initialized.set()
        return events

    monkeypatch.setattr(trace_cli, "_watch_events", observed_watch_events)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            runner.invoke,
            app,
            [
                "trace",
                "watch",
                "-s",
                session_id,
                "--limit",
                "1",
                "--timeout",
                "2",
                "--poll-interval",
                "0.01",
                "--format",
                "ndjson",
            ],
        )
        assert initialized.wait(timeout=1.0)
        store.append_diagnostic(
            TrajectoryDiagnostic(
                id="diagnostic-2",
                session_id=session_id,
                timestamp=5.0,
                level="warning",
                source="test",
                phase="reaction",
                message="new delta",
            )
        )
        watched = pending.result(timeout=3.0)

    assert watched.exit_code == 0, watched.output
    rows = _ndjson(watched.stdout)
    assert len(rows) == 1
    assert rows[0]["type"] == "diagnostic"
    assert rows[0]["diagnostic"]["id"] == "diagnostic-2"


def test_session_interrupt_reports_missing_target() -> None:
    result = CliRunner().invoke(
        app,
        ["session", "interrupt", "missing", "--message", "hello"],
    )

    assert result.exit_code == 3
    assert json.loads(result.stderr) == {
        "session_id": "missing",
        "status": "error",
        "error_type": "session_not_found",
        "error_detail": "no active session missing (socket not found)",
    }
