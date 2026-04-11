"""Focused tests for post-hoc trajectory analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

from agentm.cli.debug import _event_detail, _print_summary, _print_timeline
from agentm.core.trajectory import read_trajectory


def _write_sample(path: Path) -> Path:
    events = [
        {
            "run_id": "r1",
            "seq": 1,
            "timestamp": "2026-03-08T10:00:01",
            "agent_path": ["orchestrator"],
            "node_name": "agent",
            "event_type": "tool_call",
            "data": {"tool_name": "spawn_worker", "args": {"task_type": "scout"}},
            "task_id": None,
            "metadata": {},
            "parent_seq": None,
        },
        {
            "run_id": "r1",
            "seq": 2,
            "timestamp": "2026-03-08T10:00:15",
            "agent_path": ["worker-scout"],
            "node_name": "agent",
            "event_type": "task_complete",
            "data": {"agent_id": "worker-scout", "duration_seconds": 12.3},
            "task_id": "t-1",
            "metadata": {},
            "parent_seq": None,
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def test_read_trajectory_parses_valid_lines_and_skips_invalid(tmp_path: Path) -> None:
    path = _write_sample(tmp_path / "ok.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write("not-json\n")
    _meta, events = read_trajectory(path)
    assert len(events) == 2


def test_event_detail_formats_tool_and_completion_events() -> None:
    detail_tool = _event_detail({"event_type": "tool_call", "data": {"tool_name": "spawn_worker", "args": {"x": 1}}})
    detail_done = _event_detail({"event_type": "task_complete", "data": {"agent_id": "w", "duration_seconds": 3.2}})
    assert "spawn_worker" in detail_tool
    assert "w" in detail_done


def test_print_summary_and_timeline_do_not_crash_on_valid_events(tmp_path: Path) -> None:
    _meta, events = read_trajectory(_write_sample(tmp_path / "events.jsonl"))
    _print_summary(events)
    _print_timeline(events)
