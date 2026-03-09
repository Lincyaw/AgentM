"""Unit tests for post-hoc trajectory analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.cli.debug import _load_events, _print_summary, _print_timeline, _event_detail


@pytest.fixture
def sample_trajectory(tmp_path: Path) -> Path:
    """Create a sample trajectory JSONL file."""
    events = [
        {"run_id": "r1", "seq": 1, "timestamp": "2026-03-08T10:00:01", "agent_path": ["orchestrator"], "node_name": "agent", "event_type": "llm_end", "data": {"content": "Let me investigate"}, "task_id": None, "hypothesis_id": None, "parent_seq": None},
        {"run_id": "r1", "seq": 2, "timestamp": "2026-03-08T10:00:02", "agent_path": ["orchestrator"], "node_name": "agent", "event_type": "tool_call", "data": {"tool_name": "spawn_worker", "args": {"task_type": "scout"}}, "task_id": None, "hypothesis_id": None, "parent_seq": None},
        {"run_id": "r1", "seq": 3, "timestamp": "2026-03-08T10:00:03", "agent_path": ["worker-scout"], "node_name": "", "event_type": "task_dispatch", "data": {"task_id": "t-1", "agent_id": "worker-scout", "task_type": "scout"}, "task_id": "t-1", "hypothesis_id": None, "parent_seq": None},
        {"run_id": "r1", "seq": 4, "timestamp": "2026-03-08T10:00:15", "agent_path": ["worker-scout"], "node_name": "", "event_type": "task_complete", "data": {"task_id": "t-1", "agent_id": "worker-scout", "duration_seconds": 12.3}, "task_id": "t-1", "hypothesis_id": None, "parent_seq": None},
        {"run_id": "r1", "seq": 5, "timestamp": "2026-03-08T10:00:16", "agent_path": ["orchestrator"], "node_name": "agent", "event_type": "hypothesis_update", "data": {"hypothesis_id": "H1", "status": "formed", "description": "DB connection pool exhaustion"}, "task_id": None, "hypothesis_id": "H1", "parent_seq": None},
        {"run_id": "r1", "seq": 6, "timestamp": "2026-03-08T10:00:30", "agent_path": ["orchestrator"], "node_name": "agent", "event_type": "hypothesis_update", "data": {"hypothesis_id": "H1", "status": "confirmed", "description": "DB connection pool exhaustion"}, "task_id": None, "hypothesis_id": "H1", "parent_seq": None},
    ]
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def test_load_events(sample_trajectory: Path) -> None:
    """_load_events must parse all valid lines."""
    events = _load_events(sample_trajectory)
    assert len(events) == 6


def test_load_events_skips_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON lines must be skipped, not crash the loader."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"valid": true}\nnot json\n{"also": "valid"}\n')
    events = _load_events(path)
    assert len(events) == 2


def test_filter_by_agent_path(sample_trajectory: Path) -> None:
    """agent path filter must match prefix."""
    events = _load_events(sample_trajectory)
    filtered = [e for e in events if "worker-scout" in "/".join(e.get("agent_path", []))]
    assert len(filtered) == 2
    assert all("worker-scout" in "/".join(e["agent_path"]) for e in filtered)


def test_filter_by_event_type(sample_trajectory: Path) -> None:
    """event_type filter must match exactly."""
    events = _load_events(sample_trajectory)
    filtered = [e for e in events if e.get("event_type") == "hypothesis_update"]
    assert len(filtered) == 2


def test_event_detail_tool_call() -> None:
    """tool_call detail must include tool name and truncated args."""
    event = {"event_type": "tool_call", "data": {"tool_name": "spawn_worker", "args": {"task_type": "scout"}}}
    detail = _event_detail(event)
    assert "spawn_worker" in detail


def test_event_detail_hypothesis_update() -> None:
    """hypothesis_update detail must include id and status transition."""
    event = {"event_type": "hypothesis_update", "data": {"hypothesis_id": "H1", "status": "confirmed"}}
    detail = _event_detail(event)
    assert "H1" in detail
    assert "confirmed" in detail


def test_event_detail_task_complete() -> None:
    """task_complete detail must include agent and duration."""
    event = {"event_type": "task_complete", "data": {"agent_id": "worker-scout", "duration_seconds": 12.3}}
    detail = _event_detail(event)
    assert "worker-scout" in detail
    assert "12.3" in detail


def test_print_summary_does_not_crash(sample_trajectory: Path, capsys) -> None:
    """_print_summary must run without exceptions on valid events."""
    events = _load_events(sample_trajectory)
    _print_summary(events)  # Should not raise


def test_print_timeline_does_not_crash(sample_trajectory: Path, capsys) -> None:
    """_print_timeline must run without exceptions on valid events."""
    events = _load_events(sample_trajectory)
    _print_timeline(events)  # Should not raise
