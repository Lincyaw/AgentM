"""Unit tests for TrajectoryCollector."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentm.core.trajectory import TrajectoryCollector, TrajectoryEvent


@pytest.fixture
def tmp_output(tmp_path: Path) -> str:
    return str(tmp_path / "trajectories")


@pytest.mark.asyncio
async def test_record_assigns_monotonic_sequence(tmp_output: str) -> None:
    """Sequence numbers must be strictly increasing — out-of-order breaks timeline."""
    collector = TrajectoryCollector(run_id="test-mono", output_dir=tmp_output)
    s1 = await collector.record("tool_call", ["orchestrator"], {"tool": "a"})
    s2 = await collector.record("tool_result", ["orchestrator"], {"result": "b"})
    s3 = await collector.record("llm_end", ["orchestrator"], {"content": "c"})
    await collector.close()

    assert s1 == 1
    assert s2 == 2
    assert s3 == 3


@pytest.mark.asyncio
async def test_record_writes_valid_jsonl(tmp_output: str) -> None:
    """Every line in the JSONL file must be valid JSON — broken lines crash analysis."""
    collector = TrajectoryCollector(run_id="test-jsonl", output_dir=tmp_output)
    await collector.record("tool_call", ["orchestrator"], {"name": "spawn_worker"})
    await collector.record("task_dispatch", ["worker-scout"], {"task_id": "t1"}, task_id="t1")
    path = await collector.close()

    assert path is not None
    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "run_id" in parsed
        assert "seq" in parsed
        assert "event_type" in parsed


@pytest.mark.asyncio
async def test_close_returns_file_path(tmp_output: str) -> None:
    """close() must return path so caller can report it to user."""
    collector = TrajectoryCollector(run_id="test-path", output_dir=tmp_output)
    await collector.record("llm_end", ["orchestrator"], {"content": "x"})
    path = await collector.close()

    assert path is not None
    assert "test-path.jsonl" in path


@pytest.mark.asyncio
async def test_close_without_records_returns_none(tmp_output: str) -> None:
    """close() on unused collector must not crash."""
    collector = TrajectoryCollector(run_id="test-empty", output_dir=tmp_output)
    path = await collector.close()
    assert path is None


@pytest.mark.asyncio
async def test_events_buffer_matches_file(tmp_output: str) -> None:
    """In-memory events and file contents must match — divergence causes debug confusion."""
    collector = TrajectoryCollector(run_id="test-buffer", output_dir=tmp_output)
    await collector.record("tool_call", ["orchestrator"], {"tool": "a"})
    await collector.record("error", ["worker-scout"], {"message": "timeout"})
    path = await collector.close()

    assert path is not None
    file_events = [json.loads(line) for line in Path(path).read_text().strip().split("\n")]
    memory_events = collector.events

    assert len(file_events) == len(memory_events) == 2
    for fe, me in zip(file_events, memory_events):
        assert fe["seq"] == me["seq"]
        assert fe["event_type"] == me["event_type"]


@pytest.mark.asyncio
async def test_record_without_optional_fields(tmp_output: str) -> None:
    """Optional fields (task_id, hypothesis_id, parent_seq) default to None."""
    collector = TrajectoryCollector(run_id="test-opt", output_dir=tmp_output)
    await collector.record("llm_end", ["orchestrator"], {"content": "hi"})
    path = await collector.close()

    assert path is not None
    event = json.loads(Path(path).read_text().strip())
    assert event["task_id"] is None
    assert event["hypothesis_id"] is None
    assert event["parent_seq"] is None


@pytest.mark.asyncio
async def test_record_with_linkage_fields(tmp_output: str) -> None:
    """task_id, hypothesis_id, parent_seq must be preserved for trace reconstruction."""
    collector = TrajectoryCollector(run_id="test-link", output_dir=tmp_output)
    s1 = await collector.record(
        "tool_call", ["orchestrator"], {"tool": "spawn_worker"}, task_id="t-1"
    )
    await collector.record(
        "tool_result", ["orchestrator"], {"result": "ok"},
        task_id="t-1", hypothesis_id="H1", parent_seq=s1,
    )
    path = await collector.close()

    assert path is not None
    lines = Path(path).read_text().strip().split("\n")
    e2 = json.loads(lines[1])
    assert e2["task_id"] == "t-1"
    assert e2["hypothesis_id"] == "H1"
    assert e2["parent_seq"] == 1


@pytest.mark.asyncio
async def test_record_sync_variant(tmp_output: str) -> None:
    """record_sync writes events without async — needed for sync tool functions."""
    collector = TrajectoryCollector(run_id="test-sync", output_dir=tmp_output)
    s1 = collector.record_sync("hypothesis_update", ["orchestrator"], {"id": "H1"})
    s2 = collector.record_sync("hypothesis_update", ["orchestrator"], {"id": "H2"})
    await collector.close()

    assert s1 == 1
    assert s2 == 2
    assert len(collector.events) == 2


def test_trajectory_event_model_roundtrip() -> None:
    """TrajectoryEvent must serialize/deserialize cleanly — JSONL depends on this."""
    event = TrajectoryEvent(
        run_id="r1", seq=1, timestamp="2026-03-08T10:00:00",
        agent_path=["orchestrator"], node_name="agent",
        event_type="tool_call", data={"tool": "spawn_worker", "args": {"type": "scout"}},
        task_id="t-1",
    )
    json_str = event.model_dump_json()
    roundtripped = TrajectoryEvent.model_validate_json(json_str)
    assert roundtripped.seq == 1
    assert roundtripped.data["tool"] == "spawn_worker"
    assert roundtripped.task_id == "t-1"
