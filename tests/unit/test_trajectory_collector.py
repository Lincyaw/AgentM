"""Focused tests for TrajectoryCollector sequencing, persistence, and model roundtrip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.core.trajectory import TrajectoryCollector, TrajectoryEvent


@pytest.fixture
def tmp_output(tmp_path: Path) -> str:
    return str(tmp_path / "trajectories")


@pytest.mark.asyncio
async def test_record_and_record_sync_assign_monotonic_sequence_numbers(tmp_output: str) -> None:
    collector = TrajectoryCollector(run_id="test-mono", output_dir=tmp_output)

    assert await collector.record("tool_call", ["orchestrator"], {"tool": "a"}) == 1
    assert collector.record_sync("hypothesis_update", ["orchestrator"], {"id": "H1"}) == 2
    assert await collector.record("llm_end", ["orchestrator"], {"content": "c"}) == 3

    await collector.close()
    assert len(collector.events) == 3


@pytest.mark.asyncio
async def test_close_returns_none_without_events_and_path_with_events(tmp_output: str) -> None:
    empty = TrajectoryCollector(run_id="empty", output_dir=tmp_output)
    assert await empty.close() is None

    used = TrajectoryCollector(run_id="used", output_dir=tmp_output)
    await used.record("llm_end", ["orchestrator"], {"content": "x"})
    path = await used.close()

    assert path is not None
    assert path.endswith("used.jsonl")


@pytest.mark.asyncio
async def test_jsonl_persistence_matches_in_memory_events(tmp_output: str) -> None:
    collector = TrajectoryCollector(run_id="test-buffer", output_dir=tmp_output)
    await collector.record("tool_call", ["orchestrator"], {"tool": "a"})
    await collector.record(
        "error",
        ["worker-scout"],
        {"message": "timeout"},
        task_id="t-1",
        metadata={"k": "v"},
        parent_seq=1,
    )

    path = await collector.close()
    assert path is not None

    lines = [line for line in Path(path).read_text(encoding="utf-8").strip().split("\n") if line]
    file_events = [json.loads(line) for line in lines if "_meta" not in line]

    assert len(file_events) == len(collector.events) == 2
    assert file_events[1]["task_id"] == "t-1"
    assert file_events[1]["metadata"]["k"] == "v"
    assert file_events[1]["parent_seq"] == 1


def test_trajectory_event_model_roundtrip() -> None:
    event = TrajectoryEvent(
        run_id="r1",
        seq=1,
        timestamp="2026-03-08T10:00:00",
        agent_path=["orchestrator"],
        node_name="agent",
        event_type="tool_call",
        data={"tool": "spawn_worker", "args": {"type": "scout"}},
        task_id="t-1",
    )

    roundtripped = TrajectoryEvent.model_validate_json(event.model_dump_json())
    assert roundtripped.seq == 1
    assert roundtripped.data["tool"] == "spawn_worker"
    assert roundtripped.task_id == "t-1"
