from __future__ import annotations

import json
from pathlib import Path

from agentm.cli.judge_runner import SkeletonConfig, _extract_skeleton_from_jsonl


def test_extract_skeleton_from_jsonl_matches_by_tool_call_id(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    events = [
        {
            "_meta": {
                "run_id": "run-1",
                "thread_id": "",
                "checkpoint_db": "",
            }
        },
        {
            "seq": 1,
            "event_type": "tool_call",
            "agent_path": ["orchestrator"],
            "data": {
                "tool_name": "query_logs",
                "args": {"q": "a"},
                "tool_call_id": "tc-1",
            },
        },
        {
            "seq": 2,
            "event_type": "tool_call",
            "agent_path": ["orchestrator"],
            "data": {
                "tool_name": "query_logs",
                "args": {"q": "b"},
                "tool_call_id": "tc-2",
            },
        },
        {
            "seq": 3,
            "event_type": "tool_result",
            "agent_path": ["orchestrator"],
            "data": {
                "tool_name": "query_logs",
                "result": "result-b",
                "tool_call_id": "tc-2",
            },
        },
        {
            "seq": 4,
            "event_type": "tool_result",
            "agent_path": ["orchestrator"],
            "data": {
                "tool_name": "query_logs",
                "result": "result-a",
                "tool_call_id": "tc-1",
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    steps = _extract_skeleton_from_jsonl(path, SkeletonConfig())

    assert [step["response_preview"] for step in steps] == ["result-a", "result-b"]
    assert steps[0]["response_chars"] == len("result-a")


def test_extract_skeleton_from_jsonl_falls_back_to_result_without_tool_call_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traj.jsonl"
    events = [
        {"_meta": {"run_id": "run-1", "thread_id": "", "checkpoint_db": ""}},
        {
            "seq": 1,
            "event_type": "tool_call",
            "agent_path": ["orchestrator"],
            "data": {"tool_name": "query_logs", "args": {"q": "legacy"}},
        },
        {
            "seq": 2,
            "event_type": "tool_result",
            "agent_path": ["orchestrator"],
            "data": {"tool_name": "query_logs", "result": "legacy-result"},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    steps = _extract_skeleton_from_jsonl(path, SkeletonConfig())

    assert len(steps) == 1
    assert steps[0]["response_preview"] == "legacy-result"
