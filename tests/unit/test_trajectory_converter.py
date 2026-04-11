from __future__ import annotations

import json

from agentm.core.trajectory_converter import build_trajectory_from_events


def test_build_trajectory_from_events_separates_same_worker_name_by_task_id() -> None:
    events = [
        {
            "seq": 1,
            "event_type": "llm_start",
            "agent_path": ["orchestrator"],
            "task_id": None,
            "data": {
                "messages": [
                    {"type": "system", "content": "sys"},
                    {"type": "human", "content": "start"},
                ]
            },
        },
        {
            "seq": 2,
            "event_type": "tool_call",
            "agent_path": ["orchestrator"],
            "task_id": None,
            "data": {
                "tool_name": "dispatch_agent",
                "args": {"agent_id": "scout"},
                "tool_call_id": "orch-tc-1",
            },
        },
        {
            "seq": 3,
            "event_type": "tool_result",
            "agent_path": ["orchestrator"],
            "task_id": None,
            "data": {
                "tool_name": "dispatch_agent",
                "result": '{"task_id":"scout-1","status":"running"}',
                "tool_call_id": "orch-tc-1",
            },
        },
        {
            "seq": 4,
            "event_type": "llm_start",
            "agent_path": ["orchestrator", "scout"],
            "task_id": "scout-1",
            "data": {
                "messages": [
                    {"type": "system", "content": "worker sys"},
                    {"type": "human", "content": "first run"},
                ]
            },
        },
        {
            "seq": 5,
            "event_type": "llm_end",
            "agent_path": ["orchestrator", "scout"],
            "task_id": "scout-1",
            "data": {"content": "done first"},
        },
        {
            "seq": 6,
            "event_type": "llm_start",
            "agent_path": ["orchestrator", "scout"],
            "task_id": "scout-2",
            "data": {
                "messages": [
                    {"type": "system", "content": "worker sys"},
                    {"type": "human", "content": "second run"},
                ]
            },
        },
        {
            "seq": 7,
            "event_type": "llm_end",
            "agent_path": ["orchestrator", "scout"],
            "task_id": "scout-2",
            "data": {"content": "done second"},
        },
    ]

    data = json.loads(build_trajectory_from_events("run-1", events))
    trajectories = data["trajectories"]

    assert trajectories[0]["agent_name"] == "agentm-orchestrator"
    workers = [t for t in trajectories if t["agent_name"] == "scout"]
    assert len(workers) == 2
    assert {t["sub_agent_call_id"] for t in workers} == {"scout-1", "scout-2"}
    assert workers[0]["messages"][-1]["content"] == "done first"
    assert workers[1]["messages"][-1]["content"] == "done second"
