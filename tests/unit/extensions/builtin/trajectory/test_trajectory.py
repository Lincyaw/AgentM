from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_trajectory_persists_one_record_per_fired_event(tmp_path: Path) -> None:
    output_path = tmp_path / "trajectory.jsonl"
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (
                    "agentm.extensions.builtin.trajectory",
                    {
                        "path": str(output_path),
                        "channels": [
                            "agent_start",
                            "turn_start",
                            "context",
                            "before_send_to_llm",
                            "turn_end",
                            "agent_end",
                        ],
                    },
                ),
            ],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    await session.prompt("hello")

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    channels = [record["channel"] for record in records]
    assert channels == [
        "agent_start",
        "turn_start",
        "context",
        "before_send_to_llm",
        "turn_end",
        "agent_end",
    ]
    assert len(records) == len(channels)
    await session.shutdown()
