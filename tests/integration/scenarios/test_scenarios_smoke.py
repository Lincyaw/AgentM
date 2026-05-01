from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentm.extensions.loader import load_scenario
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario_name", "tool_name", "arguments", "expected_entry_types"),
    [
        ("general_purpose", "read", None, {"message"}),
        (
            "rca",
            "add_hypothesis",
            {"id": "H1", "description": "Disk is full"},
            {"message", "hypothesis"},
        ),
        ("trajectory_analysis", "load_trajectory", None, {"message"}),
        (
            "plan_mode",
            "submit_plan",
            {"plan": "1. Inspect\n2. Validate\n3. Report"},
            {"message", "plan"},
        ),
    ],
)
async def test_scenarios_smoke(
    tmp_path: Path,
    scenario_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
    expected_entry_types: set[str],
) -> None:
    read_path = tmp_path / "sample.txt"
    read_path.write_text("alpha\nbeta\n", encoding="utf-8")
    trajectory_path = tmp_path / "trajectory.jsonl"
    trajectory_path.write_text(
        json.dumps({"channel": "tool_call", "name": "read"}) + "\n"
        + json.dumps({"channel": "tool_result", "name": "read"})
        + "\n",
        encoding="utf-8",
    )

    scenario_arguments = arguments or _default_arguments(
        tool_name,
        read_path=read_path,
        trajectory_path=trajectory_path,
    )

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=load_scenario(scenario_name),
            provider="scripted-fake",
            provider_config={
                "tool_name": tool_name,
                "arguments": scenario_arguments,
                "final_text": f"{scenario_name} complete",
            },
            resource_loader=InMemoryResourceLoader(),
        )
    )

    try:
        assert any(tool.name == tool_name for tool in session.tools)

        final = await session.prompt(f"run {scenario_name}")
        assert final[-1].role == "assistant"

        branch = session.session_manager.get_active_branch()
        branch_types = {entry.type for entry in branch}
        assert expected_entry_types <= branch_types
    finally:
        await session.shutdown()


def _default_arguments(
    tool_name: str,
    *,
    read_path: Path,
    trajectory_path: Path,
) -> dict[str, Any]:
    if tool_name == "read":
        return {"path": str(read_path)}
    if tool_name == "load_trajectory":
        return {"path": str(trajectory_path)}
    raise AssertionError(f"missing default arguments for {tool_name!r}")
