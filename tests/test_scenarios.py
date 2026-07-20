from __future__ import annotations

from pathlib import Path

import pytest

from agentm import AgentSession, AgentSessionConfig, Model
from agentm.core.abi.stream import MessageEnd
from agentm.scenarios import load_scenario_manifest


async def _stub_stream(**kwargs: object) -> object:
    del kwargs
    yield MessageEnd(stop_reason="end_turn", usage=None)


def test_harbor_manifest_includes_base_extensions() -> None:
    spec = load_scenario_manifest(
        Path("contrib/scenarios/harbor/scenario.yaml"),
        requested_name="arl:harbor",
    )

    assert [extension.module_path for extension in spec.extensions] == [
        "agentm.extensions.builtin.observability",
        "agentm.extensions.builtin.operations",
        "agentm.extensions.builtin.local_resources",
        "agentm.extensions.builtin.loop_budget",
        "agentm.extensions.builtin.context_projection",
        "agentm.extensions.builtin.runtime_context",
        "agentm.extensions.builtin.system_prompt",
        "agentm.extensions.builtin.skill_loader",
        "agentm.extensions.builtin.memory",
        "agentm.extensions.builtin.prompt_cache",
        "agentm.extensions.builtin.llm_compaction",
        "agentm.extensions.builtin.message_patterns",
        "agentm.extensions.builtin.tool_index",
        "agentm.extensions.builtin.turn_reminder",
        "agentm.extensions.builtin.file_tools",
        "agentm.extensions.builtin.tool_bash",
        "agentm.extensions.builtin.tool_result_cap",
        "agentm.extensions.builtin.tool_error_messages",
        "agentm.extensions.builtin.structured_output",
        "agentm.extensions.builtin.read_history",
        "agentm.extensions.builtin.sub_agent",
        "agentm.extensions.builtin.background_exec",
        "agentm.extensions.builtin.workflow",
        "agentm.extensions.builtin.task_tracking",
        "agentm.extensions.builtin.trace_query",
        "agentm.extensions.builtin.retry_policy",
        "agentm.extensions.builtin.tool_bash_guard",
        "agentm.extensions.builtin.tool_filter",
        "agentm.extensions.builtin.permission",
        "agentm.extensions.builtin.thinking_retry",
        "agentm.extensions.builtin.goal",
        "policy_engine",
    ]
    assert spec.extensions[15].config["default_timeout"] == 6000
    assert spec.extensions[16].config["max_tokens"] == 1000
    assert spec.extensions[19].config["tool_result_max_tokens"] == 200
    assert spec.extensions[19].config["total_max_tokens"] == 1000


@pytest.mark.asyncio
async def test_harbor_all_atom_base_creates_session(tmp_path: Path) -> None:
    spec = load_scenario_manifest(
        Path("contrib/scenarios/harbor/scenario.yaml"),
        requested_name="arl:harbor",
    )

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=list(spec.extensions),
            stream_fn=_stub_stream,
            model=Model(
                id="stub",
                provider="stub",
                context_window=100_000,
                max_output_tokens=4096,
            ),
        )
    )
    await session.shutdown()
