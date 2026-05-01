from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.kernel import AssistantMessage, AssistantStreamEvent, MessageEnd, Model, TextContent
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm_rca.trajectory_judger.data import TrajectoryLabel
from agentm_rca.trajectory_judger.recipe import load


class _SingleReplyProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        return self._iter()

    async def _iter(self) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=self.reply)],
                timestamp=1.0,
                stop_reason="end_turn",
            )
        )


def _install_provider_module(name: str, reply: str) -> str:
    module = types.ModuleType(name)
    provider = _SingleReplyProvider(reply)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "trajectory-judger-test",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="trajectory-judger-test",
                    provider="fake",
                    context_window=8000,
                    max_output_tokens=2000,
                ),
                name="trajectory-judger-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


@pytest.mark.asyncio
async def test_recipe_loads_and_persists_structured_output() -> None:
    reply = json.dumps(
        {
            "trajectory_id": "traj-7",
            "case_id": "case-7",
            "agent_conclusion": ["checkout"],
            "ground_truth": ["checkout"],
            "is_correct": True,
            "is_partial": False,
            "category": "success",
            "sub_type": None,
            "reasoning": "The agent inspected the trajectory, found the service that matched the ground truth, and maintained a consistent evidence chain across the recorded steps so the classification is a full success.",
            "evidence": [{"step": 2, "description": "Query matched checkout", "relevance": "Shows direct confirmation"}],
            "key_steps": {"conclusion_step": 4},
            "stats": {"total_steps": 4, "total_tool_calls": 1, "unique_services_queried": 1, "root_cause_first_mentioned_step": 2},
            "analyzer_version": "1.0.0",
        }
    )
    provider_module = _install_provider_module(
        "scenarios.rca.tests.trajectory_judger._fake_provider",
        reply,
    )

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=".",
            extensions=load(),
            provider=(provider_module, {}),
        )
    )

    try:
        await session.prompt("classify this trajectory")
        branch = session.session_manager.get_active_branch()
        label_entry = next(entry for entry in branch if entry.type == "trajectory_label")

        assert isinstance(label_entry.payload, TrajectoryLabel)
        assert label_entry.payload.category == "success"
        assert any(tool.name == "jq_query" for tool in session.tools)
    finally:
        await session.shutdown()
