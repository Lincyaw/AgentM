from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm_rca.scenario.rca import build_recipe


class _RecordingProvider:
    def __init__(self, scripted: list[AssistantMessage]) -> None:
        self._scripted = scripted
        self.calls: list[dict[str, Any]] = []

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
        del signal, thinking
        self.calls.append(
            {
                "messages": list(messages),
                "tool_names": [tool.name for tool in tools],
                "system": system,
                "model": model.id,
            }
        )
        index = len(self.calls) - 1
        return self._iter(self._scripted[index])

    async def _iter(self, message: AssistantMessage) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=message)


def _provider_module(provider: _RecordingProvider) -> str:
    import sys
    import types

    module_name = "tests.fake_rca_provider"
    module = types.ModuleType(module_name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-rca",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-rca-model",
                    provider="fake",
                    context_window=16000,
                    max_output_tokens=1000,
                ),
                name="fake-rca",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    return module_name


def _text_message(text: str, *, ts: float) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=ts,
        stop_reason="end_turn",
    )


@pytest.mark.asyncio
async def test_dynamic_context_injects_transient_current_state_message(
    observability_data_dir,
) -> None:
    provider = _RecordingProvider([_text_message("done", ts=1.0)])
    recipe = build_recipe(data_dir=str(observability_data_dir))

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(observability_data_dir),
            extensions=recipe,
            provider=(_provider_module(provider), {}),
        )
    )
    try:
        messages = await session.prompt("Investigate checkout latency")
    finally:
        await session.shutdown()

    first_call = provider.calls[0]
    current_state = first_call["messages"][0]
    assert current_state.role == "user"
    assert current_state.content[0].text.startswith("<current_state>\n")
    assert "(Investigation starting -- no data collected yet)" in current_state.content[0].text
    assert "<output_schema task_type=\"scout\" model=\"ScoutAnswer\">" in str(
        first_call["system"]
    )
    assert all(
        not (
            message.role == "user"
            and message.content[0].text.startswith("<current_state>\n")
        )
        for message in messages
    )


@pytest.mark.asyncio
async def test_rca_recipe_runs_fake_scout_round_end_to_end(observability_data_dir) -> None:
    from agentm.core.kernel import ToolCallBlock

    provider = _RecordingProvider(
        [
            AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id="call-1",
                        name="update_service_profile",
                        arguments={
                            "service_name": "payments",
                            "is_anomalous": True,
                            "anomaly_summary": "p99 latency spiked",
                            "data_sources_queried": ["metrics"],
                            "key_observation": "payments regressed before checkout timed out",
                            "source_agent_id": "scout-1",
                            "source_task_type": "scout",
                        },
                    )
                ],
                timestamp=1.0,
                stop_reason="tool_use",
            ),
            _text_message("Scout findings captured", ts=2.0),
        ]
    )
    recipe = build_recipe(data_dir=str(observability_data_dir), task_type="scout")

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(observability_data_dir),
            extensions=recipe,
            provider=(_provider_module(provider), {}),
        )
    )
    try:
        messages = await session.prompt("Scout the incident")
    finally:
        await session.shutdown()

    assert len(provider.calls) == 2
    assert "update_service_profile" in provider.calls[0]["tool_names"]
    assert "query_metrics_ohlc_abnormal" in provider.calls[0]["tool_names"]
    second_state = provider.calls[1]["messages"][0].content[0].text
    assert "**`payments`**" in second_state
    assert "ANOMALOUS: p99 latency spiked" in second_state
    assert messages[-1].role == "assistant"
    assert messages[-1].content[0].text == "Scout findings captured"
