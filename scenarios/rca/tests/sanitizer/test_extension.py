from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    AssistantStreamEvent,
    ContextEvent,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm_rca.recipe import build_recipe
from agentm_rca.sanitizer.code_sanitizer import CodeSanitizer
from agentm_rca.sanitizer.extension import _SanitizerExtension
from agentm_rca.stores import HypothesisStore, ServiceProfileStore


def _extension(
    *,
    hypothesis_store: HypothesisStore | None = None,
    profile_store: ServiceProfileStore | None = None,
    code_sanitizer: CodeSanitizer | None = None,
) -> _SanitizerExtension:
    return _SanitizerExtension(
        hypothesis_store=hypothesis_store or HypothesisStore(),
        profile_store=profile_store or ServiceProfileStore(),
        code_sanitizer=code_sanitizer or CodeSanitizer(),
        critic_sanitizer=None,
        periodic_interval=5,
        max_block_retries=3,
        tool_call_budget=None,
        max_steps=30,
    )


def _assistant_text_message(text: str, *, ts: float = 1.0) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=ts,
        stop_reason="end_turn",
    )


@pytest.mark.asyncio
async def test_extension_injects_exact_sanitizer_report_for_profile_write_without_query() -> None:
    extension = _extension()
    event = ContextEvent(messages=[])

    extension.on_turn_start(TurnStartEvent(turn_index=0))
    extension.on_tool_call(
        ToolCallEvent(
            tool_call_id="call-1",
            tool_name="update_service_profile",
            args={"service_name": "payments"},
        )
    )
    extension.on_tool_result(
        ToolResultEvent(
            tool_call_id="call-1",
            tool_name="update_service_profile",
            result=ToolResult(content=[TextContent(type="text", text="ok")]),
        )
    )
    extension.on_turn_end(TurnEndEvent(turn_index=0, message=_assistant_text_message("thinking")))

    await extension.on_context(event)

    assert len(event.messages) == 1
    block = event.messages[0].content[0]
    assert isinstance(block, TextContent)
    assert block.text == (
        "<sanitizer_report>\n"
        '<finding code="P3" severity="INFO">\n'
        "Profile update for 'payments' without prior query\n"
        "</finding>\n"
        "</sanitizer_report>"
    )


@pytest.mark.asyncio
async def test_extension_injects_exact_finalize_blocked_message_for_c1() -> None:
    hypothesis_store = HypothesisStore()
    hypothesis_store.update("H1", "Root cause", status="confirmed")
    extension = _extension(hypothesis_store=hypothesis_store)
    event = ContextEvent(messages=[])

    extension.on_turn_start(TurnStartEvent(turn_index=0))
    extension.on_turn_end(
        TurnEndEvent(
            turn_index=0,
            message=_assistant_text_message("Done. <decision>finalize</decision>"),
        )
    )

    await extension.on_context(event)

    text = event.messages[0].content[0]
    assert isinstance(text, TextContent)
    assert text.text == (
        '<finalize_blocked reason="1 BLOCK findings">\n'
        "You attempted to finalize but the following conditions are not met:\n"
        "1. [C1] Hypothesis 'H1' confirmed without verify task\n"
        "Address these before attempting to finalize again.\n"
        "</finalize_blocked>\n"
        "<sanitizer_report>\n"
        '<finding code="C4" severity="WARN">\n'
        "Root cause confirmed with no alternative hypotheses explored\n"
        "</finding>\n"
        "</sanitizer_report>"
    )


class _TwoTurnProvider:
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        return self._iter(self.calls)

    async def _iter(self, call_number: int) -> AsyncIterator[AssistantStreamEvent]:
        if call_number == 1:
            yield MessageEnd(
                message=AssistantMessage(
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
                )
            )
            return

        yield MessageEnd(message=_assistant_text_message("done", ts=2.0))


def _provider_module(provider: _TwoTurnProvider) -> str:
    module_name = "tests.fake_rca_sanitizer_provider"
    module = types.ModuleType(module_name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-rca-sanitizer",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-rca-sanitizer-model",
                    provider="fake",
                    context_window=16000,
                    max_output_tokens=1000,
                ),
                name="fake-rca-sanitizer",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    return module_name


@pytest.mark.asyncio
async def test_recipe_trajectory_contains_sanitizer_report(observability_data_dir: Path) -> None:
    provider = _TwoTurnProvider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(observability_data_dir),
            extensions=build_recipe(data_dir=str(observability_data_dir)),
            provider=(_provider_module(provider), {}),
        )
    )

    try:
        await session.prompt("Investigate checkout latency")
    finally:
        await session.shutdown()

    trajectory_path = observability_data_dir / "rca_trajectory.jsonl"
    records = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    sanitizer_contexts = [
        record
        for record in records
        if record["channel"] == "context"
        and "sanitizer_report" in json.dumps(record["event"])
    ]

    assert sanitizer_contexts
    payload = json.dumps(sanitizer_contexts[-1]["event"])
    assert "P3" in payload
    assert "payments" in payload
