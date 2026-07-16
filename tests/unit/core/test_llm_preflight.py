"""Fail-stop coverage for the final mutable LLM request boundary."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import (
    AgentLoop,
    AssistantMessage,
    BeforeSendToLlmEvent,
    DiagnosticEvent,
    EndTurn,
    EventBus,
    FunctionTool,
    LlmRequestStartEvent,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
    ToolResult,
    text_message,
)


class _CapturingStream:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    async def __call__(self, **request: Any) -> AsyncIterator[MessageEnd]:
        self.request = {**request, "messages": list(request["messages"])}
        yield MessageEnd(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                timestamp=1.0,
                stop_reason="end_turn",
                termination=EndTurn(),
            )
        )


@pytest.mark.asyncio
async def test_preflight_mutations_are_consistent_at_every_llm_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider, debug dump, and telemetry must observe one finalized request."""

    monkeypatch.setenv("AGENTM_LLM_PROMPT_DUMP", "1")
    bus = EventBus()
    stream = _CapturingStream()
    original = text_message("original")
    final = text_message("mutated")
    final_model = Model(
        id="mutated-model",
        provider="test",
        context_window=8192,
        max_output_tokens=1024,
    )

    async def execute(_args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="ok")])

    final_tool = FunctionTool(
        name="mutated_tool",
        description="mutated",
        parameters={"type": "object", "properties": {}},
        fn=execute,
    )
    request_start: LlmRequestStartEvent | None = None
    prompt_dump: dict[str, Any] | None = None

    def mutate_preflight(event: BeforeSendToLlmEvent) -> None:
        event.messages[:] = [final]
        event.model = final_model
        event.tools = [final_tool]
        event.system = "final-system"

    def capture_start(event: LlmRequestStartEvent) -> None:
        nonlocal request_start
        request_start = event

    def capture_diagnostic(event: DiagnosticEvent) -> None:
        nonlocal prompt_dump
        if event.source == "llm_prompt_dump":
            prompt_dump = json.loads(event.message)

    bus.on(BeforeSendToLlmEvent.CHANNEL, mutate_preflight)
    bus.on(LlmRequestStartEvent.CHANNEL, capture_start)
    bus.on(DiagnosticEvent.CHANNEL, capture_diagnostic)

    loop = AgentLoop(stream_fn=stream, bus=bus)
    await loop.run(
        messages=[original],
        model=Model(
            id="test-model",
            provider="test",
            context_window=4096,
            max_output_tokens=512,
        ),
        tools=[],
        system="original-system",
    )

    assert stream.request is not None
    assert stream.request["messages"] == [final]
    assert stream.request["model"] is final_model
    assert stream.request["tools"] == [final_tool]
    assert stream.request["system"] == "final-system"

    assert request_start is not None
    assert request_start.message_count == 1
    assert request_start.tool_count == 1
    assert request_start.model_id == "mutated-model"
    assert request_start.system_chars == len("final-system")
    assert request_start.system_text == "final-system"

    assert prompt_dump is not None
    assert prompt_dump["system"] == "final-system"
    assert prompt_dump["messages"][0]["content"][0]["text"] == "mutated"


@pytest.mark.asyncio
async def test_preflight_tool_replacement_updates_execution_index() -> None:
    bus = EventBus()
    executed: list[str] = []
    stream_calls = 0

    async def execute_replacement(_args: dict[str, Any]) -> ToolResult:
        executed.append("replacement")
        return ToolResult(content=[TextContent(type="text", text="ok")])

    replacement = FunctionTool(
        name="replacement",
        description="replacement tool",
        parameters={"type": "object", "properties": {}},
        fn=execute_replacement,
    )

    async def stream(**_request: Any) -> AsyncIterator[MessageEnd]:
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            yield MessageEnd(
                AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            type="tool_call",
                            id="call-1",
                            name="replacement",
                            arguments={},
                        )
                    ],
                    timestamp=1.0,
                    stop_reason="tool_use",
                )
            )
            return
        yield MessageEnd(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                timestamp=2.0,
                stop_reason="end_turn",
                termination=EndTurn(),
            )
        )

    def replace_tools(event: BeforeSendToLlmEvent) -> None:
        event.tools = [replacement]

    bus.on(BeforeSendToLlmEvent.CHANNEL, replace_tools)
    loop = AgentLoop(stream_fn=stream, bus=bus)

    await loop.run(
        messages=[text_message("run")],
        model=Model(
            id="test-model",
            provider="test",
            context_window=4096,
            max_output_tokens=512,
        ),
        tools=[],
    )

    assert executed == ["replacement"]
