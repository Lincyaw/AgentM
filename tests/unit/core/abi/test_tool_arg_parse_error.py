"""Fail-stop coverage for streamed tool-call argument parse errors.

Malformed provider JSON used to become ``arguments={}`` with no observable
signal, so policy atoms could not decide whether to retry, repair, or stop.
This locks down the LLM-stream boundary contract: providers keep the parsed
``ToolCallBlock`` invariant, but the loop emits a typed parse-error event on
both the raw stream channel and the dedicated bus channel.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import (
    AgentLoop,
    EventBus,
    LoopConfig,
    MessageEnd,
    Model,
    TextContent,
    ToolCallArgsParseError,
    ToolCallBlock,
    ToolCallEnd,
    ToolCallStart,
    text_message,
)
from agentm.core.abi.stream import AssistantStreamEvent
from agentm.llm._common import StreamAccumulator


class _MalformedJsonStream:
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
        acc = StreamAccumulator()
        acc.add_text(None, "calling")
        acc.add_tool_call("call-1", "demo", '{"unterminated"')
        yield ToolCallStart(id="call-1", name="demo")
        yield ToolCallEnd(id="call-1")
        message = acc.assemble(
            stop_reason="tool_calls",
            termination=None,
            usage=None,
            timestamp=123.0,
        )
        for parse_error in acc.parse_errors:
            yield parse_error
        yield MessageEnd(message=message)


def _model() -> Model:
    return Model(
        id="test", provider="fake", context_window=4096, max_output_tokens=512
    )


@pytest.mark.asyncio
async def test_tool_call_args_parse_error_is_emitted_on_bus() -> None:
    bus = EventBus()
    parse_errors: list[ToolCallArgsParseError] = []
    stream_parse_errors: list[ToolCallArgsParseError] = []

    bus.on(ToolCallArgsParseError.CHANNEL, lambda event: parse_errors.append(event))
    bus.on(
        "stream_delta",
        lambda event: stream_parse_errors.append(event.delta)
        if isinstance(event.delta, ToolCallArgsParseError)
        else None,
    )

    loop = AgentLoop(
        stream_fn=_MalformedJsonStream(),
        bus=bus,
        config=LoopConfig(max_turns=1),
    )
    messages = await loop.run(messages=[text_message("hi")], model=_model(), tools=[])

    assert len(parse_errors) == 1
    assert parse_errors == stream_parse_errors
    assert parse_errors[0].tool_call_id == "call-1"
    assert parse_errors[0].raw == '{"unterminated"'
    assistant = messages[1]
    block = assistant.content[1]
    assert isinstance(block, ToolCallBlock)
    assert block.arguments == {}
    text = assistant.content[0]
    assert isinstance(text, TextContent)
    assert text.text == "calling"
