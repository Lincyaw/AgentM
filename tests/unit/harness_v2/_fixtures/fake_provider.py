"""Fake LLM provider extension for the smoke test.

Registers a deterministic two-call ``StreamFn`` that drives the loop through
exactly one tool-use turn followed by a final-answer turn.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.kernel import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallBlock,
    ToolCallEnd,
    ToolCallStart,
)

from agentm.harness.extension import ProviderConfig


class FakeStream:
    """Two-shot fake stream mirroring the kernel smoke test."""

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
        self.calls += 1
        return self._iter(self.calls)

    async def _iter(self, n: int) -> AsyncIterator[AssistantStreamEvent]:
        if n == 1:
            yield ToolCallStart(id="call-1", name="echo")
            yield ToolCallArgsDelta(id="call-1", args_json_delta='{"text":"hi"}')
            yield ToolCallEnd(id="call-1")
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            type="tool_call",
                            id="call-1",
                            name="echo",
                            arguments={"text": "hi"},
                        )
                    ],
                    timestamp=1.0,
                    stop_reason="tool_use",
                )
            )
        else:
            yield TextDelta(text="done")
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="done")],
                    timestamp=2.0,
                    stop_reason="end_turn",
                )
            )


def install(api: Any, config: dict[str, Any]) -> None:
    stream = FakeStream()
    model = Model(
        id="fake",
        provider="fake",
        context_window=10000,
        max_output_tokens=1000,
    )
    api.register_provider(
        "fake",
        ProviderConfig(stream_fn=stream, model=model, name="fake"),
    )
