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


class ScriptedStream:
    def __init__(self, *, tool_name: str, arguments: dict[str, Any], final_text: str) -> None:
        self._tool_name = tool_name
        self._arguments = arguments
        self._final_text = final_text
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

    async def _iter(self, call_index: int) -> AsyncIterator[AssistantStreamEvent]:
        if call_index == 1:
            import json

            yield ToolCallStart(id="call-1", name=self._tool_name)
            yield ToolCallArgsDelta(
                id="call-1",
                args_json_delta=json.dumps(self._arguments, sort_keys=True),
            )
            yield ToolCallEnd(id="call-1")
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            type="tool_call",
                            id="call-1",
                            name=self._tool_name,
                            arguments=dict(self._arguments),
                        )
                    ],
                    timestamp=1.0,
                    stop_reason="tool_use",
                )
            )
            return

        yield TextDelta(text=self._final_text)
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=self._final_text)],
                timestamp=2.0,
                stop_reason="end_turn",
            )
        )


def install(api: Any, config: dict[str, Any]) -> None:
    stream = ScriptedStream(
        tool_name=str(config["tool_name"]),
        arguments=dict(config.get("arguments", {})),
        final_text=str(config.get("final_text", "done")),
    )
    model = Model(
        id="scripted-fake",
        provider="scripted-fake",
        context_window=10000,
        max_output_tokens=1000,
    )
    api.register_provider(
        "scripted-fake",
        ProviderConfig(stream_fn=stream, model=model, name="scripted-fake"),
    )
