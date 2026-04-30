from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentm.core.kernel import (
    AssistantMessage,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
)
from agentm.harness.extension import ProviderConfig


class RecordingStream:
    def __init__(self, scripted_messages: list[AssistantMessage]) -> None:
        self._scripted_messages = scripted_messages
        self.calls = 0
        self.seen_systems: list[str | None] = []
        self.seen_messages: list[list[Any]] = []
        self.seen_tool_names: list[list[str]] = []

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[Any]:
        self.calls += 1
        self.seen_systems.append(system)
        self.seen_messages.append(list(messages))
        self.seen_tool_names.append([tool.name for tool in tools])
        index = min(self.calls - 1, len(self._scripted_messages) - 1)
        return self._iter(self._scripted_messages[index])

    async def _iter(self, message: AssistantMessage) -> AsyncIterator[Any]:
        yield MessageEnd(message=message)


LAST_STREAM: RecordingStream | None = None


def install(api: Any, config: dict[str, Any]) -> None:
    global LAST_STREAM

    response_texts = [str(text) for text in config.get("response_texts", ["ok"])]
    tool_calls = config.get("tool_calls", [])
    scripted_messages: list[AssistantMessage] = []

    for index, text in enumerate(response_texts):
        content: list[Any]
        stop_reason = "end_turn"
        if index < len(tool_calls):
            tc = tool_calls[index]
            content = [
                ToolCallBlock(
                    type="tool_call",
                    id=str(tc.get("id", f"call-{index + 1}")),
                    name=str(tc["name"]),
                    arguments=dict(tc.get("arguments", {})),
                )
            ]
            stop_reason = "tool_use"
        else:
            content = [TextContent(type="text", text=text)]

        scripted_messages.append(
            AssistantMessage(
                role="assistant",
                content=content,
                timestamp=float(index + 1),
                stop_reason=stop_reason,
            )
        )

    LAST_STREAM = RecordingStream(scripted_messages)
    api.register_provider(
        "recording",
        ProviderConfig(
            stream_fn=LAST_STREAM,
            model=Model(
                id=str(config.get("model_id", "recording-model")),
                provider="recording",
                context_window=int(config.get("context_window", 4096)),
                max_output_tokens=int(config.get("max_output_tokens", 1024)),
            ),
            name="recording",
        ),
    )
