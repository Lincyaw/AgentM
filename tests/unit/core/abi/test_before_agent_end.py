from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import (
    AgentLoop,
    AssistantMessage,
    BeforeAgentEndEvent,
    EventBus,
    FunctionTool,
    LoopConfig,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
    ToolResult,
    text_message,
)


class _ScriptedStream:
    def __init__(self, scripted: list[AssistantMessage]) -> None:
        self._scripted = scripted
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
    ) -> AsyncIterator[MessageEnd]:
        del messages, model, tools, system, signal, thinking
        idx = self.calls
        self.calls += 1
        return self._iter(self._scripted[idx])

    async def _iter(self, message: AssistantMessage) -> AsyncIterator[MessageEnd]:
        yield MessageEnd(message=message)


@pytest.mark.asyncio
async def test_before_agent_end_cancel_appends_message_and_keeps_loop_alive() -> None:
    stream = _ScriptedStream(
        [
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="wrapping up")],
                timestamp=1.0,
                stop_reason="end_turn",
            ),
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="final answer")],
                timestamp=2.0,
                stop_reason="end_turn",
            ),
        ]
    )
    bus = EventBus()
    before_stop_reasons: list[str] = []
    end_stop_reasons: list[str] = []

    async def on_before_agent_end(event: BeforeAgentEndEvent) -> dict[str, Any] | None:
        before_stop_reasons.append(event.stop_reason)
        if len(before_stop_reasons) == 1:
            return {
                "cancel": True,
                "append": [text_message("<subagent_result>ready</subagent_result>")],
            }
        return None

    bus.on("before_agent_end", on_before_agent_end)
    bus.on("agent_end", lambda e: end_stop_reasons.append(e.stop_reason))

    loop = AgentLoop(stream_fn=stream, bus=bus)
    messages = await loop.run(
        messages=[text_message("start")],
        model=Model(id="test", provider="fake", context_window=4096, max_output_tokens=512),
        tools=[],
    )

    assert stream.calls == 2
    assert before_stop_reasons == ["end_turn", "end_turn"]
    assert end_stop_reasons == ["end_turn"]
    assert messages[2].role == "user"
    injected_block = messages[2].content[0]
    assert getattr(injected_block, "text", None) == "<subagent_result>ready</subagent_result>"
    assert messages[-1].role == "assistant"
    final_block = messages[-1].content[0]
    assert getattr(final_block, "text", None) == "final answer"


@pytest.mark.asyncio
async def test_max_turns_still_fires_before_agent_end_but_cannot_be_cancelled() -> None:
    stream = _ScriptedStream(
        [
            AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id="t1",
                        name="noop",
                        arguments={},
                    )
                ],
                timestamp=1.0,
                stop_reason="tool_use",
            )
        ]
    )
    bus = EventBus()
    before_stop_reasons: list[str] = []
    end_stop_reasons: list[str] = []

    async def on_before_agent_end(event: BeforeAgentEndEvent) -> dict[str, Any]:
        before_stop_reasons.append(event.stop_reason)
        return {
            "cancel": True,
            "append": [text_message("max-turns hook ran")],
        }

    bus.on("before_agent_end", on_before_agent_end)
    bus.on("agent_end", lambda e: end_stop_reasons.append(e.stop_reason))

    async def _noop(_args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="ok")])

    loop = AgentLoop(stream_fn=stream, bus=bus, config=LoopConfig(max_turns=1))
    messages = await loop.run(
        messages=[text_message("start")],
        model=Model(id="test", provider="fake", context_window=4096, max_output_tokens=512),
        tools=[
            FunctionTool(
                name="noop",
                description="No-op tool",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                fn=_noop,
            )
        ],
    )

    assert before_stop_reasons == ["max_turns"]
    assert end_stop_reasons == ["max_turns"]
    assert messages[-1].role == "user"
    final_block = messages[-1].content[0]
    assert getattr(final_block, "text", None) == "max-turns hook ran"
