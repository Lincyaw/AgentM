from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import (
    AgentLoop,
    AssistantMessage,
    EventBus,
    LlmRequestStartEvent,
    LoopConfig,
    MessageEnd,
    Model,
    TextContent,
    text_message,
    AgentMessage,
)


class _Stream:
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
    ) -> AsyncIterator[MessageEnd]:
        del messages, model, tools, system, signal, thinking
        self.calls += 1
        return self._iter(self.calls)

    async def _iter(self, call: int) -> AsyncIterator[MessageEnd]:
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"done {call}")],
                timestamp=float(call),
                stop_reason="end_turn",
            )
        )


def _model() -> Model:
    return Model(
        id="test", provider="fake", context_window=4096, max_output_tokens=512
    )


@pytest.mark.asyncio
async def test_turn_id_is_monotone_across_prompt_runs_after_branch_fork() -> None:
    bus = EventBus()
    seen: list[tuple[int, int]] = []

    def _capture(event: LlmRequestStartEvent) -> None:
        seen.append((event.turn_index, event.turn_id))

    bus.on(LlmRequestStartEvent.CHANNEL, _capture)
    loop = AgentLoop(stream_fn=_Stream(), bus=bus, config=LoopConfig(max_turns=1))

    first_branch: list[AgentMessage] = [text_message("first")]
    await loop.run(messages=first_branch, model=_model(), tools=[])

    forked_mid_second_prompt: list[AgentMessage] = [
        *first_branch,
        text_message("second after fork"),
    ]
    await loop.run(messages=forked_mid_second_prompt, model=_model(), tools=[])

    assert seen == [(0, 0), (0, 1)]
