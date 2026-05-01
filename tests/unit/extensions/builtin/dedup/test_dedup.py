from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    MessageEnd,
    TextContent,
    ToolCallBlock,
    ToolResultMessage,
)
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig

from agentm.extensions.builtin import dedup
from tests.support.provider_registry import temporary_provider


@pytest.mark.asyncio
async def test_handler_blocks_recent_duplicate_call() -> None:
    class API:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def on(self, channel: str, handler: Any) -> Any:
            self.handlers[channel] = handler
            return lambda: None

    api = API()
    dedup.install(cast(Any, api), {"window": 2})
    api.handlers["agent_start"](object())

    first = api.handlers["tool_call"](
        dedup.ToolCallEvent(tool_call_id="c1", tool_name="echo", args={"a": 1})
    )
    second = api.handlers["tool_call"](
        dedup.ToolCallEvent(tool_call_id="c2", tool_name="echo", args={"a": 1})
    )

    assert first is None
    assert second == {"block": True, "reason": "duplicate of recent call"}


@pytest.mark.asyncio
async def test_integration_blocks_second_duplicate_call(tmp_path) -> None:
    class RepeatStream:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **_: Any) -> AsyncIterator[Any]:
            self.calls += 1
            return self._iter(self.calls)

        async def _iter(self, call_index: int) -> AsyncIterator[Any]:
            if call_index == 1:
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
                return
            if call_index == 2:
                yield MessageEnd(
                    message=AssistantMessage(
                        role="assistant",
                        content=[
                            ToolCallBlock(
                                type="tool_call",
                                id="call-2",
                                name="echo",
                                arguments={"text": "hi"},
                            )
                        ],
                        timestamp=2.0,
                        stop_reason="tool_use",
                    )
                )
                return
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="done")],
                    timestamp=3.0,
                    stop_reason="end_turn",
                )
            )

    with temporary_provider(RepeatStream(), provider_id="fake-repeat") as provider_id:
        config = AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("tests.unit.harness_v2._fixtures.echo_ext", {}),
                ("agentm.extensions.builtin.dedup", {"window": 10}),
            ],
            provider=provider_id,
            resource_loader=InMemoryResourceLoader(),
        )
        session = await AgentSession.create(config)

        final = await session.prompt("hello")

        duplicate_result = final[4]
        assert isinstance(duplicate_result, ToolResultMessage)
        assert isinstance(duplicate_result.content[0].content[0], TextContent)
        assert "duplicate of recent call" in duplicate_result.content[0].content[0].text

        await session.shutdown()
