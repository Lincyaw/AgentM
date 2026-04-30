from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
    ToolResultMessage,
)
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.extension import ProviderConfig

from agentm.extensions.builtin import dedup


@pytest.mark.asyncio
async def test_handler_blocks_recent_duplicate_call() -> None:
    class API:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def on(self, channel: str, handler: Any) -> Any:
            self.handlers[channel] = handler
            return lambda: None

    api = API()
    dedup.install(api, {"window": 2})
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
    provider_name = "tests.unit.extensions.builtin.dedup._provider"
    provider_mod = types.ModuleType(provider_name)

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

    def install_provider(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-repeat",
            ProviderConfig(
                stream_fn=RepeatStream(),
                model=Model(
                    id="fake-repeat",
                    provider="fake",
                    context_window=10000,
                    max_output_tokens=1000,
                ),
                name="fake-repeat",
            ),
        )

    provider_mod.install = install_provider  # type: ignore[attr-defined]
    sys.modules[provider_name] = provider_mod

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            ("tests.unit.harness_v2._fixtures.echo_ext", {}),
            ("agentm.extensions.builtin.dedup", {"window": 10}),
        ],
        provider=(provider_name, {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    final = await session.prompt("hello")

    duplicate_result = final[4]
    assert isinstance(duplicate_result, ToolResultMessage)
    assert "duplicate of recent call" in duplicate_result.content[0].content[0].text

    await session.shutdown()
