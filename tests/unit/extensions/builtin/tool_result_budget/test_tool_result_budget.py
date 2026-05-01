from __future__ import annotations

import sys
import types
from typing import Any, cast

import pytest

from agentm.core.kernel import FunctionTool, ImageContent, TextContent, ToolResult, ToolResultEvent, ToolResultMessage
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig

from agentm.extensions.builtin import tool_result_budget


@pytest.mark.asyncio
async def test_handler_truncates_text_and_keeps_images() -> None:
    class API:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def on(self, channel: str, handler: Any) -> Any:
            self.handlers[channel] = handler
            return lambda: None

    api = API()
    tool_result_budget.install(cast(Any, api), {"max_chars": 5})

    result = api.handlers["tool_result"](
        ToolResultEvent(
            tool_call_id="c1",
            tool_name="image_tool",
            result=ToolResult(
                content=[
                    TextContent(type="text", text="abcdefghij"),
                    ImageContent(type="image", data=b"img", mime_type="image/png"),
                ]
            ),
        )
    )

    assert isinstance(result, ToolResult)
    assert result.content[0] == TextContent(type="text", text="abcde")
    assert isinstance(result.content[1], ImageContent)
    assert isinstance(result.content[2], TextContent)
    assert "truncated 5 chars" in result.content[2].text


@pytest.mark.asyncio
async def test_integration_truncates_large_tool_output(tmp_path) -> None:
    module_name = "tests.unit.extensions.builtin.tool_result_budget._big_tool_ext"
    mod = types.ModuleType(module_name)

    async def _big_result(_args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="x" * 20)])

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_tool(
            FunctionTool(
                name="echo",
                description="echo",
                parameters={},
                fn=_big_result,
            )
        )

    mod.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = mod

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            (module_name, {}),
            ("agentm.extensions.builtin.tool_result_budget", {"max_chars": 5}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    final = await session.prompt("hello")

    tool_result_message = final[2]
    assert isinstance(tool_result_message, ToolResultMessage)
    assert isinstance(tool_result_message.content[0].content[0], TextContent)
    assert tool_result_message.content[0].content[0].text == "xxxxx"
    assert isinstance(tool_result_message.content[0].content[1], TextContent)
    assert "tool_result_budget truncated" in tool_result_message.content[0].content[1].text

    await session.shutdown()
