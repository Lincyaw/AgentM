from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from agentm.core.kernel import FunctionTool, TextContent, ToolResultMessage
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig

from agentm.extensions.builtin import tool_filter


@pytest.mark.asyncio
async def test_install_filters_registered_tools_immediately() -> None:
    class API:
        def __init__(self) -> None:
            self._tools = [
                FunctionTool(
                    name="echo",
                    description="echo",
                    parameters={},
                    fn=lambda _args: _tool_result("echo"),
                ),
                FunctionTool(
                    name="other",
                    description="other",
                    parameters={},
                    fn=lambda _args: _tool_result("other"),
                ),
            ]

    async def _tool_result(text: str):
        from agentm.core.kernel import ToolResult

        return ToolResult(content=[TextContent(type="text", text=text)])

    api = API()
    tool_filter.install(api, {"deny": ["echo"]})

    assert [tool.name for tool in api._tools] == ["other"]


@pytest.mark.asyncio
async def test_integration_filters_tool_before_loop_runs(tmp_path) -> None:
    module_name = "tests.unit.extensions.builtin.tool_filter._two_tools_ext"
    mod = types.ModuleType(module_name)

    async def _make_result(text: str):
        from agentm.core.kernel import ToolResult

        return ToolResult(content=[TextContent(type="text", text=text)])

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_tool(
            FunctionTool(
                name="echo",
                description="echo",
                parameters={},
                fn=lambda _args: _make_result("echo"),
            )
        )
        api.register_tool(
            FunctionTool(
                name="other",
                description="other",
                parameters={},
                fn=lambda _args: _make_result("other"),
            )
        )

    mod.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = mod

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            (module_name, {}),
            ("agentm.extensions.builtin.tool_filter", {"deny": ["echo"]}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    assert [tool.name for tool in session.tools] == ["other"]

    final = await session.prompt("hello")

    tool_result_message = final[2]
    assert isinstance(tool_result_message, ToolResultMessage)
    assert tool_result_message.content[0].content[0].text == "Unknown tool: echo"

    await session.shutdown()
