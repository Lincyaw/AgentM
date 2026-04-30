from __future__ import annotations

from typing import Any

import pytest

from agentm.core.kernel import TextContent, ToolCallEvent, ToolResultMessage
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig

from agentm.extensions.builtin import permission


@pytest.mark.asyncio
async def test_handler_blocks_denylisted_tool() -> None:
    class API:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def on(self, channel: str, handler: Any) -> Any:
            self.handlers[channel] = handler
            return lambda: None

    api = API()
    permission.install(api, {"deny": ["bash"]})

    result = api.handlers["tool_call"](
        ToolCallEvent(tool_call_id="c1", tool_name="bash", args={})
    )

    assert result == {
        "block": True,
        "reason": "tool 'bash' denied by denylist",
    }


@pytest.mark.asyncio
async def test_integration_blocks_tool_call_in_session(tmp_path) -> None:
    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            ("tests.unit.harness_v2._fixtures.echo_ext", {}),
            ("agentm.extensions.builtin.permission", {"deny": ["echo"]}),
        ],
        provider=("tests.unit.harness_v2._fixtures.fake_provider", {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    final = await session.prompt("hello")

    tool_result_message = final[2]
    assert isinstance(tool_result_message, ToolResultMessage)
    first_block = tool_result_message.content[0]
    assert isinstance(first_block.content[0], TextContent)
    assert "Tool call blocked" in first_block.content[0].text
    assert "denylist" in first_block.content[0].text

    await session.shutdown()
