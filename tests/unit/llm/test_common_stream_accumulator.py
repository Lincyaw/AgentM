"""Provider-common stream assembly contracts."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from agentm.core.abi import TextContent, ToolCallBlock
from agentm.core.abi.tool import Tool
from agentm.llm._common import StreamAccumulator, ToolSpecAdapter, encode_tool_args


@dataclass
class _Tool:
    parameters: dict[str, Any]
    name: str = "echo"
    description: str = "Echo input"

    async def execute(self, args: dict[str, Any], **_: Any) -> Any:
        return args


def test_stream_accumulator_supports_third_provider_stub() -> None:
    class StubToolSpecAdapter(ToolSpecAdapter):
        def vendor_spec(self, tool: Tool) -> dict[str, Any]:
            return {"stub_name": tool.name, "schema": tool.parameters}

        def encode_tool_args(self, args: Mapping[str, Any]) -> str:
            return encode_tool_args(args)

    acc = StreamAccumulator()
    acc.add_text(0, "hello")
    acc.add_text(0, " world")
    acc.add_tool_call("tool-1", "echo", '{"text":"café"}')

    message = acc.assemble(
        stop_reason="tool",
        termination=None,
        usage=None,
        timestamp=42.0,
    )

    assert message.timestamp == 42.0
    assert isinstance(message.content[0], TextContent)
    assert message.content[0].text == "hello world"
    assert isinstance(message.content[1], ToolCallBlock)
    assert message.content[1].arguments == {"text": "café"}
    assert StubToolSpecAdapter().encode_tool_args({"text": "café"}) == '{"text": "café"}'
    assert StubToolSpecAdapter().vendor_spec(_Tool(parameters={"type": "object"})) == {
        "stub_name": "echo",
        "schema": {"type": "object"},
    }


def test_stream_accumulator_preserves_indexed_provider_order() -> None:
    acc = StreamAccumulator()
    acc.add_tool_call("tool-1", "echo", "{}", index=0)
    acc.add_text(1, "after")

    message = acc.assemble(
        stop_reason="tool",
        termination=None,
        usage=None,
        timestamp=42.0,
    )

    assert isinstance(message.content[0], ToolCallBlock)
    assert isinstance(message.content[1], TextContent)
