from __future__ import annotations

from typing import cast

from agentm.core.abi import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    Tool,
    ToolResult,
    Usage,
)
from agentm.core.lib.render import (
    assistant_text,
    final_summary,
    tool_call_text,
    tool_result_format,
    tool_result_text,
)


def test_assistant_and_tool_result_text_are_stable() -> None:
    message = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="hello"),
            ToolCallBlock(
                type="tool_call", id="tc-1", name="read", arguments={"path": "x"}
            ),
            TextContent(type="text", text="world"),
        ],
        timestamp=1.0,
        usage=Usage(input_tokens=3, output_tokens=5, cache_read=7, cache_write=11),
    )
    result = ToolResult(
        content=[TextContent(type="text", text="ok"), TextContent(type="text", text="done")]
    )

    assert assistant_text(message) == "hello\nworld"
    assert (
        tool_call_text(cast(ToolCallBlock, message.content[1]))
        == 'read({"path": "x"})'
    )
    assert tool_result_text(result) == "ok\n\ndone"

    report = final_summary([message])
    assert report.text == "hello\nworld"
    assert report.tool_calls == 1
    assert report.usage.input_tokens == 3
    assert report.usage.cache_write == 11


def test_tool_result_format_uses_metadata_hint_instead_of_tool_name() -> None:
    class _Tool:
        name = "anything"
        metadata = {"result_format": "diff"}

    assert (
        tool_result_format("anything", "Updated 'x'", tool=cast(Tool, _Tool()))
        == "diff"
    )
    assert tool_result_format("tool_edit", "Updated 'x'") == "text"
