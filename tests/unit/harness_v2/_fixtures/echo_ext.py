"""Echo tool extension for the smoke test."""

from __future__ import annotations

from typing import Any

from agentm.core.kernel import FunctionTool, TextContent, ToolResult


async def _echo(args: dict[str, Any]) -> ToolResult:
    text = str(args.get("text", ""))
    return ToolResult(
        content=[TextContent(type="text", text=f"echoed: {text}")]
    )


def install(api: Any, config: dict[str, Any]) -> None:
    api.register_tool(
        FunctionTool(
            name="echo",
            description="Echo back the provided text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            fn=_echo,
        )
    )
