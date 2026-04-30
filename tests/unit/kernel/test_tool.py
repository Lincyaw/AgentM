"""Tests for the kernel tool contract."""

from __future__ import annotations

import pytest

from agentm.core.kernel.tool import FunctionTool, Tool, ToolResult
from agentm.core.kernel.messages import TextContent


def _make_tool(fn) -> FunctionTool:  # type: ignore[no-untyped-def]
    return FunctionTool(
        name="t",
        description="desc",
        parameters={"type": "object", "properties": {}},
        fn=fn,
    )


def test_function_tool_satisfies_tool_protocol() -> None:
    """`runtime_checkable` Protocol must accept FunctionTool. This is the
    contract the agent loop relies on when it accepts arbitrary tools."""

    async def fn(args: dict[str, object]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="ok")])

    tool = _make_tool(fn)
    assert isinstance(tool, Tool)


@pytest.mark.asyncio
async def test_function_tool_execute_returns_wrapped_result() -> None:
    """Whatever the wrapped fn returns must reach the caller verbatim."""

    sentinel = ToolResult(
        content=[TextContent(type="text", text="payload")],
        details={"k": 1},
    )

    async def fn(args: dict[str, object]) -> ToolResult:
        return sentinel

    tool = _make_tool(fn)
    result = await tool.execute({"x": 1})
    assert result is sentinel


@pytest.mark.asyncio
async def test_function_tool_does_not_swallow_exceptions() -> None:
    """The kernel deliberately leaves exception→error-result conversion to
    the loop, so all tool implementations get the same treatment. If
    FunctionTool silently wrapped exceptions, that uniformity would break."""

    async def fn(args: dict[str, object]) -> ToolResult:
        raise RuntimeError("bang")

    tool = _make_tool(fn)
    with pytest.raises(RuntimeError, match="bang"):
        await tool.execute({})
