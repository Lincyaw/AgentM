"""Tool utilities for the Agent Harness layer.

Core types (Tool, ToolCallable, tool_from_function) live in agentm.core.tool
and are re-exported here for backward compatibility.
"""
from __future__ import annotations

from typing import Any, Callable, overload

from agentm.core.tool import Tool, ToolCallable, tool_from_function  # re-export

__all__ = ["Tool", "ToolCallable", "tool_from_function", "tool"]


# ---------------------------------------------------------------------------
# @tool decorator — works as @tool or @tool(name=..., description=...)
# ---------------------------------------------------------------------------

@overload
def tool(fn: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator to turn a function into a Tool.

    Usage::

        @tool
        def my_tool(x: int) -> str: ...

        @tool(name="alias")
        async def my_tool(x: int) -> str: ...
    """

    def _wrap(f: Callable[..., Any]) -> Tool:
        return tool_from_function(f, name=name, description=description)

    if fn is not None:
        # Called as @tool (no parentheses)
        return _wrap(fn)
    # Called as @tool(...) (with parentheses)
    return _wrap
