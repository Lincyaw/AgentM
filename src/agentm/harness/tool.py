"""Unified Tool abstraction for the AgentM SDK. No langchain dependency."""
from __future__ import annotations

import asyncio
import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints, overload

from pydantic import TypeAdapter

from agentm.harness.types import ToolCallable


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _schema_from_func(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive JSON Schema from function signature + type hints.

    Uses pydantic TypeAdapter for complex types (Literal, Optional, list[str]).
    Skips 'self', 'cls', 'return'. Params without defaults go into 'required'.
    """
    # Unwrap functools.partial to get type hints from the original function
    raw = fn
    while isinstance(raw, functools.partial):
        raw = raw.func
    hints = get_type_hints(raw)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "return"):
            continue
        annotation = hints.get(name, str)
        field_schema = TypeAdapter(annotation).json_schema()
        if param.default is not param.empty:
            field_schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = field_schema
    return {"type": "object", "properties": properties, "required": required}


# ---------------------------------------------------------------------------
# Tool dataclass
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """SDK canonical tool type. No langchain dependency."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolCallable

    async def ainvoke(self, args: dict[str, Any]) -> str:
        """Execute the tool. Normalizes return to str."""
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(**args)
        else:
            result = self.func(**args)
        return result if isinstance(result, str) else str(result)

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI function-calling format for model.bind_tools()."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def tool_from_function(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Create a Tool from an existing (possibly partial-bound) function."""
    resolved_name: str = name if name is not None else getattr(func, "__name__", "unknown")
    resolved_desc: str = description if description is not None else (func.__doc__ or "").strip() or resolved_name
    return Tool(
        name=resolved_name,
        description=resolved_desc,
        parameters=_schema_from_func(func),
        func=func,
    )


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
