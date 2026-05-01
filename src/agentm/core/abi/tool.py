"""Kernel tool contract.

Implements §3.2 (Tool Execution boundary) of
`.claude/designs/pluggable-architecture.md` — the bare ``Tool`` Protocol the
agent loop sees, plus a ``FunctionTool`` adapter for tests and simple cases.

Schemas are raw JSON Schema dicts; no pydantic in the kernel.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .messages import ImageContent, TextContent


@dataclass(slots=True, init=False)
class ToolResult:
    """The result of one tool execution.

    ``content`` is the user-visible payload (text and/or images) that becomes
    a ``ToolResultBlock``. ``extras`` is opaque structured data the harness
    or extensions may use (e.g. for richer rendering); the kernel never reads
    it. ``details`` remains as a backwards-compatible alias because earlier
    extensions and tests already used that name.
    """

    content: list[TextContent | ImageContent]
    is_error: bool = False
    extras: Any = None

    def __init__(
        self,
        content: list[TextContent | ImageContent],
        is_error: bool = False,
        details: Any = None,
        extras: Any = None,
    ) -> None:
        self.content = content
        self.is_error = is_error
        if extras is not None and details is not None and extras != details:
            raise ValueError("ToolResult received conflicting details and extras")
        self.extras = extras if extras is not None else details

    @property
    def details(self) -> Any:
        return self.extras

    @details.setter
    def details(self, value: Any) -> None:
        self.extras = value


@runtime_checkable
class Tool(Protocol):
    """Bare execution contract every tool must satisfy.

    The agent loop only depends on these four members:

    - ``name`` / ``description`` / ``parameters`` (JSON Schema dict) — used
      when assembling the tool catalog passed to the LLM stream.
    - ``execute(args, *, signal, on_update)`` — the call that runs the tool.

    ``signal`` is an :class:`asyncio.Event`; tools may poll it to abort
    cooperatively. ``on_update`` lets long-running tools push progress events
    (the kernel itself doesn't dispatch them; the harness wires it up).
    """

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[Any], None] | None = None,
    ) -> ToolResult: ...


@dataclass(slots=True)
class FunctionTool:
    """Concrete ``Tool`` adapter wrapping an async callable.

    Useful for tests and trivial cases where a full tool class would be
    overkill. The wrapped ``fn`` is called with the raw ``args`` dict; if it
    raises, the exception **propagates** — ``FunctionTool`` deliberately does
    not convert exceptions to ``ToolResult(is_error=True)``. The agent loop is
    responsible for that conversion so the policy is uniform across all tool
    implementations.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], Awaitable[ToolResult]]
    # Mirrors the Tool protocol surface; not consumed by the kernel itself.
    metadata: dict[str, Any] = field(default_factory=dict)

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[Any], None] | None = None,
    ) -> ToolResult:
        """Invoke the wrapped function. Exceptions propagate unchanged."""

        return await self.fn(args)


__all__ = ["FunctionTool", "Tool", "ToolResult"]
