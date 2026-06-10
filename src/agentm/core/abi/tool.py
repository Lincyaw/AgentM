"""Kernel tool contract.

Implements §3.2 (Tool Execution boundary) of
`.claude/designs/pluggable-architecture.md` — the bare ``Tool`` Protocol the
agent loop sees, plus the ``ToolResult`` / ``ToolOutcome`` data shapes.

Concrete adapters such as ``FunctionTool`` live outside the ABI surface in
``agentm.core._internal.tools``; they are re-exported from this package's
``__init__`` for ergonomics but are not part of the boundary contract.

Schemas are raw JSON Schema dicts; no pydantic in the kernel.

Tools may return either a bare :class:`ToolResult` (legacy/simple tools) or a
structured :class:`ToolOutcome` — see :mod:`agentm.core.abi.loop` for how the
kernel resolves each variant into the next loop action. This split keeps the
"this tool wants to terminate" decision out of ``ToolResult`` itself, where it
would have to compete with ``is_error`` for meaning.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .messages import ImageContent, TextContent


# --- File-op metadata vocabulary -------------------------------------------
#
# Tools that touch the filesystem self-describe via ``metadata[FILE_OP_METADATA_KEY]``
# so kernel subsystems (notably the compaction engine) can route file
# operations without hard-coding tool names. See issue #76.

FILE_OP_METADATA_KEY = "file_op"
FILE_OP_READ = "read"
FILE_OP_WRITE = "write"
FILE_OP_EDIT = "edit"
TOOL_RESULT_FORMAT_METADATA_KEY = "result_format"


@dataclass(slots=True)
class ToolResult:
    """The result of one tool execution.

    ``content`` is the user-visible payload (text and/or images) that becomes
    a ``ToolResultBlock``. ``extras`` is opaque structured data the runtime
    or extensions may use (e.g. for richer rendering); the kernel never reads
    it.
    """

    content: list[TextContent | ImageContent]
    is_error: bool = False
    extras: Any = None


@dataclass(slots=True, frozen=True)
class ToolOutcome:
    """Sealed sum-type base for what a tool returns.

    Subclasses are the only legal instantiations: :class:`ToolContinue` for
    normal tool execution and :class:`ToolTerminate` to declare this tool a
    terminal action that ends the agent loop. The kernel pattern-matches on
    the concrete type to decide the next action.

    Tools may also return a bare :class:`ToolResult`; the kernel normalizes
    it to ``ToolContinue(result)`` so existing tools keep working.
    """


@dataclass(slots=True, frozen=True)
class ToolContinue(ToolOutcome):
    """Normal tool execution — the loop should keep running."""

    result: ToolResult


@dataclass(slots=True, frozen=True)
class ToolTerminate(ToolOutcome):
    """The tool succeeded and asks the loop to terminate after this turn.

    ``reason`` is opaque to the kernel — surfaced verbatim through the
    ``ToolTerminated`` cause to extensions and observability so downstream
    consumers can distinguish *which* terminal tool fired.

    Namespace convention (recommended): prefix ``reason`` with the
    extension or scenario short name and a colon, e.g.
    ``"rca:final-report-submitted"`` or ``"sandbox:shutdown"``.
    The kernel cannot enumerate scenario-defined reasons, so a
    namespaced string keeps observers safe from collisions when two
    scenarios pick the same bare label. The §11 validator emits a
    soft warning for unprefixed reasons; existing single-scenario
    reasons keep working.
    """

    result: ToolResult
    reason: str


@runtime_checkable
class Tool(Protocol):
    """Bare execution contract every tool must satisfy.

    The agent loop only depends on these four members:

    - ``name`` / ``description`` / ``parameters`` (JSON Schema dict) — used
      when assembling the tool catalog passed to the LLM stream.
    - ``execute(args, *, signal)`` — the call that runs the tool.

    ``signal`` is an :class:`asyncio.Event`; tools may poll it to abort
    cooperatively. Streaming progress is intentionally *not* part of the
    kernel surface: the previous ``on_update`` parameter was never wired
    through and has been removed. A future progress channel will be a
    deliberate event-bus extension, not a dead Protocol parameter.

    Returning a bare :class:`ToolResult` is treated as ``ToolContinue(result)``;
    a tool that wants to end the loop returns :class:`ToolTerminate` instead.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult | ToolOutcome: ...


@dataclass(slots=True)
class FunctionTool:
    """Concrete ``Tool`` adapter wrapping an async callable.

    Useful for tests and trivial cases where a full tool class would be
    overkill. The wrapped ``fn`` is called with the raw ``args`` dict; if it
    raises, the exception **propagates** — ``FunctionTool`` deliberately does
    not convert exceptions to ``ToolResult(is_error=True)``. The agent loop is
    responsible for that conversion so the policy is uniform across all tool
    implementations.

    ``fn`` may return either a bare :class:`ToolResult` or any
    :class:`ToolOutcome`; the kernel handles both.

    ``parameters`` accepts either a JSON Schema dict or a
    :class:`pydantic.BaseModel` subclass. A Pydantic class is
    automatically converted to a provider-neutral JSON Schema at
    construction time via ``pydantic_to_tool_schema``.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], Awaitable[ToolResult | ToolOutcome]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | type,
        fn: Callable[[dict[str, Any]], Awaitable[ToolResult | ToolOutcome]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.metadata = metadata or {}
        if isinstance(parameters, type):
            from agentm.core.lib.tool_schema import pydantic_to_tool_schema

            self.parameters = pydantic_to_tool_schema(parameters)
        else:
            self.parameters = parameters

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult | ToolOutcome:
        """Invoke the wrapped function. Exceptions propagate unchanged."""

        return await self.fn(args)


__all__ = [
    "FILE_OP_EDIT",
    "FILE_OP_METADATA_KEY",
    "FILE_OP_READ",
    "FILE_OP_WRITE",
    "FunctionTool",
    "TOOL_RESULT_FORMAT_METADATA_KEY",
    "Tool",
    "ToolContinue",
    "ToolOutcome",
    "ToolResult",
    "ToolTerminate",
]
