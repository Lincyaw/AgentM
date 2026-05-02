"""Kernel tool contract.

Implements §3.2 (Tool Execution boundary) of
`.claude/designs/pluggable-architecture.md` — the bare ``Tool`` Protocol the
agent loop sees, plus a ``FunctionTool`` adapter for tests and simple cases.

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
    ``"rca:final-report-submitted"`` or ``"plan_mode:plan-accepted"``.
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
    - ``execute(args, *, signal, on_update)`` — the call that runs the tool.

    ``signal`` is an :class:`asyncio.Event`; tools may poll it to abort
    cooperatively. ``on_update`` lets long-running tools push progress events
    (the kernel itself doesn't dispatch them; the harness wires it up).

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
        on_update: Callable[[Any], None] | None = None,
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
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], Awaitable[ToolResult | ToolOutcome]]
    # Mirrors the Tool protocol surface; not consumed by the kernel itself.
    metadata: dict[str, Any] = field(default_factory=dict)

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
        on_update: Callable[[Any], None] | None = None,
    ) -> ToolResult | ToolOutcome:
        """Invoke the wrapped function. Exceptions propagate unchanged."""

        return await self.fn(args)


__all__ = [
    "FunctionTool",
    "Tool",
    "ToolContinue",
    "ToolOutcome",
    "ToolResult",
    "ToolTerminate",
]
