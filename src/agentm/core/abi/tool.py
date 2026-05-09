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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .messages import ImageContent, TextContent


@dataclass(slots=True)
class ToolResult:
    """The result of one tool execution.

    ``content`` is the user-visible payload (text and/or images) that becomes
    a ``ToolResultBlock``. ``extras`` is opaque structured data the harness
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


__all__ = [
    "Tool",
    "ToolContinue",
    "ToolOutcome",
    "ToolResult",
    "ToolTerminate",
]
