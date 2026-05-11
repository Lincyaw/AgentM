"""Concrete tool adapters that live outside the ABI surface.

`agentm.core.abi.tool` defines only the bare ``Tool`` Protocol and the
``ToolResult`` / ``ToolOutcome`` data shapes — those are public boundary
types every layer agrees on. The concrete ``FunctionTool`` wrapper used by
tests and trivial atoms is an *implementation*, not part of the boundary,
so it lives here in ``_internal``.

It is re-exported from :mod:`agentm.core.abi` and :mod:`agentm.core.runtime` for
ergonomic access; nothing imports it from ``_internal`` directly except the
re-export shims (see ``core/abi/__init__.py``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi.tool import ToolOutcome, ToolResult


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
    ) -> ToolResult | ToolOutcome:
        """Invoke the wrapped function. Exceptions propagate unchanged."""

        return await self.fn(args)


__all__ = ["FunctionTool"]
