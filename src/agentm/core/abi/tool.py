# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Kernel tool contract.

Implements the Tool Execution boundary in
``docs/refactor-abstract-inventory.md``: the bare ``Tool`` Protocol the agent
loop sees, plus the ``ToolResult`` / ``ToolOutcome`` data shapes.

``FunctionTool`` is a concrete adapter that wraps an async callable — it
lives in this module alongside the Protocol for ergonomic ``from
agentm.core.abi import FunctionTool`` access.

Schemas are raw JSON Schema dicts; no pydantic in the kernel.

Tools may return either a bare :class:`ToolResult` (simple tools) or a
structured :class:`ToolOutcome` — see :mod:`agentm.core.abi.loop` for how the
kernel resolves each variant into the next loop action. This split keeps the
"this tool wants to terminate" decision out of ``ToolResult`` itself, where it
would have to compete with ``is_error`` for meaning.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .cancel import CancelSignal
from .messages import ImageContent, JsonValue, TextContent, freeze_json

if TYPE_CHECKING:
    from .tool_executor import ToolExecutionRequirements


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


@dataclass(slots=True, frozen=True, init=False)
class ToolResult:
    """The result of one tool execution.

    ``content`` is the user-visible payload (text and/or images) that becomes
    a ``ToolResultBlock``. ``extras`` is immutable JSON data so results remain
    portable across process, event-bus, and trajectory persistence boundaries.
    """

    content: tuple[TextContent | ImageContent, ...]
    is_error: bool
    extras: JsonValue

    def __init__(
        self,
        content: Sequence[TextContent | ImageContent],
        is_error: bool = False,
        extras: object = None,
    ) -> None:
        blocks = tuple(content)
        if not all(isinstance(item, (TextContent, ImageContent)) for item in blocks):
            raise TypeError("tool result content must contain text or image blocks")
        if not isinstance(is_error, bool):
            raise TypeError("tool result is_error must be a bool")
        object.__setattr__(self, "content", blocks)
        object.__setattr__(self, "is_error", is_error)
        object.__setattr__(self, "extras", freeze_json(extras))


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

    def __post_init__(self) -> None:
        if type(self) is ToolOutcome:
            raise TypeError(
                "ToolOutcome is abstract; use ToolContinue or ToolTerminate"
            )


@dataclass(slots=True, frozen=True)
class ToolContinue(ToolOutcome):
    """Normal tool execution — the loop should keep running."""

    result: ToolResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, ToolResult):
            raise TypeError("ToolContinue result must be a ToolResult")


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
    scenarios pick the same bare label. The validator emits a
    soft warning for unprefixed reasons; existing single-scenario
    reasons keep working.
    """

    result: ToolResult
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, ToolResult):
            raise TypeError("ToolTerminate result must be a ToolResult")
        if not isinstance(self.reason, str) or not self.reason:
            raise TypeError("ToolTerminate reason must be a non-empty string")


@runtime_checkable
class Tool(Protocol):
    """Bare execution contract every tool must satisfy.

    The agent loop only depends on these four members:

    - ``name`` / ``description`` / ``parameters`` (JSON Schema dict) — used
      when assembling the tool list passed to the LLM stream.
    - ``execute(args, *, signal)`` — the call that runs the tool.

    ``signal`` is a :class:`CancelSignal`; tools may poll it to abort
    cooperatively. Streaming progress is intentionally *not* part of the
    kernel surface: the previous ``on_update`` parameter was never wired
    through and has been removed. A future progress channel will be a
    deliberate event-bus extension, not a dead Protocol parameter.

    Returning a bare :class:`ToolResult` is treated as ``ToolContinue(result)``;
    a tool that wants to end the loop returns :class:`ToolTerminate` instead.
    Tools may also expose an optional ``metadata`` dict for cross-cutting
    classification such as file operation type or result format. Execution
    substrate is runtime policy, not a per-tool ABI contract.
    """

    name: str
    description: str
    parameters: dict[str, object]

    async def execute(
        self,
        args: dict[str, object],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome: ...


@runtime_checkable
class ToolMetadataProvider(Protocol):
    """Optional typed capability for tool classification metadata."""

    metadata: dict[str, object]


@dataclass(slots=True)
class FunctionTool:
    """Concrete ``Tool`` adapter wrapping an async callable.

    Useful for tests and trivial cases where a full tool class would be
    overkill. The wrapped ``fn`` is called with the raw ``args`` dict, and also
    receives ``signal=`` when its signature declares that parameter. If it
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
    parameters: dict[str, object]
    fn: Callable[..., Awaitable[ToolResult | ToolOutcome]]
    metadata: dict[str, object] = field(default_factory=dict)
    execution_requirements: "ToolExecutionRequirements | None" = None
    _accepts_signal: bool = False

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, object] | type,
        fn: Callable[..., Awaitable[ToolResult | ToolOutcome]],
        metadata: dict[str, object] | None = None,
        execution_requirements: "ToolExecutionRequirements | None" = None,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.metadata = metadata or {}
        self.execution_requirements = execution_requirements
        self._accepts_signal = _accepts_signal(fn)
        if isinstance(parameters, type):
            from agentm.core.lib.tool_schema import pydantic_to_tool_schema

            self.parameters = pydantic_to_tool_schema(parameters)
        else:
            self.parameters = parameters

    async def execute(
        self,
        args: dict[str, object],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        """Invoke the wrapped function. Exceptions propagate unchanged."""

        if self._accepts_signal:
            return await self.fn(args, signal=signal)
        return await self.fn(args)


def _accepts_signal(fn: Callable[..., object]) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "signal" in parameters


__all__ = [
    "FILE_OP_EDIT",
    "FILE_OP_METADATA_KEY",
    "FILE_OP_READ",
    "FILE_OP_WRITE",
    "FunctionTool",
    "TOOL_RESULT_FORMAT_METADATA_KEY",
    "Tool",
    "ToolContinue",
    "ToolMetadataProvider",
    "ToolOutcome",
    "ToolResult",
    "ToolTerminate",
]
