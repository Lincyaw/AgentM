"""Tool execution capability port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.operations import EnvironmentRef
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult


IsolationLevel = Literal["none", "thread", "process", "environment"]
FilesystemAccess = Literal["none", "read", "write"]
ToolConcurrency = Literal["exclusive", "parallel_safe"]
ToolInterruptBehavior = Literal["block", "cancel"]


@dataclass(frozen=True, slots=True)
class ToolExecutionRequirements:
    """Requirements a tool may declare without choosing the runtime backend."""

    isolation: IsolationLevel = "none"
    killable: bool = False
    filesystem: FilesystemAccess = "none"
    network: bool = False
    concurrency: ToolConcurrency = "exclusive"
    interrupt: ToolInterruptBehavior = "block"
    environment_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionCapabilities:
    """Capabilities provided by a concrete tool executor backend."""

    environment: EnvironmentRef | None = None
    isolation: tuple[IsolationLevel, ...] = ("none",)
    filesystem: tuple[FilesystemAccess, ...] = ("none",)
    killable: bool = False
    network: bool = False
    concurrency: tuple[ToolConcurrency, ...] = ("exclusive", "parallel_safe")
    interrupt: tuple[ToolInterruptBehavior, ...] = ("block",)


@dataclass(frozen=True, slots=True)
class ToolExecutionRequest:
    """One tool invocation plus optional execution requirements."""

    tool: Tool
    args: Mapping[str, object]
    requirements: ToolExecutionRequirements | None = None
    environment: EnvironmentRef | None = None
    cwd: str | None = None
    metadata: Mapping[str, str | int | float | bool | None] | None = None


@runtime_checkable
class ToolExecutionRequirementsProvider(Protocol):
    """Optional protocol for tools that declare executor requirements."""

    execution_requirements: ToolExecutionRequirements


@runtime_checkable
class ToolExecutor(Protocol):
    """Runtime-owned boundary that executes tool calls."""

    def capabilities(self) -> ToolExecutionCapabilities:
        ...

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        ...


def tool_execution_requirements(tool: Tool) -> ToolExecutionRequirements | None:
    """Return typed requirements declared by a tool, if any."""

    candidate = getattr(tool, "execution_requirements", None)
    if isinstance(candidate, ToolExecutionRequirements):
        return candidate
    return None


__all__ = [
    "FilesystemAccess",
    "IsolationLevel",
    "ToolConcurrency",
    "ToolExecutionCapabilities",
    "ToolInterruptBehavior",
    "ToolExecutionRequest",
    "ToolExecutionRequirements",
    "ToolExecutionRequirementsProvider",
    "ToolExecutor",
    "tool_execution_requirements",
]
