# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Tool execution port.

A tool declares only what is true of the tool: whether its work can be
abandoned mid-flight, and whether it can run alongside others. Where it runs
is a property of the session's composition, not of the tool, so isolation,
filesystem reach and network access are not asked here -- the execution world
is chosen once, by whoever binds the operations ports, and a tool that could
name a different one would be describing a deployment it cannot see.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import freeze_json
from agentm.core.abi.operations import EnvironmentOperations, EnvironmentRef
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult

ToolConcurrency = Literal["exclusive", "parallel_safe"]
ToolInterruptBehavior = Literal["block", "cancel"]


@dataclass(frozen=True, slots=True)
class ToolExecutionRequirements:
    """Requirements a tool may declare without choosing the runtime backend."""

    killable: bool = False
    concurrency: ToolConcurrency = "exclusive"
    interrupt: ToolInterruptBehavior = "block"

    def __post_init__(self) -> None:
        if self.concurrency not in {"exclusive", "parallel_safe"}:
            raise ValueError(f"invalid tool concurrency: {self.concurrency!r}")
        if self.interrupt not in {"block", "cancel"}:
            raise ValueError(f"invalid tool interrupt behavior: {self.interrupt!r}")
        if not isinstance(self.killable, bool):
            raise TypeError("tool killable requirement must be a bool")


@dataclass(frozen=True, slots=True)
class ToolExecutionRequest:
    """One tool invocation with fully resolved execution requirements."""

    tool: Tool
    args: Mapping[str, object]
    requirements: ToolExecutionRequirements = field(
        default_factory=ToolExecutionRequirements
    )
    environment: EnvironmentRef | None = None
    cwd: str | None = None
    metadata: Mapping[str, str | int | float | bool | None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool, Tool):
            raise TypeError("tool execution request tool does not satisfy Tool")
        frozen_args = freeze_json(self.args)
        if not isinstance(frozen_args, Mapping):
            raise TypeError("tool execution request args must be an object")
        object.__setattr__(self, "args", frozen_args)
        if not isinstance(self.requirements, ToolExecutionRequirements):
            raise TypeError(
                "tool execution request requirements must be ToolExecutionRequirements"
            )
        if self.environment is not None and not isinstance(
            self.environment,
            EnvironmentRef,
        ):
            raise TypeError(
                "tool execution request environment must be EnvironmentRef or None"
            )
        if self.cwd is not None and (not isinstance(self.cwd, str) or not self.cwd):
            raise TypeError("tool execution request cwd must be non-empty or None")
        if self.metadata is not None:
            object.__setattr__(
                self,
                "metadata",
                _freeze_metadata(self.metadata),
            )


@runtime_checkable
class ToolExecutionRequirementsProvider(Protocol):
    """Optional protocol for tools that declare executor requirements."""

    execution_requirements: ToolExecutionRequirements | None


@runtime_checkable
class EnvironmentExecutableTool(Protocol):
    """Tool adapter that can execute through a selected environment backend."""

    async def execute_in_environment(
        self,
        args: Mapping[str, object],
        *,
        environment: EnvironmentOperations,
        cwd: str | None = None,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Runtime-owned boundary that executes tool calls."""

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome: ...


def tool_execution_requirements(tool: Tool) -> ToolExecutionRequirements:
    """Resolve a tool declaration to a complete executor contract.

    Tools without a declaration receive the neutral in-process requirements.
    This is an ABI default, not a runtime recovery path.
    """

    if not isinstance(tool, ToolExecutionRequirementsProvider):
        return ToolExecutionRequirements()
    candidate = tool.execution_requirements
    if candidate is None:
        return ToolExecutionRequirements()
    if not isinstance(candidate, ToolExecutionRequirements):
        raise TypeError(f"tool {tool.name!r} declares invalid execution_requirements")
    return candidate


def _freeze_metadata(
    value: Mapping[str, str | int | float | bool | None],
) -> Mapping[str, str | int | float | bool | None]:
    copied: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("tool execution metadata keys must be strings")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise TypeError(f"tool execution metadata {key!r} must be a JSON scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"tool execution metadata {key!r} must be finite")
        copied[key] = item
    return MappingProxyType(copied)


__all__ = [
    "EnvironmentExecutableTool",
    "ToolConcurrency",
    "ToolExecutionRequest",
    "ToolExecutionRequirements",
    "ToolExecutionRequirementsProvider",
    "ToolExecutor",
    "ToolInterruptBehavior",
    "tool_execution_requirements",
]
