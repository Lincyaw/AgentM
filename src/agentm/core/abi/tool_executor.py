"""Tool execution capability port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import freeze_json
from agentm.core.abi.operations import EnvironmentOperations, EnvironmentRef
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

    def __post_init__(self) -> None:
        if self.isolation not in {"none", "thread", "process", "environment"}:
            raise ValueError(f"invalid tool isolation: {self.isolation!r}")
        if self.filesystem not in {"none", "read", "write"}:
            raise ValueError(f"invalid tool filesystem access: {self.filesystem!r}")
        if self.concurrency not in {"exclusive", "parallel_safe"}:
            raise ValueError(f"invalid tool concurrency: {self.concurrency!r}")
        if self.interrupt not in {"block", "cancel"}:
            raise ValueError(f"invalid tool interrupt behavior: {self.interrupt!r}")
        if not isinstance(self.killable, bool) or not isinstance(self.network, bool):
            raise TypeError("tool killable and network requirements must be bools")
        if self.environment_id is not None and (
            not isinstance(self.environment_id, str) or not self.environment_id
        ):
            raise TypeError("tool environment_id must be a non-empty string or None")


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

    def __post_init__(self) -> None:
        if self.environment is not None and not isinstance(
            self.environment,
            EnvironmentRef,
        ):
            raise TypeError("executor environment must be an EnvironmentRef or None")
        _validate_capability_tuple(
            self.isolation,
            {"none", "thread", "process", "environment"},
            "executor isolation",
        )
        _validate_capability_tuple(
            self.filesystem,
            {"none", "read", "write"},
            "executor filesystem",
        )
        _validate_capability_tuple(
            self.concurrency,
            {"exclusive", "parallel_safe"},
            "executor concurrency",
        )
        _validate_capability_tuple(
            self.interrupt,
            {"block", "cancel"},
            "executor interrupt",
        )
        if not isinstance(self.killable, bool) or not isinstance(self.network, bool):
            raise TypeError("executor killable and network capabilities must be bools")


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
        if self.requirements.environment_id is not None and (
            self.environment is None
            or self.environment.id != self.requirements.environment_id
        ):
            raise ValueError(
                "tool execution request environment does not satisfy environment_id"
            )
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

    def capabilities(self) -> ToolExecutionCapabilities: ...

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


def _validate_capability_tuple(
    value: object,
    allowed: set[str],
    label: str,
) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or item not in allowed for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty tuple of supported values")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")


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
