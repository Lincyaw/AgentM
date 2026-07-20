"""Batch tool orchestration ABI.

``ToolExecutor`` executes one call. ``ToolOrchestrator`` owns the scheduling
policy across a batch of calls emitted by one assistant message: exclusive vs
parallel-safe grouping, cooperative cancellation, and sibling-error cascading.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import ToolCallBlock, freeze_json
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import ToolExecutionRequirements, ToolExecutor

ToolExecutionStatus = Literal["completed", "failed", "cancelled", "skipped"]


@dataclass(frozen=True, slots=True)
class ToolWorkItem:
    """Resolved tool call ready for execution."""

    index: int
    call: ToolCallBlock
    tool: Tool
    args: Mapping[str, object]
    requirements: ToolExecutionRequirements = field(
        default_factory=ToolExecutionRequirements
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or self.index < 0
        ):
            raise ValueError("tool work item index must be a non-negative integer")
        if not isinstance(self.call, ToolCallBlock):
            raise TypeError("tool work item call must be a ToolCallBlock")
        if not isinstance(self.tool, Tool):
            raise TypeError("tool work item tool does not satisfy Tool")
        frozen_args = freeze_json(self.args)
        if not isinstance(frozen_args, Mapping):
            raise TypeError("tool work item args must be an object")
        object.__setattr__(self, "args", frozen_args)
        if not isinstance(self.requirements, ToolExecutionRequirements):
            raise TypeError(
                "tool work item requirements must be ToolExecutionRequirements"
            )


@dataclass(frozen=True, slots=True)
class ToolOrchestrationRequest:
    """A batch of resolved tool calls from one assistant response."""

    items: tuple[ToolWorkItem, ...]
    session_id: str = ""
    turn_id: str = ""
    turn_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("tool orchestration items must be a tuple")
        if not all(isinstance(item, ToolWorkItem) for item in self.items):
            raise TypeError("tool orchestration items must contain ToolWorkItem values")
        indexes = tuple(item.index for item in self.items)
        if len(indexes) != len(set(indexes)):
            raise ValueError("tool orchestration item indexes must be unique")
        for label, value in (
            ("session_id", self.session_id),
            ("turn_id", self.turn_id),
        ):
            if not isinstance(value, str) or not value:
                raise TypeError(f"tool orchestration {label} must be non-empty")
        if (
            not isinstance(self.turn_index, int)
            or isinstance(self.turn_index, bool)
            or self.turn_index < 0
        ):
            raise ValueError(
                "tool orchestration turn_index must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class ToolOrchestrationResult:
    """Terminal result for one work item."""

    item: ToolWorkItem
    status: ToolExecutionStatus
    output: ToolResult | ToolOutcome | None = None
    error: BaseException | None = None
    cancel_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item, ToolWorkItem):
            raise TypeError("tool orchestration result item must be a ToolWorkItem")
        if self.status not in {"completed", "failed", "cancelled", "skipped"}:
            raise ValueError(f"invalid tool orchestration status: {self.status!r}")
        if self.output is not None and not isinstance(
            self.output,
            (ToolResult, ToolOutcome),
        ):
            raise TypeError(
                "tool orchestration output must be a tool result or outcome"
            )
        if self.error is not None and not isinstance(self.error, BaseException):
            raise TypeError("tool orchestration error must be an exception or None")
        if self.cancel_reason is not None and (
            not isinstance(self.cancel_reason, str) or not self.cancel_reason
        ):
            raise TypeError(
                "tool orchestration cancel_reason must be non-empty or None"
            )
        if self.status == "completed":
            if (
                self.output is None
                or self.error is not None
                or self.cancel_reason is not None
            ):
                raise ValueError("completed tool orchestration requires only an output")
            return
        if self.status == "failed":
            if (
                self.error is None
                or self.output is not None
                or self.cancel_reason is not None
            ):
                raise ValueError("failed tool orchestration requires only an error")
            return
        if (
            self.output is not None
            or self.error is not None
            or self.cancel_reason is None
        ):
            raise ValueError(
                f"{self.status} tool orchestration requires only a cancel reason"
            )


@runtime_checkable
class ToolOrchestrator(Protocol):
    """Replaceable scheduler for batches of tool calls."""

    def stream_batch(
        self,
        request: ToolOrchestrationRequest,
        *,
        signal: CancelSignal | None = None,
        executor: ToolExecutor | None = None,
    ) -> AsyncIterator[ToolOrchestrationResult]:
        """Yield each request item exactly once as soon as it terminates.

        Results must reference the original ``ToolWorkItem`` object from
        ``request.items``. Completion order is intentionally unconstrained;
        the runtime restores assistant call order for provider context.
        """
        ...


__all__ = [
    "ToolExecutionStatus",
    "ToolOrchestrationRequest",
    "ToolOrchestrationResult",
    "ToolOrchestrator",
    "ToolWorkItem",
]
