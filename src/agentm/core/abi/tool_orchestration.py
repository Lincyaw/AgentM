"""Batch tool orchestration ABI.

``ToolExecutor`` executes one call. ``ToolOrchestrator`` owns the scheduling
policy across a batch of calls emitted by one assistant message: exclusive vs
parallel-safe grouping, cooperative cancellation, and sibling-error cascading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import ToolCallBlock
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


@dataclass(frozen=True, slots=True)
class ToolOrchestrationRequest:
    """A batch of resolved tool calls from one assistant response."""

    items: tuple[ToolWorkItem, ...]
    session_id: str = ""
    turn_id: str = ""
    turn_index: int = 0


@dataclass(frozen=True, slots=True)
class ToolOrchestrationResult:
    """Terminal result for one work item."""

    item: ToolWorkItem
    status: ToolExecutionStatus
    output: ToolResult | ToolOutcome | None = None
    error: BaseException | None = None
    cancel_reason: str | None = None


@runtime_checkable
class ToolOrchestrator(Protocol):
    """Replaceable scheduler for batches of tool calls."""

    async def execute_batch(
        self,
        request: ToolOrchestrationRequest,
        *,
        signal: CancelSignal | None = None,
        executor: ToolExecutor | None = None,
    ) -> Sequence[ToolOrchestrationResult]:
        ...


__all__ = [
    "ToolExecutionStatus",
    "ToolOrchestrationRequest",
    "ToolOrchestrationResult",
    "ToolOrchestrator",
    "ToolWorkItem",
]
