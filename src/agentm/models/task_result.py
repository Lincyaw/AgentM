"""TaskResult — generic typed result from a sub-agent execution.

``TaskResult[R]`` carries a typed payload alongside execution metadata.
The type parameter ``R`` is determined by the strategy's
``get_answer_schemas()`` — each task type maps to a Pydantic model
whose ``model_dump()`` output populates ``result``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

R = TypeVar("R")


@dataclass(frozen=True)
class TaskResult(Generic[R]):
    """Typed result from a sub-agent task execution.

    Attributes:
        task_id: Unique identifier for the task.
        agent_id: Identifier of the sub-agent that executed the task.
        status: Terminal status — ``"completed"`` or ``"failed"``.
        result: Typed payload from the sub-agent's structured output.
            ``None`` if the task failed or produced no output.
        error_summary: Human-readable error description if the task failed.
        duration_seconds: Wall-clock duration of the execution.
        metadata: Arbitrary key-value pairs for extensibility.
    """

    task_id: str
    agent_id: str
    status: str
    result: R | None = None
    error_summary: str | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
