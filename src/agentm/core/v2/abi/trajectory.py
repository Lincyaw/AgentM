"""Core trajectory data types.

All types here are frozen dataclasses — once a Turn is committed to a
Trajectory it is immutable.

Type hierarchy::

    Turn
      ├── trigger: Trigger    (what caused this turn)
      ├── rounds: tuple[Round, ...]
      │     ├── response: AssistantMessage
      │     └── tool_results: tuple[ToolRecord, ...]
      ├── outcome: Outcome    (step / stop / inject)
      └── meta: TurnMeta      (observability data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    ToolResultBlock,
)

if TYPE_CHECKING:
    from agentm.core.v2.abi.trigger import Trigger

TurnRef = Union[int, str]


@dataclass(frozen=True, slots=True)
class TurnMeta:
    """Observability data attached to a committed Turn."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ns: int = 0
    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolRecord:
    """One tool call and its final result (post all bus hooks)."""

    call: ToolCallBlock
    result: ToolResultBlock
    backgrounded: bool = False


@dataclass(frozen=True, slots=True)
class Round:
    """One LLM call and its consequent tool executions."""

    response: AssistantMessage
    tool_results: tuple[ToolRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class Outcome:
    """The resolved decision at the end of a Turn."""

    action: Literal["step", "stop", "inject"]
    cause: object | None = None
    injected: tuple[AgentMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class Turn:
    """One committed turn in a trajectory.  Immutable after commit."""

    index: int
    id: str
    trigger: Trigger | object
    rounds: tuple[Round, ...]
    outcome: Outcome
    timestamp: float
    meta: TurnMeta = field(default_factory=TurnMeta)


__all__ = [
    "Outcome",
    "Round",
    "ToolRecord",
    "Turn",
    "TurnMeta",
    "TurnRef",
]
