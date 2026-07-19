"""Context projection ABI.

Trajectory is the durable source of truth. Projection decides which messages
derived from committed turns are sent to the model under a context budget.
Compaction is one projection strategy, not a replacement history model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.trajectory import Turn


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Provider-facing budget available to a context projection."""

    max_messages: int | None = None
    max_input_tokens: int | None = None
    reserved_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TurnRange:
    """Inclusive turn-index range represented in a projection decision."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ProjectionReport:
    """Explainable metadata for the last projection decision."""

    kept: tuple[TurnRange, ...] = ()
    summarized: tuple[TurnRange, ...] = ()
    dropped: tuple[TurnRange, ...] = ()
    synthetic_message_count: int = 0
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@runtime_checkable
class ContextProjection(Protocol):
    """Project committed turns into provider-bound messages."""

    def project(
        self,
        turns: Sequence[Turn],
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]:
        ...

    def explain(self) -> ProjectionReport:
        ...


__all__ = [
    "ContextBudget",
    "ContextProjection",
    "ProjectionReport",
    "TurnRange",
]
