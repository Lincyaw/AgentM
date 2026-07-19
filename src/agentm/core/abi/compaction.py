"""Context projection ABI.

Trajectory is the durable source of truth. Projection decides which messages
derived from committed turns are sent to the model under a context budget.
Compaction is one projection strategy, not a replacement history model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    TrajectoryBranchId,
    TrajectoryHeadId,
    TrajectoryNode,
    Turn,
)


ProjectionSource = Literal["turns", "node_chain"]


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
class ProjectionInput:
    """Provider/context projection input with optional node-chain precision.

    ``turns`` is the authoritative committed turn prefix. ``nodes`` is an
    optional message-level chain from ``TrajectoryNodeStore`` for projections
    that need exact mid-turn replay, compact-boundary traversal, sidechain
    filtering, prompt-cache identity, or content references.
    """

    turns: Sequence[Turn] = field(default_factory=tuple)
    nodes: Sequence[TrajectoryNode] = field(default_factory=tuple)
    source: ProjectionSource = "turns"
    session_id: str = ""
    root_session_id: str | None = None
    parent_session_id: str | None = None
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    leaf_node_id: str | None = None
    logical_parent_id: str | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ProjectionReport:
    """Explainable metadata for the last projection decision."""

    source: ProjectionSource = "turns"
    session_id: str = ""
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    leaf_node_id: str | None = None
    kept: tuple[TurnRange, ...] = ()
    summarized: tuple[TurnRange, ...] = ()
    dropped: tuple[TurnRange, ...] = ()
    content_refs: tuple[str, ...] = ()
    cache_keys: tuple[str, ...] = ()
    synthetic_message_count: int = 0
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@runtime_checkable
class ContextProjection(Protocol):
    """Project committed turns into provider-bound messages.

    This turn-only method is the compatibility floor. Projections that need
    exact node-chain replay should also implement ``NodeChainContextProjection``.
    """

    def project(
        self,
        turns: Sequence[Turn],
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]:
        ...

    def explain(self) -> ProjectionReport:
        ...


@runtime_checkable
class NodeChainContextProjection(Protocol):
    """Optional exact-replay projection over ``ProjectionInput.nodes``."""

    def project_chain(
        self,
        projection_input: ProjectionInput,
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]:
        ...


def supports_node_chain_projection(projection: object) -> bool:
    """Return whether a projection consumes message-level trajectory chains."""

    return callable(getattr(projection, "project_chain", None))


def project_context(
    projection: ContextProjection,
    projection_input: ProjectionInput,
    budget: ContextBudget,
) -> Sequence[AgentMessage]:
    """Run a projection through the most precise protocol it implements."""

    if isinstance(projection, NodeChainContextProjection):
        return projection.project_chain(projection_input, budget)
    return projection.project(projection_input.turns, budget)


__all__ = [
    "ContextBudget",
    "ContextProjection",
    "NodeChainContextProjection",
    "ProjectionInput",
    "ProjectionReport",
    "ProjectionSource",
    "project_context",
    "supports_node_chain_projection",
    "TurnRange",
]
