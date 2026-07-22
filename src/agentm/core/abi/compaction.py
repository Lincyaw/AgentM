"""Context projection ABI.

Trajectory is the durable source of truth. Projection decides which messages
derived from committed turns are sent to the model under a context budget.
Compaction is one projection strategy, not a replacement history model.
"""

# code-health: ignore-file[AM025] -- validates immutable ABI DTO boundaries

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import AgentMessage, JsonValue, freeze_json
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


@dataclass(frozen=True, slots=True)
class TurnRange:
    """Inclusive turn-index range represented in a projection decision."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CompactionRequest:
    """Store-driven request to derive a summary from committed history.

    Implementation-specific policy belongs in ``strategy`` and ``options``;
    the orchestration contract remains independent of any LLM, prompt, or
    execution backend.
    """

    source_session_id: str
    through_turn_id: str | None = None
    start_after_turn_id: str | None = None
    previous_summary: str | None = None
    strategy: str = "llm_structured_checkpoint"
    options: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, required_value in (
            ("source_session_id", self.source_session_id),
            ("strategy", self.strategy),
        ):
            if not isinstance(required_value, str) or not required_value:
                raise ValueError(f"compaction request {label} must be non-empty")
        for label, optional_value in (
            ("through_turn_id", self.through_turn_id),
            ("start_after_turn_id", self.start_after_turn_id),
            ("previous_summary", self.previous_summary),
        ):
            if optional_value is not None and (
                not isinstance(optional_value, str) or not optional_value
            ):
                raise ValueError(
                    f"compaction request {label} must be non-empty when set"
                )
        if (self.start_after_turn_id is None) != (self.previous_summary is None):
            raise ValueError(
                "incremental compaction requires both start_after_turn_id "
                "and previous_summary"
            )
        frozen_options = freeze_json(self.options)
        frozen_metadata = freeze_json(self.metadata)
        if not isinstance(frozen_options, Mapping) or not isinstance(
            frozen_metadata, Mapping
        ):
            raise TypeError("compaction request options and metadata must be objects")
        object.__setattr__(self, "options", frozen_options)
        object.__setattr__(self, "metadata", frozen_metadata)


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Auditable summary artifact produced from one committed prefix."""

    source_session_id: str
    covered: TurnRange
    covered_through_turn_id: str
    summary: str
    producer_ref: str
    resource_ref: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("source_session_id", self.source_session_id),
            ("covered_through_turn_id", self.covered_through_turn_id),
            ("summary", self.summary),
            ("producer_ref", self.producer_ref),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"compaction result {label} must be non-empty")
        if not isinstance(self.covered, TurnRange):
            raise TypeError("compaction result covered must be a TurnRange")
        if self.resource_ref is not None and (
            not isinstance(self.resource_ref, str) or not self.resource_ref
        ):
            raise ValueError(
                "compaction result resource_ref must be non-empty when set"
            )
        frozen_metadata = freeze_json(self.metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise TypeError("compaction result metadata must be an object")
        object.__setattr__(self, "metadata", frozen_metadata)


@runtime_checkable
class SessionCompactor(Protocol):
    """Generate a summary artifact without mutating the source session."""

    async def compact(
        self,
        request: CompactionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> CompactionResult: ...


@runtime_checkable
class CompactionPublisher(Protocol):
    """Publish a generated artifact for subsequent context projection."""

    async def publish(
        self,
        result: CompactionResult,
        *,
        signal: CancelSignal | None = None,
    ) -> CompactionResult: ...


@dataclass(frozen=True, slots=True)
class ProjectionInput:
    """Provider/context projection input with optional node-chain precision.

    ``turns`` is the authoritative committed turn prefix. ``nodes`` is an
    optional committed message chain from ``TrajectoryStore`` for projections
    that need message-level replay, compact-boundary traversal, sidechain
    filtering, prompt-cache identity, or content references. Projection changes
    provider input only; it does not create external-world snapshots or make a
    mid-turn message an executable fork boundary.
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
    """Project one explicitly selected trajectory view into model context."""

    @property
    def source(self) -> ProjectionSource:
        """Trajectory view the runtime must materialize for this projection."""

        ...

    def project(
        self,
        projection_input: ProjectionInput,
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]: ...

    def explain(self) -> ProjectionReport: ...


@runtime_checkable
class ContextCompactionService(Protocol):
    """Schedule and execute compaction at a driver step boundary."""

    @property
    def pending(self) -> bool: ...

    def request(self) -> None:
        """Schedule compaction after the active step without interrupting it."""
        ...

    async def execute(
        self,
        turns: Sequence[Turn],
        *,
        signal: CancelSignal | None = None,
    ) -> ProjectionReport | None:
        """Consume a pending request; called by the session driver only."""
        ...


__all__ = [
    "CompactionPublisher",
    "CompactionRequest",
    "CompactionResult",
    "ContextBudget",
    "ContextCompactionService",
    "ContextProjection",
    "ProjectionInput",
    "ProjectionReport",
    "ProjectionSource",
    "SessionCompactor",
    "TurnRange",
]
