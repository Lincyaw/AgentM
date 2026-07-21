"""Concrete context projection policies for trajectory-node replay."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentm.core.abi.compaction import (
    ContextBudget,
    ContextProjection,
    ProjectionInput,
    ProjectionReport,
    TurnRange,
)
from agentm.core.abi.context import turn_to_messages
from agentm.core.abi.manifest import AtomInstallPriority
from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.roles import CONTEXT_PROJECTION_SERVICE
from agentm.core.abi.session_api import AtomAPI
from agentm.core.lib.context_projection import ExactNodeChainProjection
from agentm.extensions import ExtensionManifest


class ContextProjectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["exact_node_chain", "tail"] = "exact_node_chain"
    max_messages: int | None = None
    include_hidden: bool = True
    include_replay_only: bool = True
    include_sidechain: bool = False
    metadata_only_content_refs: bool = True


MANIFEST = ExtensionManifest(
    name="context_projection",
    description="Override the runtime's default exact context projection strategy.",
    registers=("service:context_projection",),
    config_schema=ContextProjectionConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
)


class TailContextProjection(ContextProjection):
    """Tail projection over authoritative committed turns."""

    source: Literal["turns"] = "turns"

    def __init__(self, *, max_messages: int | None = None) -> None:
        self._max_messages = max_messages
        self._report = ProjectionReport(source="turns")

    def project(
        self,
        projection_input: ProjectionInput,
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]:
        if projection_input.source != self.source:
            raise ValueError(
                f"tail projection requires source {self.source!r}, "
                f"got {projection_input.source!r}"
            )
        limited, kept_indexes, dropped_indexes = _limit_message_groups(
            [
                (turn.index, list(turn_to_messages(turn)))
                for turn in projection_input.turns
            ],
            self._max_messages
            if self._max_messages is not None
            else budget.max_messages,
        )
        self._report = ProjectionReport(
            source="turns",
            kept=_turn_ranges(kept_indexes),
            dropped=_turn_ranges(dropped_indexes),
            metadata={"message_count": len(limited)},
        )
        return limited

    def explain(self) -> ProjectionReport:
        return self._report


def install(api: AtomAPI, config: ContextProjectionConfig) -> None:
    projection: ContextProjection
    if config.mode == "tail":
        projection = TailContextProjection(max_messages=config.max_messages)
    else:
        projection = ExactNodeChainProjection(
            max_messages=config.max_messages,
            include_hidden=config.include_hidden,
            include_replay_only=config.include_replay_only,
            include_sidechain=config.include_sidechain,
            metadata_only_content_refs=config.metadata_only_content_refs,
        )
    api.services.register(
        CONTEXT_PROJECTION_SERVICE,
        projection,
        ContextProjection,
        scope="session",
    )


def _limit_message_groups(
    groups: Sequence[tuple[int, list[AgentMessage]]],
    limit: int | None,
) -> tuple[list[AgentMessage], tuple[int, ...], tuple[int, ...]]:
    """Apply a soft message limit without splitting a committed turn."""

    total = sum(len(messages) for _, messages in groups)
    if limit is None or limit <= 0 or total <= limit:
        return (
            [message for _, messages in groups for message in messages],
            tuple(index for index, _ in groups),
            (),
        )

    start = len(groups)
    kept_count = 0
    for position in range(len(groups) - 1, -1, -1):
        group_size = len(groups[position][1])
        if kept_count and kept_count + group_size > limit:
            break
        start = position
        kept_count += group_size
        if kept_count >= limit:
            break

    kept_groups = groups[start:]
    dropped_groups = groups[:start]
    return (
        [message for _, messages in kept_groups for message in messages],
        tuple(index for index, _ in kept_groups),
        tuple(index for index, _ in dropped_groups),
    )


def _turn_ranges(indexes: Iterable[int]) -> tuple[TurnRange, ...]:
    ordered = sorted(set(indexes))
    if not ordered:
        return ()
    ranges: list[TurnRange] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index != previous + 1:
            ranges.append(TurnRange(start=start, end=previous))
            start = index
        previous = index
    ranges.append(TurnRange(start=start, end=previous))
    return tuple(ranges)


__all__ = [
    "ExactNodeChainProjection",
    "TailContextProjection",
    "install",
    "MANIFEST",
]
