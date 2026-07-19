"""Concrete context projection policies for trajectory-node replay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel

from agentm.core.abi.compaction import (
    ContextBudget,
    ContextProjection,
    ProjectionInput,
    ProjectionReport,
    TurnRange,
)
from agentm.core.abi.context import turn_to_messages
from agentm.core.abi.manifest import AtomInstallPriority
from agentm.core.abi.messages import (
    AgentMessage,
    MessageMeta,
    TextContent,
    UserMessage,
)
from agentm.core.abi.roles import CONTEXT_PROJECTION_SERVICE
from agentm.core.abi.session_api import AtomAPI
from agentm.core.abi.trajectory import TrajectoryNode, Turn
from agentm.extensions import ExtensionManifest


class ContextProjectionConfig(BaseModel):
    mode: Literal["exact_node_chain", "tail"] = "exact_node_chain"
    max_messages: int | None = None
    include_hidden: bool = True
    include_replay_only: bool = True
    include_sidechain: bool = False
    metadata_only_content_refs: bool = True


MANIFEST = ExtensionManifest(
    name="context_projection",
    description="Register concrete context projection strategies.",
    registers=("context_policy:projection", "service:context_projection"),
    config_schema=ContextProjectionConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
)


class ExactNodeChainProjection(ContextProjection):
    """Replay provider context from the exact trajectory node chain."""

    def __init__(
        self,
        *,
        max_messages: int | None = None,
        include_hidden: bool = True,
        include_replay_only: bool = True,
        include_sidechain: bool = False,
        metadata_only_content_refs: bool = True,
    ) -> None:
        self._max_messages = max_messages
        self._include_hidden = include_hidden
        self._include_replay_only = include_replay_only
        self._include_sidechain = include_sidechain
        self._metadata_only_content_refs = metadata_only_content_refs
        self._report = ProjectionReport(source="node_chain")

    def project(
        self,
        turns: Sequence[Turn],
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]:
        messages: list[AgentMessage] = []
        for turn in turns:
            messages.extend(turn_to_messages(turn))
        limited, dropped = _limit_messages(
            messages,
            self._max_messages if self._max_messages is not None else budget.max_messages,
        )
        self._report = ProjectionReport(
            source="turns",
            kept=_turn_range(turns) if not dropped else (),
            dropped=_turn_range(turns) if dropped else (),
            metadata={"message_count": len(limited)},
        )
        return limited

    def project_chain(
        self,
        projection_input: ProjectionInput,
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]:
        messages: list[AgentMessage] = []
        content_refs: list[str] = []
        cache_keys: list[str] = []
        synthetic_count = 0
        for node in projection_input.nodes:
            if node.is_sidechain and not self._include_sidechain:
                continue
            if node.content_ref is not None:
                content_refs.append(node.content_ref)
            if node.cache_key is not None:
                cache_keys.append(node.cache_key)
            message = _node_message(
                node,
                include_hidden=self._include_hidden,
                include_replay_only=self._include_replay_only,
                metadata_only_content_refs=self._metadata_only_content_refs,
            )
            if message is None:
                continue
            if message.meta.synthetic:
                synthetic_count += 1
            messages.append(message)

        limit = self._max_messages if self._max_messages is not None else budget.max_messages
        limited, dropped = _limit_messages(messages, limit)
        self._report = ProjectionReport(
            source="node_chain",
            session_id=projection_input.session_id,
            branch_id=projection_input.branch_id,
            head_id=projection_input.head_id,
            leaf_node_id=projection_input.leaf_node_id,
            kept=_turn_ranges_from_nodes(projection_input.nodes),
            dropped=tuple(
                TurnRange(start=-1, end=-1) for _ in range(1 if dropped else 0)
            ),
            content_refs=tuple(dict.fromkeys(content_refs)),
            cache_keys=tuple(dict.fromkeys(cache_keys)),
            synthetic_message_count=synthetic_count,
            metadata={
                "message_count": len(limited),
                "node_count": len(projection_input.nodes),
            },
        )
        return limited

    def explain(self) -> ProjectionReport:
        return self._report


class TailContextProjection(ContextProjection):
    """Turn-compatible tail projection with explainable truncation."""

    def __init__(self, *, max_messages: int | None = None) -> None:
        self._max_messages = max_messages
        self._report = ProjectionReport(source="turns")

    def project(
        self,
        turns: Sequence[Turn],
        budget: ContextBudget,
    ) -> Sequence[AgentMessage]:
        messages: list[AgentMessage] = []
        for turn in turns:
            messages.extend(turn_to_messages(turn))
        limited, dropped = _limit_messages(
            messages,
            self._max_messages if self._max_messages is not None else budget.max_messages,
        )
        self._report = ProjectionReport(
            source="turns",
            kept=_turn_range(turns) if not dropped else (),
            dropped=_turn_range(turns) if dropped else (),
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


def _node_message(
    node: TrajectoryNode,
    *,
    include_hidden: bool,
    include_replay_only: bool,
    metadata_only_content_refs: bool,
) -> AgentMessage | None:
    if node.kind == "snip":
        return None
    message = node.message
    if message is not None:
        meta = message.meta
        if meta.replay == "skip":
            return None
        if meta.visibility == "hidden" and not include_hidden:
            return None
        if meta.visibility == "replay_only" and not include_replay_only:
            return None
        if node.cache_key is not None or node.content_ref is not None:
            tags = dict(meta.tags)
            if node.cache_key is not None:
                tags.setdefault("cache_key", node.cache_key)
            if node.content_ref is not None:
                tags.setdefault("content_ref", node.content_ref)
            message = replace(message, meta=replace(meta, tags=tags))
        return message
    if node.content_ref is None or not metadata_only_content_refs:
        return None
    return UserMessage(
        role="user",
        content=[
            TextContent(
                type="text",
                text=f"[content_ref:{node.content_ref}]",
            )
        ],
        timestamp=node.timestamp,
        meta=MessageMeta(
            synthetic=True,
            synthetic_kind=node.kind,
            origin="trajectory",
            visibility="replay_only",
            token_accounting="metadata_only",
            replay="metadata_only",
            tags={"content_ref": node.content_ref},
        ),
    )


def _limit_messages(
    messages: list[AgentMessage],
    limit: int | None,
) -> tuple[list[AgentMessage], bool]:
    if limit is None or limit <= 0 or len(messages) <= limit:
        return messages, False
    return messages[-limit:], True


def _turn_range(turns: Sequence[Turn]) -> tuple[TurnRange, ...]:
    if not turns:
        return ()
    return (TurnRange(start=turns[0].index, end=turns[-1].index),)


def _turn_ranges_from_nodes(nodes: Sequence[TrajectoryNode]) -> tuple[TurnRange, ...]:
    indexes = [node.turn_index for node in nodes if node.turn_index is not None]
    if not indexes:
        return ()
    return (TurnRange(start=min(indexes), end=max(indexes)),)


__all__ = [
    "ExactNodeChainProjection",
    "TailContextProjection",
    "install",
    "MANIFEST",
]
