"""Runtime helpers for message-level trajectory nodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import cast

from agentm.core.abi.context import turn_to_messages
from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.store import TrajectoryNodeQuery
from agentm.core.abi.trajectory import (
    ContentReplacementState,
    PromptCacheState,
    TRAJECTORY_NODE_INDEXES,
    TrajectoryIndexSpec,
    TrajectoryLeaf,
    TrajectoryNode,
    TrajectoryNodeRole,
    Turn,
)
from agentm.core.abi.trigger import TriggerRenderer


def messages_to_nodes(
    *,
    messages: Sequence[AgentMessage],
    session_id: str,
    node_id_prefix: str,
    start_seq: int = 0,
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
    parent_node_id: str | None = None,
    logical_parent_id: str | None = None,
    turn_id: str | None = None,
    turn_index: int | None = None,
    agent_id: str | None = None,
    is_sidechain: bool = False,
    timestamp: float = 0.0,
) -> list[TrajectoryNode]:
    """Convert linear messages into linked trajectory nodes."""

    nodes: list[TrajectoryNode] = []
    parent = parent_node_id
    for offset, message in enumerate(messages):
        msg_role = getattr(message, "role", "control")
        role = (
            cast(TrajectoryNodeRole, msg_role)
            if msg_role in {"user", "assistant", "tool_result"}
            else "control"
        )
        node = TrajectoryNode(
            id=f"{node_id_prefix}:{offset}",
            session_id=session_id,
            seq=start_seq + offset,
            kind="message",
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            role=role,
            parent_id=parent,
            logical_parent_id=logical_parent_id if offset == 0 else None,
            turn_id=turn_id,
            turn_index=turn_index,
            message_index=offset,
            agent_id=agent_id,
            is_sidechain=is_sidechain,
            message=message,
            timestamp=timestamp,
        )
        nodes.append(node)
        parent = node.id
    return nodes


def turn_to_nodes(
    turn: Turn,
    *,
    session_id: str,
    start_seq: int = 0,
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
    parent_node_id: str | None = None,
    logical_parent_id: str | None = None,
    agent_id: str | None = None,
    is_sidechain: bool = False,
    renderers: dict[str, TriggerRenderer] | None = None,
) -> list[TrajectoryNode]:
    """Project one committed turn into message-level linked nodes."""

    return messages_to_nodes(
        messages=turn_to_messages(turn, renderers),
        session_id=session_id,
        node_id_prefix=f"turn:{turn.id}",
        start_seq=start_seq,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
        parent_node_id=parent_node_id,
        logical_parent_id=logical_parent_id,
        turn_id=turn.id,
        turn_index=turn.index,
        agent_id=agent_id,
        is_sidechain=is_sidechain,
        timestamp=turn.timestamp,
    )


def build_chain(
    nodes: Iterable[TrajectoryNode],
    leaf_node_id: str,
    *,
    include_logical_parent: bool = False,
) -> list[TrajectoryNode]:
    """Reconstruct a visible chain by following parent links."""

    by_id = {node.id: node for node in nodes}
    removed = {
        removed_id
        for node in by_id.values()
        if node.kind == "snip"
        for removed_id in node.removed_node_ids
    }
    chain: list[TrajectoryNode] = []
    current = by_id.get(leaf_node_id)
    while current is not None:
        if current.id not in removed:
            chain.append(current)
        if current.kind == "compact_boundary" and not include_logical_parent:
            break
        next_id = current.parent_id
        if next_id is None and include_logical_parent:
            next_id = current.logical_parent_id
        current = by_id.get(next_id) if next_id else None
    chain.reverse()
    return chain


def leaf_nodes(nodes: Iterable[TrajectoryNode]) -> list[TrajectoryLeaf]:
    """Return nodes with no visible children."""

    materialized = list(nodes)
    removed = {
        removed_id
        for node in materialized
        if node.kind == "snip"
        for removed_id in node.removed_node_ids
    }
    parents = {
        node.parent_id
        for node in materialized
        if node.parent_id is not None and node.id not in removed
    }
    return [
        TrajectoryLeaf(
            session_id=node.session_id,
            node_id=node.id,
            seq=node.seq,
            agent_id=node.agent_id,
            is_sidechain=node.is_sidechain,
        )
        for node in materialized
        if node.kind in {"message", "compact_boundary"}
        and node.id not in removed
        and node.id not in parents
    ]


class InMemoryTrajectoryNodeStore:
    """Reference in-memory implementation of ``TrajectoryNodeStore``."""

    def __init__(self) -> None:
        self._nodes: dict[str, list[TrajectoryNode]] = {}
        self._node_ids: set[str] = set()
        self._content_states: dict[tuple[str, str], ContentReplacementState] = {}
        self._prompt_cache_states: dict[tuple[str, str], PromptCacheState] = {}

    @property
    def indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_NODE_INDEXES

    def append_nodes(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
    ) -> None:
        if not nodes:
            return
        current = self._nodes.get(session_id, [])
        expected = current[-1].seq + 1 if current else 0
        copied = list(nodes)
        batch_ids: set[str] = set()
        for node in copied:
            if node.session_id != session_id:
                raise ValueError("node session_id does not match append session")
            if node.seq != expected:
                raise ValueError(f"node seq {node.seq} does not follow {expected - 1}")
            if node.id in self._node_ids or node.id in batch_ids:
                raise ValueError(f"duplicate trajectory node id: {node.id}")
            batch_ids.add(node.id)
            expected += 1
        self._nodes.setdefault(session_id, []).extend(copied)
        self._node_ids.update(batch_ids)

    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        if query.session_id:
            nodes = list(self._nodes.get(query.session_id, ()))
        else:
            nodes = [
                node
                for session_nodes in self._nodes.values()
                for node in session_nodes
            ]
        if query.node_id is not None:
            nodes = [node for node in nodes if node.id == query.node_id]
        if query.root_session_id is not None:
            nodes = [
                node for node in nodes if node.root_session_id == query.root_session_id
            ]
        if query.parent_session_id is not None:
            nodes = [
                node
                for node in nodes
                if node.parent_session_id == query.parent_session_id
            ]
        if query.agent_id is not None:
            nodes = [node for node in nodes if node.agent_id == query.agent_id]
        if query.is_sidechain is not None:
            nodes = [
                node for node in nodes if node.is_sidechain == query.is_sidechain
            ]
        if query.kinds:
            nodes = [node for node in nodes if node.kind in query.kinds]
        if query.parent_id is not None:
            nodes = [node for node in nodes if node.parent_id == query.parent_id]
        if query.logical_parent_id is not None:
            nodes = [
                node for node in nodes
                if node.logical_parent_id == query.logical_parent_id
            ]
        if query.turn_id is not None:
            nodes = [node for node in nodes if node.turn_id == query.turn_id]
        if query.turn_index is not None:
            nodes = [node for node in nodes if node.turn_index == query.turn_index]
        if query.after_seq is not None:
            nodes = [node for node in nodes if node.seq > query.after_seq]
        if query.before_seq is not None:
            nodes = [node for node in nodes if node.seq < query.before_seq]
        nodes.sort(key=lambda node: node.seq, reverse=query.sort == "desc")
        if query.limit is not None:
            nodes = nodes[: query.limit]
        return nodes

    def load_chain(
        self,
        session_id: str,
        leaf_node_id: str,
        *,
        include_logical_parent: bool = False,
    ) -> list[TrajectoryNode]:
        nodes: Iterable[TrajectoryNode]
        if include_logical_parent:
            nodes = (
                node
                for session_nodes in self._nodes.values()
                for node in session_nodes
            )
        else:
            nodes = self._nodes.get(session_id, ())
        return build_chain(
            nodes,
            leaf_node_id,
            include_logical_parent=include_logical_parent,
        )

    def leaves(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> list[TrajectoryLeaf]:
        nodes = self.query_nodes(
            TrajectoryNodeQuery(
                session_id=session_id,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
            )
        )
        return leaf_nodes(nodes)

    def save_content_replacement_state(
        self,
        session_id: str,
        state: ContentReplacementState,
    ) -> None:
        self._content_states[(session_id, state.state_key)] = state

    def load_content_replacement_state(
        self,
        session_id: str,
        state_key: str,
    ) -> ContentReplacementState | None:
        return self._content_states.get((session_id, state_key))

    def clone_content_replacement_state(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        state_key: str,
        target_leaf_id: str | None = None,
    ) -> ContentReplacementState | None:
        state = self.load_content_replacement_state(source_session_id, state_key)
        if state is None:
            return None
        cloned = replace(
            state,
            source_session_id=source_session_id,
            source_leaf_id=target_leaf_id,
        )
        self.save_content_replacement_state(target_session_id, cloned)
        return cloned

    def save_prompt_cache_state(
        self,
        session_id: str,
        state: PromptCacheState,
    ) -> None:
        self._prompt_cache_states[(session_id, state.cache_key)] = state

    def load_prompt_cache_state(
        self,
        session_id: str,
        cache_key: str,
    ) -> PromptCacheState | None:
        return self._prompt_cache_states.get((session_id, cache_key))


__all__ = [
    "InMemoryTrajectoryNodeStore",
    "build_chain",
    "leaf_nodes",
    "messages_to_nodes",
    "turn_to_nodes",
]
