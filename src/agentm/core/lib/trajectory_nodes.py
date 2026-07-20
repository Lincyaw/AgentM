"""Backend-neutral helpers and reference storage for trajectory nodes."""

from __future__ import annotations

from _thread import RLock
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from typing import Concatenate, ParamSpec, Protocol, TypeVar

from agentm.core.abi.context import render_trigger
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)
from agentm.core.abi.store import TrajectoryNodeQuery
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    ContentReplacementState,
    PromptCacheState,
    TRAJECTORY_HEAD_INDEXES,
    TRAJECTORY_NODE_INDEXES,
    TrajectoryBranchId,
    TrajectoryHead,
    TrajectoryHeadAdvance,
    TrajectoryHeadId,
    TrajectoryIndexSpec,
    TrajectoryLeaf,
    TrajectoryNode,
    Turn,
)
from agentm.core.abi.trigger import TriggerRenderer
from agentm.core.lib.trajectory_store import (
    validate_initial_node_state,
    validate_node_append_state,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _TrajectoryStateLockOwner(Protocol):
    _trajectory_state_lock: RLock


_S = TypeVar("_S", bound=_TrajectoryStateLockOwner)


def synchronized_trajectory_state(
    method: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    """Serialize one complete in-memory trajectory-state operation."""

    @wraps(method)
    def synchronized(
        self: _S,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        with self._trajectory_state_lock:
            return method(self, *args, **kwargs)

    return synchronized


@dataclass(frozen=True, slots=True)
class NodeProjectionContext:
    """Stable identity and linkage for one sequence of projected messages."""

    session_id: str
    node_id_prefix: str
    start_seq: int = 0
    root_session_id: str | None = None
    parent_session_id: str | None = None
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    parent_node_id: str | None = None
    logical_parent_id: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    round_indexes: Sequence[int | None] | None = None
    message_index_start: int = 0
    agent_id: str | None = None
    is_sidechain: bool = False
    timestamp: float = 0.0


def messages_to_nodes(
    messages: Sequence[AgentMessage],
    context: NodeProjectionContext,
) -> list[TrajectoryNode]:
    """Convert linear messages into linked trajectory nodes."""

    nodes: list[TrajectoryNode] = []
    parent = context.parent_node_id
    for offset, message in enumerate(messages):
        tool_call_ids, tool_names, cache_key, content_ref = _message_indexes(message)
        node = TrajectoryNode(
            id=f"{context.node_id_prefix}:{offset}",
            session_id=context.session_id,
            seq=context.start_seq + offset,
            kind="message",
            root_session_id=context.root_session_id,
            parent_session_id=context.parent_session_id,
            branch_id=context.branch_id,
            head_id=context.head_id,
            role=message.role,
            parent_id=parent,
            logical_parent_id=(context.logical_parent_id if offset == 0 else None),
            turn_id=context.turn_id,
            turn_index=context.turn_index,
            round_index=(
                context.round_indexes[offset]
                if context.round_indexes is not None
                and offset < len(context.round_indexes)
                else None
            ),
            message_index=context.message_index_start + offset,
            agent_id=context.agent_id,
            is_sidechain=context.is_sidechain,
            tool_call_ids=tool_call_ids,
            tool_names=tool_names,
            cache_key=cache_key,
            content_ref=content_ref,
            visibility=message.meta.visibility,
            message=message,
            timestamp=context.timestamp,
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
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID,
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID,
    parent_node_id: str | None = None,
    logical_parent_id: str | None = None,
    agent_id: str | None = None,
    is_sidechain: bool = False,
    renderers: dict[str, TriggerRenderer] | None = None,
) -> list[TrajectoryNode]:
    """Project one committed turn into message-level linked nodes."""

    indexed_messages = _turn_indexed_messages(turn, renderers)
    messages = [message for message, _ in indexed_messages]
    if not turn.outcome.cause.replayable:
        messages = [
            replace(message, meta=replace(message.meta, replay="skip"))
            for message in messages
        ]
    round_indexes = [round_index for _, round_index in indexed_messages]
    return messages_to_nodes(
        messages,
        NodeProjectionContext(
            session_id=session_id,
            node_id_prefix=f"session:{session_id}:turn:{turn.id}",
            start_seq=start_seq,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            branch_id=branch_id,
            head_id=head_id,
            parent_node_id=parent_node_id,
            logical_parent_id=logical_parent_id,
            turn_id=turn.id,
            turn_index=turn.index,
            round_indexes=round_indexes,
            agent_id=agent_id,
            is_sidechain=is_sidechain,
            timestamp=turn.timestamp,
        ),
    )


def turns_to_nodes(
    turns: Sequence[Turn],
    *,
    session_id: str,
    start_seq: int = 0,
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID,
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID,
    parent_node_id: str | None = None,
    logical_parent_id: str | None = None,
    agent_id: str | None = None,
    is_sidechain: bool = False,
    renderers: dict[str, TriggerRenderer] | None = None,
) -> list[TrajectoryNode]:
    """Project committed turns into one linked message-node chain."""

    nodes: list[TrajectoryNode] = []
    next_seq = start_seq
    parent = parent_node_id
    logical_parent = logical_parent_id
    for turn in turns:
        projected = turn_to_nodes(
            turn,
            session_id=session_id,
            start_seq=next_seq,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            branch_id=branch_id,
            head_id=head_id,
            parent_node_id=parent,
            logical_parent_id=logical_parent,
            agent_id=agent_id,
            is_sidechain=is_sidechain,
            renderers=renderers,
        )
        if not projected:
            continue
        nodes.extend(projected)
        next_seq = projected[-1].seq + 1
        parent = projected[-1].id
        logical_parent = None
    return nodes


def _turn_indexed_messages(
    turn: Turn,
    renderers: dict[str, TriggerRenderer] | None,
) -> list[tuple[AgentMessage, int | None]]:
    messages: list[tuple[AgentMessage, int | None]] = []
    messages.extend(
        (message, None) for message in render_trigger(turn.trigger, renderers)
    )

    injected_by_round: dict[int, list[AgentMessage]] = {}
    for injection in turn.outcome.injected:
        injected_by_round.setdefault(injection.after_round, []).extend(
            injection.messages
        )
    messages.extend((message, None) for message in injected_by_round.get(-1, ()))
    for round_index, rnd in enumerate(turn.rounds):
        messages.append((rnd.response, round_index))
        if rnd.tool_results:
            result_blocks = [
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id=tr.call.id,
                    content=list(tr.result.content),
                    is_error=tr.result.is_error,
                    deterministic=tr.result.deterministic,
                    extras=tr.result.extras,
                )
                for tr in rnd.tool_results
            ]
            messages.append(
                (
                    ToolResultMessage(
                        role="tool_result",
                        content=result_blocks,
                        timestamp=0.0,
                    ),
                    round_index,
                )
            )
        messages.extend(
            (message, round_index) for message in injected_by_round.get(round_index, ())
        )
    return messages


def _message_indexes(
    message: AgentMessage,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None, str | None]:
    tool_call_ids: list[str] = []
    tool_names: list[str] = []
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolCallBlock):
                tool_call_ids.append(block.id)
                tool_names.append(block.name)
    elif isinstance(message, ToolResultMessage):
        tool_call_ids.extend(block.tool_call_id for block in message.content)

    tags = message.meta.tags
    cache_key = tags.get("cache_key")
    content_ref = tags.get("content_ref")
    return (
        tuple(tool_call_ids),
        tuple(tool_names),
        cache_key if isinstance(cache_key, str) else None,
        content_ref if isinstance(content_ref, str) else None,
    )


def build_chain(
    nodes: Iterable[TrajectoryNode],
    leaf_node_id: str,
    *,
    include_logical_parent: bool = False,
) -> list[TrajectoryNode]:
    """Reconstruct a visible chain by following parent links."""

    by_id = {node.id: node for node in nodes}
    chain: list[TrajectoryNode] = []
    current = by_id.get(leaf_node_id)
    visited: set[str] = set()
    while current is not None:
        if current.id in visited:
            raise ValueError(f"trajectory node parent cycle includes {current.id}")
        visited.add(current.id)
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
    parents = {node.parent_id for node in materialized if node.parent_id is not None}
    return [
        TrajectoryLeaf(
            session_id=node.session_id,
            node_id=node.id,
            seq=node.seq,
            branch_id=node.branch_id,
            head_id=node.head_id,
            agent_id=node.agent_id,
            is_sidechain=node.is_sidechain,
        )
        for node in materialized
        if node.kind in {"message", "compact_boundary", "system_prompt"}
        and node.id not in parents
    ]


class TrajectoryIndexState:
    """Backend-neutral in-memory state machine for trajectory node indexes."""

    _trajectory_state_lock: RLock

    def __init__(self) -> None:
        self._trajectory_state_lock = RLock()
        self._nodes: dict[str, list[TrajectoryNode]] = {}
        self._node_ids: set[str] = set()
        self._heads: dict[tuple[str, str], TrajectoryHead] = {}
        self._content_states: dict[tuple[str, str], ContentReplacementState] = {}
        self._prompt_cache_states: dict[tuple[str, str], PromptCacheState] = {}

    @property
    def indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_NODE_INDEXES

    @property
    def head_indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        return TRAJECTORY_HEAD_INDEXES

    @synchronized_trajectory_state
    def _commit_nodes(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        advance_head: TrajectoryHeadAdvance | None = None,
    ) -> None:
        if not nodes:
            return
        current = self._nodes.get(session_id, [])
        expected = current[-1].seq + 1 if current else 0
        copied = list(nodes)
        if advance_head is None:
            raise ValueError("trajectory node append requires a head advance")
        current_head = self._heads.get((session_id, advance_head.head_id))
        validate_node_append_state(
            session_id,
            copied,
            advance_head,
            current_head=current_head,
            expected_seq=expected,
        )
        batch_ids: set[str] = set()
        for node in copied:
            if node.id in self._node_ids or node.id in batch_ids:
                raise ValueError(f"duplicate trajectory node id: {node.id}")
            batch_ids.add(node.id)
        self._validate_head_advance(session_id, advance_head, batch_ids)
        self._nodes.setdefault(session_id, []).extend(copied)
        self._node_ids.update(batch_ids)
        self._heads[(session_id, advance_head.head_id)] = advance_head.to_head()

    @synchronized_trajectory_state
    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        if query.session_id:
            self._require_index_session(query.session_id)
            nodes = list(self._nodes.get(query.session_id, ()))
        else:
            nodes = [
                node for session_nodes in self._nodes.values() for node in session_nodes
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
        if query.branch_id is not None:
            nodes = [node for node in nodes if node.branch_id == query.branch_id]
        if query.head_id is not None:
            nodes = [node for node in nodes if node.head_id == query.head_id]
        if query.agent_id is not None:
            nodes = [node for node in nodes if node.agent_id == query.agent_id]
        if query.is_sidechain is not None:
            nodes = [node for node in nodes if node.is_sidechain == query.is_sidechain]
        if query.kinds:
            nodes = [node for node in nodes if node.kind in query.kinds]
        if query.role is not None:
            nodes = [node for node in nodes if node.role == query.role]
        if query.parent_id is not None:
            nodes = [node for node in nodes if node.parent_id == query.parent_id]
        if query.logical_parent_id is not None:
            nodes = [
                node
                for node in nodes
                if node.logical_parent_id == query.logical_parent_id
            ]
        if query.turn_id is not None:
            nodes = [node for node in nodes if node.turn_id == query.turn_id]
        if query.turn_index is not None:
            nodes = [node for node in nodes if node.turn_index == query.turn_index]
        if query.round_index is not None:
            nodes = [node for node in nodes if node.round_index == query.round_index]
        if query.message_index is not None:
            nodes = [
                node for node in nodes if node.message_index == query.message_index
            ]
        if query.tool_call_id is not None:
            nodes = [node for node in nodes if query.tool_call_id in node.tool_call_ids]
        if query.tool_name is not None:
            nodes = [node for node in nodes if query.tool_name in node.tool_names]
        if query.cache_key is not None:
            nodes = [node for node in nodes if node.cache_key == query.cache_key]
        if query.content_ref is not None:
            nodes = [node for node in nodes if node.content_ref == query.content_ref]
        if query.visibility is not None:
            nodes = [node for node in nodes if node.visibility == query.visibility]
        if query.after_seq is not None:
            nodes = [node for node in nodes if node.seq > query.after_seq]
        if query.before_seq is not None:
            nodes = [node for node in nodes if node.seq < query.before_seq]
        if query.since_timestamp is not None:
            nodes = [node for node in nodes if node.timestamp >= query.since_timestamp]
        if query.until_timestamp is not None:
            nodes = [node for node in nodes if node.timestamp <= query.until_timestamp]
        nodes.sort(
            key=(
                (lambda node: node.seq)
                if query.session_id
                else (lambda node: (node.session_id, node.seq))
            ),
            reverse=query.sort == "desc",
        )
        if query.limit is not None:
            nodes = nodes[: query.limit]
        return nodes

    @synchronized_trajectory_state
    def get_head(
        self,
        session_id: str,
        *,
        head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> TrajectoryHead | None:
        self._require_index_session(session_id)
        head = self._heads.get((session_id, head_id))
        if head is None:
            return None
        if branch_id is not None and head.branch_id != branch_id:
            return None
        if agent_id is not None and head.agent_id != agent_id:
            return None
        if is_sidechain is not None and head.is_sidechain != is_sidechain:
            return None
        return head

    @synchronized_trajectory_state
    def list_heads(
        self,
        session_id: str,
        *,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
        include_inactive: bool = False,
    ) -> list[TrajectoryHead]:
        self._require_index_session(session_id)
        heads = [
            head
            for (head_session_id, _), head in self._heads.items()
            if head_session_id == session_id
        ]
        if branch_id is not None:
            heads = [head for head in heads if head.branch_id == branch_id]
        if agent_id is not None:
            heads = [head for head in heads if head.agent_id == agent_id]
        if is_sidechain is not None:
            heads = [head for head in heads if head.is_sidechain == is_sidechain]
        if not include_inactive:
            heads = [head for head in heads if head.status == "active"]
        heads.sort(key=lambda head: (head.updated_at, head.head_id))
        return heads

    @synchronized_trajectory_state
    def load_chain(
        self,
        session_id: str,
        leaf_node_id: str,
        *,
        include_logical_parent: bool = False,
    ) -> list[TrajectoryNode]:
        self._require_index_session(session_id)
        local_leaf = any(node.id == leaf_node_id for node in self._nodes[session_id])
        inherited_leaf = include_logical_parent and any(
            head_session_id == session_id and head.logical_parent_id == leaf_node_id
            for (head_session_id, _head_id), head in self._heads.items()
        )
        if not local_leaf and not inherited_leaf:
            raise ValueError(f"unknown trajectory leaf node: {leaf_node_id}")
        nodes: Iterable[TrajectoryNode]
        if include_logical_parent:
            nodes = (
                node for session_nodes in self._nodes.values() for node in session_nodes
            )
        else:
            nodes = self._nodes.get(session_id, ())
        return build_chain(
            nodes,
            leaf_node_id,
            include_logical_parent=include_logical_parent,
        )

    @synchronized_trajectory_state
    def leaves(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> list[TrajectoryLeaf]:
        self._require_index_session(session_id)
        nodes = self.query_nodes(
            TrajectoryNodeQuery(
                session_id=session_id,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
            )
        )
        return leaf_nodes(nodes)

    @synchronized_trajectory_state
    def _initialize_index(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        head: TrajectoryHead,
    ) -> None:
        if session_id in self._nodes or any(
            head_session_id == session_id for head_session_id, _head_id in self._heads
        ):
            raise ValueError(f"trajectory index already exists: {session_id}")
        copied = list(nodes)
        validate_initial_node_state(session_id, copied, head)
        batch_ids: set[str] = set()
        for node in copied:
            if node.id in batch_ids:
                raise ValueError(f"duplicate trajectory node id: {node.id}")
            batch_ids.add(node.id)

        duplicates = batch_ids & self._node_ids
        if duplicates:
            duplicate = sorted(duplicates)[0]
            raise ValueError(f"duplicate trajectory node id: {duplicate}")
        if head.session_id != session_id:
            raise ValueError("head session_id does not match index session")
        self._nodes[session_id] = copied
        self._node_ids.update(batch_ids)
        self._heads[(session_id, head.head_id)] = head

    @synchronized_trajectory_state
    def save_content_replacement_state(
        self,
        session_id: str,
        state: ContentReplacementState,
    ) -> None:
        self._require_index_session(session_id)
        self._content_states[(session_id, state.state_key)] = state

    @synchronized_trajectory_state
    def load_content_replacement_state(
        self,
        session_id: str,
        state_key: str,
    ) -> ContentReplacementState | None:
        self._require_index_session(session_id)
        return self._content_states.get((session_id, state_key))

    @synchronized_trajectory_state
    def clone_content_replacement_state(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        state_key: str,
        target_leaf_id: str | None = None,
    ) -> ContentReplacementState | None:
        self._require_index_session(source_session_id)
        self._require_index_session(target_session_id)
        state = self.load_content_replacement_state(source_session_id, state_key)
        if state is None:
            return None
        cloned = replace(
            state,
            source_session_id=source_session_id,
            source_leaf_id=state.leaf_node_id or state.source_leaf_id,
            leaf_node_id=target_leaf_id,
        )
        self.save_content_replacement_state(target_session_id, cloned)
        return cloned

    @synchronized_trajectory_state
    def save_prompt_cache_state(
        self,
        session_id: str,
        state: PromptCacheState,
    ) -> None:
        self._require_index_session(session_id)
        self._prompt_cache_states[(session_id, state.cache_key)] = state

    @synchronized_trajectory_state
    def load_prompt_cache_state(
        self,
        session_id: str,
        cache_key: str,
    ) -> PromptCacheState | None:
        self._require_index_session(session_id)
        return self._prompt_cache_states.get((session_id, cache_key))

    def _require_index_session(self, session_id: str) -> None:
        if session_id not in self._nodes:
            raise KeyError(session_id)

    def _validate_head_advance(
        self,
        session_id: str,
        advance: TrajectoryHeadAdvance,
        batch_ids: set[str],
    ) -> None:
        if advance.session_id != session_id:
            raise ValueError("head advance session_id does not match append session")
        if advance.node_id not in batch_ids:
            raise ValueError("head advance node_id is not part of the append batch")
        current = self._heads.get((session_id, advance.head_id))
        if current is not None and current.node_id != advance.previous_node_id:
            raise ValueError(
                "trajectory head changed before append: "
                f"{current.node_id!r} != {advance.previous_node_id!r}"
            )
        if current is None and advance.previous_node_id is not None:
            previous_known = any(
                node.id == advance.previous_node_id
                for node in self._nodes.get(session_id, ())
            )
            if not previous_known:
                previous_known = advance.previous_node_id in self._node_ids
            if not previous_known:
                raise ValueError(
                    f"head advance previous_node_id is unknown: "
                    f"{advance.previous_node_id}"
                )


__all__ = [
    "TrajectoryIndexState",
    "synchronized_trajectory_state",
    "build_chain",
    "leaf_nodes",
    "messages_to_nodes",
    "turn_to_nodes",
    "turns_to_nodes",
]
