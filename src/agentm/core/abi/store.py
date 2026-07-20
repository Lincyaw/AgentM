"""TrajectoryStore — persistence protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, Sequence, runtime_checkable

from agentm.core.abi.trajectory import (
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
    TrajectoryNodeKind,
    TrajectoryNodeRole,
    Turn,
    TurnCheckpoint,
    TurnRef,
)


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Metadata for a session record in the store.

    ``config`` carries JSON-safe, resumable session context such as root
    session id, depth, scenario name, and scenario-local base directory.
    """

    id: str
    parent_id: str | None = None
    fork_point: TurnRef | None = None
    purpose: str = "root"
    cwd: str = ""
    created_at: float = 0.0
    config: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("session metadata id must be a non-empty string")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id
        ):
            raise ValueError("session metadata parent_id must be a non-empty string")
        if self.fork_point is not None:
            if isinstance(self.fork_point, bool) or not isinstance(
                self.fork_point,
                (str, int),
            ):
                raise TypeError(
                    "session metadata fork_point must be a string or integer"
                )
            if isinstance(self.fork_point, str) and not self.fork_point:
                raise ValueError("session metadata fork_point cannot be empty")
            if isinstance(self.fork_point, int) and self.fork_point < 0:
                raise ValueError("session metadata fork_point cannot be negative")
        if not isinstance(self.purpose, str) or not self.purpose:
            raise ValueError("session metadata purpose must be a non-empty string")
        if not isinstance(self.cwd, str):
            raise TypeError("session metadata cwd must be a string")
        if (
            not isinstance(self.created_at, (int, float))
            or isinstance(self.created_at, bool)
            or not math.isfinite(self.created_at)
        ):
            raise ValueError("session metadata created_at must be a finite number")
        copied_config: dict[str, str | int | float | bool | None] = {}
        for key, value in self.config.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "session metadata config keys must be non-empty strings"
                )
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(
                    f"session metadata config value {key!r} is not a scalar"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"session metadata config value {key!r} must be finite"
                )
            copied_config[key] = value
        object.__setattr__(self, "config", MappingProxyType(copied_config))


@runtime_checkable
class TrajectoryStore(Protocol):
    """Single persistence boundary for trajectory state and indexes.

    ``save_checkpoint`` durably replaces the latest incomplete state for the
    active turn. ``commit_turn`` atomically commits the final turn, its
    provider-visible message nodes, and the matching explicit head advance while
    superseding the checkpoint. ``commit_compaction`` atomically commits a
    compact boundary, its content-replacement state, and its head advance,
    anchored to an existing committed turn. ``load`` and ``load_prefix`` return
    committed turns only, so incomplete work is never replayed on resume.

    Node/head methods are read and policy-state facets of the same selected
    store, not an independently configurable projection backend. Physical
    tables and indexes remain backend implementation details.

    Methods are synchronous blocking ports; async runtimes must offload calls
    instead of running backend I/O on the event loop.
    """

    @property
    def indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        """Return node index/order declarations supported by this store."""
        ...

    @property
    def head_indexes(self) -> tuple[TrajectoryIndexSpec, ...]:
        """Return head index/order declarations supported by this store."""
        ...

    def create_session(
        self,
        meta: SessionMeta,
        *,
        turns: Sequence[Turn] = (),
        nodes: Sequence[TrajectoryNode] = (),
        head: TrajectoryHead,
    ) -> None:
        """Atomically create a session and its initial committed state."""
        ...

    def save_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None: ...

    def load_checkpoint(self, session_id: str) -> TurnCheckpoint | None: ...

    def discard_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        """Compare-and-discard one orphan checkpoint after successful recovery.

        The operation is idempotent when no checkpoint exists and fails when a
        different checkpoint snapshot occupies the session, so recovery cannot
        erase concurrent or otherwise unexpected incomplete work.
        """
        ...

    def commit_turn(self, session_id: str, commit: TrajectoryCommit) -> None:
        """Atomically publish one turn, its nodes, and its head advance."""
        ...

    def commit_compaction(
        self,
        session_id: str,
        commit: TrajectoryCompactionCommit,
    ) -> None:
        """Atomically publish compact boundaries, state, and a head advance."""
        ...

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]: ...

    def load_prefix(
        self, session_id: str, up_to: TurnRef
    ) -> tuple[SessionMeta, list[Turn]]: ...

    def session_children(self, session_id: str) -> list[str]: ...

    def session_exists(self, session_id: str) -> bool: ...

    def list_sessions(self) -> list[SessionMeta]: ...

    def query_nodes(self, query: TrajectoryNodeQuery) -> list[TrajectoryNode]:
        """Return nodes matching a portable query."""
        ...

    def get_head(
        self,
        session_id: str,
        *,
        head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> TrajectoryHead | None:
        """Return the explicit append head for a chain, if one exists."""
        ...

    def list_heads(
        self,
        session_id: str,
        *,
        branch_id: TrajectoryBranchId | None = None,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
        include_inactive: bool = False,
    ) -> list[TrajectoryHead]:
        """Return explicit heads for branch/fork/resume selection."""
        ...

    def load_chain(
        self,
        session_id: str,
        leaf_node_id: str,
        *,
        include_logical_parent: bool = False,
    ) -> list[TrajectoryNode]:
        """Reconstruct one visible chain by following parent links.

        ``include_logical_parent`` lets fork/resume and compact-boundary reads
        continue through ``logical_parent_id`` when the physical parent chain
        intentionally stops.
        """
        ...

    def leaves(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        is_sidechain: bool | None = None,
    ) -> list[TrajectoryLeaf]:
        """Return visible leaf nodes for diagnostics and branch discovery.

        New append paths should use explicit heads, not infer current state
        from leaves.
        """
        ...

    def save_content_replacement_state(
        self,
        session_id: str,
        state: ContentReplacementState,
    ) -> None:
        """Persist prompt-cache/content-replacement state."""
        ...

    def load_content_replacement_state(
        self,
        session_id: str,
        state_key: str,
    ) -> ContentReplacementState | None:
        """Load persisted content-replacement state."""
        ...

    def clone_content_replacement_state(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        state_key: str,
        target_leaf_id: str | None = None,
    ) -> ContentReplacementState | None:
        """Clone state for fork/resume while preserving deterministic decisions."""
        ...

    def save_prompt_cache_state(
        self,
        session_id: str,
        state: PromptCacheState,
    ) -> None:
        """Persist provider prompt-cache identity for a chain prefix."""
        ...

    def load_prompt_cache_state(
        self,
        session_id: str,
        cache_key: str,
    ) -> PromptCacheState | None:
        """Load provider prompt-cache identity for a chain prefix."""
        ...


@dataclass(frozen=True, slots=True)
class TrajectoryCommit:
    """One atomic committed-turn mutation."""

    turn: Turn
    nodes: tuple[TrajectoryNode, ...]
    advance_head: TrajectoryHeadAdvance | None

    def __post_init__(self) -> None:
        if not isinstance(self.turn, Turn):
            raise TypeError("trajectory commit turn must be a Turn")
        if not isinstance(self.nodes, tuple) or not all(
            isinstance(node, TrajectoryNode) for node in self.nodes
        ):
            raise TypeError("trajectory commit nodes must be a tuple of TrajectoryNode")
        for node in self.nodes:
            if node.kind != "message":
                raise ValueError(
                    "trajectory turn commits can contain only message nodes"
                )
            if node.turn_id != self.turn.id or node.turn_index != self.turn.index:
                raise ValueError(
                    "trajectory commit nodes must belong to the committed turn"
                )
        if self.nodes:
            if self.advance_head is None:
                raise ValueError(
                    "trajectory commit with message nodes requires a head advance"
                )
            if self.advance_head.node_id != self.nodes[-1].id:
                raise ValueError(
                    "trajectory commit head must advance to the final message node"
                )
        elif self.advance_head is not None:
            raise ValueError(
                "trajectory commit cannot advance a head without message nodes"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryCompactionCommit:
    """One atomic compaction mutation anchored to committed history.

    Control nodes may change the visible message chain but cannot represent
    external-world progress. Every node therefore names the committed turn
    whose effect snapshot remains authoritative. Policy state is included in
    the same mutation so a head can never expose a compact boundary without
    the state required to interpret it.
    """

    boundary: TrajectoryNode
    advance_head: TrajectoryHeadAdvance
    content_replacement_state: ContentReplacementState

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, TrajectoryNode):
            raise TypeError("trajectory compaction boundary must be a TrajectoryNode")
        if self.boundary.kind != "compact_boundary":
            raise ValueError("trajectory compaction requires a compact boundary")
        if not isinstance(self.advance_head, TrajectoryHeadAdvance):
            raise TypeError(
                "trajectory compaction commit advance_head must be "
                "TrajectoryHeadAdvance"
            )
        if self.advance_head.node_id != self.boundary.id:
            raise ValueError("trajectory compaction head must advance to its boundary")
        if not isinstance(
            self.content_replacement_state,
            ContentReplacementState,
        ):
            raise TypeError(
                "trajectory compaction commit state must be ContentReplacementState"
            )
        if self.content_replacement_state.leaf_node_id != self.boundary.id:
            raise ValueError("trajectory compaction state must identify its boundary")


TrajectoryNodeSort = Literal["asc", "desc"]


@dataclass(frozen=True, slots=True)
class TrajectoryNodeQuery:
    """Portable query shape for committed message indexes."""

    session_id: str = ""
    node_id: str | None = None
    root_session_id: str | None = None
    parent_session_id: str | None = None
    branch_id: TrajectoryBranchId | None = None
    head_id: TrajectoryHeadId | None = None
    agent_id: str | None = None
    is_sidechain: bool | None = None
    kinds: tuple[TrajectoryNodeKind, ...] = ()
    role: TrajectoryNodeRole | None = None
    parent_id: str | None = None
    logical_parent_id: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    round_index: int | None = None
    message_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    cache_key: str | None = None
    content_ref: str | None = None
    visibility: str | None = None
    after_seq: int | None = None
    before_seq: int | None = None
    since_timestamp: float | None = None
    until_timestamp: float | None = None
    limit: int | None = None
    sort: TrajectoryNodeSort = "asc"


__all__ = [
    "SessionMeta",
    "TRAJECTORY_HEAD_INDEXES",
    "TRAJECTORY_NODE_INDEXES",
    "TrajectoryCommit",
    "TrajectoryCompactionCommit",
    "TrajectoryStore",
    "TrajectoryNodeQuery",
    "TrajectoryNodeSort",
]
