"""TrajectoryStore — persistence protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    TrajectoryProjectionStatus,
    Turn,
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
    config: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@runtime_checkable
class TrajectoryStore(Protocol):
    """Persistence boundary for trajectories.

    ``append`` must be atomic — a Turn is either fully written or not.
    Methods are synchronous blocking ports; async runtimes must offload calls
    instead of running backend I/O on the event loop.
    """

    def create_session(self, meta: SessionMeta) -> None: ...

    def create_session_with_turns(
        self, meta: SessionMeta, turns: Sequence[Turn]
    ) -> None: ...

    def append(self, session_id: str, turn: Turn) -> None: ...

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]: ...

    def load_prefix(
        self, session_id: str, up_to: TurnRef
    ) -> tuple[SessionMeta, list[Turn]]: ...

    def session_children(self, session_id: str) -> list[str]: ...

    def session_exists(self, session_id: str) -> bool: ...

    def list_sessions(self) -> list[SessionMeta]: ...


TrajectoryNodeSort = Literal["asc", "desc"]


@dataclass(frozen=True, slots=True)
class TrajectoryNodeQuery:
    """Portable query shape for message-level trajectory node stores."""

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
    limit: int | None = None
    sort: TrajectoryNodeSort = "asc"


@runtime_checkable
class TrajectoryNodeStore(Protocol):
    """Append-only message-level trajectory persistence boundary.

    The node store is the rebuildable projection/read model for the
    authoritative turn log. JSONL implementations can satisfy this by scanning
    records; SQL-like stores should expose the same query semantics using the
    advertised logical index specs. ClickHouse implementations can map the
    same fields to ORDER BY / primary-key and skip-index choices.

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

    def append_nodes(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        advance_head: TrajectoryHeadAdvance | None = None,
    ) -> None:
        """Atomically append nodes and optionally advance one append head.

        When ``advance_head`` is present the store must compare the current
        head with ``previous_node_id`` and advance to ``node_id`` in the same
        transaction as the node insert. This is the SDK-level concurrency
        contract that prevents branch/head races across JSONL, SQL, and OLAP
        stores.
        """
        ...

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
        """Return visible leaf nodes for diagnostics and projection repair.

        New append paths should use explicit heads, not infer current state
        from leaves.
        """
        ...

    def replace_session_projection(
        self,
        session_id: str,
        nodes: Sequence[TrajectoryNode],
        *,
        heads: Sequence[TrajectoryHead] = (),
        status: TrajectoryProjectionStatus | None = None,
    ) -> None:
        """Atomically replace all projected nodes/heads for a session.

        This is the recovery hook for projection stores after a crash,
        migration, or failed incremental projection append.
        """
        ...

    def projection_status(
        self,
        session_id: str,
    ) -> TrajectoryProjectionStatus | None:
        """Return projection high-water/health metadata, if persisted."""
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


__all__ = [
    "SessionMeta",
    "TRAJECTORY_HEAD_INDEXES",
    "TRAJECTORY_NODE_INDEXES",
    "TrajectoryStore",
    "TrajectoryNodeQuery",
    "TrajectoryNodeSort",
    "TrajectoryNodeStore",
]
