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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    MessageVisibility,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.resource import ResourceMutation
from agentm.core.abi.termination import TerminationCause

if TYPE_CHECKING:
    from agentm.core.abi.trigger import Trigger

TurnRef = Union[int, str]
TrajectoryNodeRef = str
TrajectoryBranchId = str
TrajectoryHeadId = str
DEFAULT_TRAJECTORY_BRANCH_ID = "main"
DEFAULT_TRAJECTORY_HEAD_ID = "main"
TrajectoryNodeKind = Literal[
    "message",
    "compact_boundary",
    "content_replacement",
    "config_change",
    "snip",
    "checkpoint",
]
TrajectoryNodeRole = Literal["user", "assistant", "tool_result", "control"]
TrajectoryHeadStatus = Literal["active", "dead", "archived"]
TrajectoryProjectionState = Literal["current", "stale", "failed"]
TrajectoryIndexField = Literal[
    "root_session_id",
    "session_id",
    "parent_session_id",
    "branch_id",
    "head_id",
    "seq",
    "id",
    "parent_id",
    "logical_parent_id",
    "turn_id",
    "turn_index",
    "round_index",
    "message_index",
    "agent_id",
    "is_sidechain",
    "kind",
    "role",
    "tool_call_id",
    "tool_name",
    "cache_key",
    "content_ref",
    "visibility",
    "timestamp",
]


@dataclass(frozen=True, slots=True)
class TrajectoryIndexSpec:
    """Backend-neutral index declaration for trajectory node stores."""

    name: str
    fields: tuple[TrajectoryIndexField, ...]
    unique: bool = False
    purpose: str = ""


TRAJECTORY_NODE_INDEXES: tuple[TrajectoryIndexSpec, ...] = (
    TrajectoryIndexSpec(
        name="trajectory_nodes_id",
        fields=("id",),
        unique=True,
        purpose="global stable node lookup",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_session_seq",
        fields=("session_id", "seq"),
        unique=True,
        purpose="append-order scans and resume replay",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_session_id",
        fields=("session_id", "id"),
        unique=True,
        purpose="stable node lookup",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_parent",
        fields=("session_id", "parent_id"),
        purpose="leaf detection and child scans",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_branch_seq",
        fields=("session_id", "branch_id", "head_id", "seq"),
        purpose="branch/head ordered replay and active-head repair",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_logical_parent",
        fields=("logical_parent_id", "session_id", "seq"),
        purpose="fork and compact-boundary logical lineage scans",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_agent_leaf",
        fields=("session_id", "agent_id", "is_sidechain", "seq"),
        purpose="main/sidechain filtering and agent resume",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_root_session_seq",
        fields=("root_session_id", "session_id", "seq"),
        purpose="trace-scope scans in SQL/ClickHouse backends",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_turn",
        fields=("session_id", "turn_index", "round_index", "message_index", "turn_id"),
        purpose="turn-to-node projection and trace joins",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_tool_call",
        fields=("root_session_id", "tool_call_id", "session_id", "seq"),
        purpose="tool_use/tool_result completion and replay diagnostics",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_cache",
        fields=("root_session_id", "cache_key", "session_id", "seq"),
        purpose="prompt-cache/content-replacement prefix lookup",
    ),
)


TRAJECTORY_HEAD_INDEXES: tuple[TrajectoryIndexSpec, ...] = (
    TrajectoryIndexSpec(
        name="trajectory_heads_id",
        fields=("session_id", "head_id"),
        unique=True,
        purpose="current append point lookup",
    ),
    TrajectoryIndexSpec(
        name="trajectory_heads_branch",
        fields=("root_session_id", "session_id", "branch_id", "agent_id", "is_sidechain"),
        purpose="branch, agent, and sidechain head selection",
    ),
)


@dataclass(frozen=True, slots=True)
class ContentReplacementState:
    """Persistent prompt-cache/content-replacement state.

    The state is keyed and cloneable so forked/resumed sessions make byte-stable
    replacement decisions for messages they share with their parent.
    """

    state_key: str
    seen_tool_call_ids: tuple[str, ...] = ()
    replacements: Mapping[str, str] = field(default_factory=dict)
    source_session_id: str | None = None
    source_leaf_id: str | None = None
    leaf_node_id: str | None = None
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromptCacheState:
    """Provider-facing prompt-cache identity attached to a chain prefix."""

    cache_key: str
    leaf_node_id: str | None = None
    content_replacement_state_key: str | None = None
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    provider: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryHead:
    """Named append head for one branch/agent/sidechain chain.

    A session may have many visible leaves, but appending must target one
    explicit head. Fork/resume can create an empty head with
    ``logical_parent_id`` pointing at an inherited prefix; the first append in
    the child then materializes a new physical branch without copying storage
    backend internals.
    """

    session_id: str
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    node_id: str | None = None
    seq: int | None = None
    root_session_id: str | None = None
    parent_session_id: str | None = None
    logical_parent_id: str | None = None
    agent_id: str | None = None
    is_sidechain: bool = False
    status: TrajectoryHeadStatus = "active"
    updated_at: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryHeadAdvance:
    """Atomic compare-and-advance request for a trajectory head."""

    session_id: str
    node_id: str
    seq: int
    previous_node_id: str | None = None
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    root_session_id: str | None = None
    parent_session_id: str | None = None
    logical_parent_id: str | None = None
    agent_id: str | None = None
    is_sidechain: bool = False
    status: TrajectoryHeadStatus = "active"
    updated_at: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_head(self) -> TrajectoryHead:
        """Materialize the head record after the append succeeds."""
        return TrajectoryHead(
            session_id=self.session_id,
            head_id=self.head_id,
            branch_id=self.branch_id,
            node_id=self.node_id,
            seq=self.seq,
            root_session_id=self.root_session_id,
            parent_session_id=self.parent_session_id,
            logical_parent_id=self.logical_parent_id,
            agent_id=self.agent_id,
            is_sidechain=self.is_sidechain,
            status=self.status,
            updated_at=self.updated_at,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryForkPoint:
    """SDK-level fork/resume anchor.

    ``turn_ref`` preserves the existing turn-prefix fork API. ``node_id`` and
    ``head_id`` let hosts fork from message-level nodes or active heads once a
    ``TrajectoryNodeStore`` is present.
    """

    session_id: str
    turn_ref: TurnRef | None = None
    node_id: str | None = None
    head_id: TrajectoryHeadId | None = None
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    include_logical_parent: bool = True


@dataclass(frozen=True, slots=True)
class SessionConfigChange:
    """Explicit session config/provider change represented as a control node."""

    change_id: str
    session_id: str
    key: str
    before: str | None = None
    after: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    node_id: str | None = None
    reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryProjectionStatus:
    """Projection health for rebuildable message-node stores."""

    session_id: str
    state: TrajectoryProjectionState = "current"
    high_water_turn_id: str | None = None
    high_water_turn_index: int | None = None
    node_count: int = 0
    updated_at: float = 0.0
    error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryNode:
    """Message-level append-only trajectory node.

    Stores may persist this directly (SQL/ClickHouse) or derive it from turn
    records (JSONL). The fields are deliberately backend-neutral: identity,
    parent linkage, agent/sidechain ownership, turn joins, and opaque payload.
    """

    id: str
    session_id: str
    seq: int
    kind: TrajectoryNodeKind
    root_session_id: str | None = None
    parent_session_id: str | None = None
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    role: TrajectoryNodeRole = "control"
    parent_id: str | None = None
    logical_parent_id: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    round_index: int | None = None
    message_index: int | None = None
    agent_id: str | None = None
    is_sidechain: bool = False
    tool_call_ids: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    cache_key: str | None = None
    content_ref: str | None = None
    visibility: MessageVisibility = "visible"
    message: AgentMessage | None = None
    payload: Mapping[str, object] = field(default_factory=dict)
    removed_node_ids: tuple[str, ...] = ()
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class TrajectoryLeaf:
    """Current visible leaf for one session/agent chain."""

    session_id: str
    node_id: str
    seq: int
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID
    head_id: TrajectoryHeadId = DEFAULT_TRAJECTORY_HEAD_ID
    agent_id: str | None = None
    is_sidechain: bool = False
    status: TrajectoryHeadStatus = "active"


@dataclass(frozen=True, slots=True)
class TurnMeta:
    """Observability data attached to a committed Turn."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ns: int = 0
    model_id: str | None = None
    resource_mutations: tuple[ResourceMutation, ...] = ()


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
class InjectedMessages:
    """Messages causally inserted after one completed ReAct round."""

    after_round: int
    messages: tuple[AgentMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class Outcome:
    """Why a turn ended and causally anchored inline injections."""

    cause: TerminationCause
    injected: tuple[InjectedMessages, ...] = ()


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
    "ContentReplacementState",
    "DEFAULT_TRAJECTORY_BRANCH_ID",
    "DEFAULT_TRAJECTORY_HEAD_ID",
    "InjectedMessages",
    "Outcome",
    "PromptCacheState",
    "Round",
    "SessionConfigChange",
    "TRAJECTORY_HEAD_INDEXES",
    "TRAJECTORY_NODE_INDEXES",
    "ToolRecord",
    "TrajectoryBranchId",
    "TrajectoryForkPoint",
    "TrajectoryHead",
    "TrajectoryHeadAdvance",
    "TrajectoryHeadId",
    "TrajectoryHeadStatus",
    "TrajectoryIndexField",
    "TrajectoryIndexSpec",
    "TrajectoryLeaf",
    "TrajectoryNode",
    "TrajectoryNodeKind",
    "TrajectoryNodeRef",
    "TrajectoryNodeRole",
    "TrajectoryProjectionState",
    "TrajectoryProjectionStatus",
    "Turn",
    "TurnMeta",
    "TurnRef",
]
