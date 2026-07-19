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
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.resource import ResourceMutation
from agentm.core.abi.termination import TerminationCause

if TYPE_CHECKING:
    from agentm.core.abi.trigger import Trigger

TurnRef = Union[int, str]
TrajectoryNodeRef = str
TrajectoryNodeKind = Literal[
    "message",
    "compact_boundary",
    "content_replacement",
    "snip",
    "checkpoint",
]
TrajectoryNodeRole = Literal["user", "assistant", "tool_result", "control"]
TrajectoryIndexField = Literal[
    "root_session_id",
    "session_id",
    "parent_session_id",
    "seq",
    "id",
    "parent_id",
    "logical_parent_id",
    "turn_id",
    "turn_index",
    "agent_id",
    "is_sidechain",
    "kind",
    "role",
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
        fields=("session_id", "turn_index", "turn_id"),
        purpose="turn-to-node projection and trace joins",
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
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromptCacheState:
    """Provider-facing prompt-cache identity attached to a chain prefix."""

    cache_key: str
    leaf_node_id: str | None = None
    content_replacement_state_key: str | None = None
    provider: str | None = None
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
    role: TrajectoryNodeRole = "control"
    parent_id: str | None = None
    logical_parent_id: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    round_index: int | None = None
    message_index: int | None = None
    agent_id: str | None = None
    is_sidechain: bool = False
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
    agent_id: str | None = None
    is_sidechain: bool = False


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
    "InjectedMessages",
    "Outcome",
    "PromptCacheState",
    "Round",
    "TRAJECTORY_NODE_INDEXES",
    "ToolRecord",
    "TrajectoryIndexField",
    "TrajectoryIndexSpec",
    "TrajectoryLeaf",
    "TrajectoryNode",
    "TrajectoryNodeKind",
    "TrajectoryNodeRef",
    "TrajectoryNodeRole",
    "Turn",
    "TurnMeta",
    "TurnRef",
]
