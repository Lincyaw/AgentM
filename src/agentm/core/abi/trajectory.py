# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Core trajectory data types.

All types here are frozen dataclasses — once a Turn is committed to a
Trajectory it is immutable.

Type hierarchy::

    Turn
      ├── trigger: Trigger    (what caused this turn)
      ├── response: AssistantMessage | None
      ├── tool_results: tuple[ToolRecord, ...]
      ├── outcome: Outcome    (step / stop / inject)
      └── meta: TurnMeta      (observability data)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Union

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    JsonValue,
    MessageVisibility,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
    freeze_json,
)
from agentm.core.abi.resource import ResourceMutation
from agentm.core.abi.termination import TerminationCause

if TYPE_CHECKING:
    from agentm.core.abi.trigger import Trigger, TriggerMetadata

TurnRef = Union[int, str]
TrajectoryNodeRef = str
TrajectoryBranchId = str
TrajectoryHeadId = str
DEFAULT_TRAJECTORY_BRANCH_ID = "main"
DEFAULT_TRAJECTORY_HEAD_ID = "main"
TrajectoryNodeKind = Literal[
    "message",
    "compact_boundary",
    "system_prompt",
]
TrajectoryNodeRole = Literal["user", "assistant", "tool_result", "control"]
TrajectoryHeadStatus = Literal["active", "dead", "archived"]
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
    "run_id",
    "run_step",
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


def _require_string(value: object, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _require_index(value: object, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_finite(value: object, label: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")


def _freeze_metadata(
    value: Mapping[str, object],
    label: str,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    frozen = freeze_json(value)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{label} must be an object")
    return frozen


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a tuple of non-empty strings")
    return value


@dataclass(frozen=True, slots=True)
class TrajectoryIndexSpec:
    """Backend-neutral declaration for a trajectory message index."""

    name: str
    fields: tuple[TrajectoryIndexField, ...]
    unique: bool = False
    purpose: str = ""

    def __post_init__(self) -> None:
        _require_string(self.name, "trajectory index name")
        if not self.fields:
            raise ValueError("trajectory index fields cannot be empty")
        if not isinstance(self.unique, bool):
            raise TypeError("trajectory index unique must be a bool")
        if not isinstance(self.purpose, str):
            raise TypeError("trajectory index purpose must be a string")


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
        purpose="branch/head ordered replay and consistency checks",
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
        fields=("session_id", "turn_index", "message_index", "turn_id"),
        purpose="committed turn/message ordering and trace joins",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_prompt_run",
        fields=("session_id", "run_id", "run_step", "message_index"),
        purpose="prompt-run step ordering and continuation diagnostics",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_tool_call",
        fields=("root_session_id", "tool_call_id", "session_id", "seq"),
        purpose="tool_use/tool_result completion and replay diagnostics",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_tool_name",
        fields=("root_session_id", "tool_name", "session_id", "seq"),
        purpose="tool-name trajectory filtering and usage diagnostics",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_cache",
        fields=("root_session_id", "cache_key", "session_id", "seq"),
        purpose="prompt-cache/content-replacement prefix lookup",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_content_ref",
        fields=("content_ref", "session_id", "seq"),
        purpose="referenced summary and artifact lineage lookup",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_visibility",
        fields=("session_id", "kind", "role", "visibility", "seq"),
        purpose="provider-visible and control-node replay filtering",
    ),
    TrajectoryIndexSpec(
        name="trajectory_nodes_session_timestamp",
        fields=("session_id", "timestamp", "seq"),
        purpose="time-range trajectory scans",
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
        fields=(
            "root_session_id",
            "session_id",
            "branch_id",
            "agent_id",
            "is_sidechain",
        ),
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

    def __post_init__(self) -> None:
        _require_string(self.state_key, "content replacement state key")
        object.__setattr__(
            self,
            "seen_tool_call_ids",
            _string_tuple(
                self.seen_tool_call_ids,
                "content replacement seen tool call ids",
            ),
        )
        if not isinstance(self.replacements, Mapping) or not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in self.replacements.items()
        ):
            raise ValueError(
                "content replacements must map non-empty strings to non-empty strings"
            )
        object.__setattr__(
            self,
            "replacements",
            MappingProxyType(dict(self.replacements)),
        )
        _require_string(
            self.source_session_id,
            "content replacement source_session_id",
            optional=True,
        )
        _require_string(
            self.source_leaf_id,
            "content replacement source_leaf_id",
            optional=True,
        )
        _require_string(
            self.leaf_node_id,
            "content replacement leaf_node_id",
            optional=True,
        )
        _require_string(self.branch_id, "content replacement branch_id")
        _require_string(self.head_id, "content replacement head_id")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "content replacement metadata"),
        )


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

    def __post_init__(self) -> None:
        _require_string(self.cache_key, "prompt cache key")
        _require_string(
            self.leaf_node_id,
            "prompt cache leaf_node_id",
            optional=True,
        )
        _require_string(
            self.content_replacement_state_key,
            "prompt cache content_replacement_state_key",
            optional=True,
        )
        _require_string(self.branch_id, "prompt cache branch_id")
        _require_string(self.head_id, "prompt cache head_id")
        _require_string(self.provider, "prompt cache provider", optional=True)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "prompt cache metadata"),
        )


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

    def __post_init__(self) -> None:
        _validate_head_fields(
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
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "trajectory head metadata"),
        )


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

    def __post_init__(self) -> None:
        _validate_head_fields(
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
        )
        _require_string(
            self.previous_node_id,
            "trajectory head previous_node_id",
            optional=True,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "trajectory head advance metadata"),
        )

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
    """SDK-level selector for one executable fork boundary.

    ``turn_ref`` selects a committed turn directly. ``node_id`` and ``head_id``
    provide stable graph selectors, but the selected node must still represent
    a committed turn boundary. A compact boundary is valid because it is a
    control-only mutation anchored to an already committed turn. A message from
    the middle of a turn is not executable fork state because external effects
    are snapshotted only at turn commit.
    """

    session_id: str
    turn_ref: TurnRef | None = None
    node_id: str | None = None
    head_id: TrajectoryHeadId | None = None
    branch_id: TrajectoryBranchId = DEFAULT_TRAJECTORY_BRANCH_ID

    def __post_init__(self) -> None:
        _require_string(self.session_id, "trajectory fork session_id")
        anchors = sum(
            anchor is not None for anchor in (self.turn_ref, self.node_id, self.head_id)
        )
        if anchors != 1:
            raise ValueError(
                "trajectory fork point must set exactly one of turn_ref, node_id, "
                "or head_id"
            )
        if self.turn_ref is not None:
            if isinstance(self.turn_ref, bool) or not isinstance(
                self.turn_ref,
                (str, int),
            ):
                raise TypeError("trajectory fork turn_ref must be a string or integer")
            if isinstance(self.turn_ref, str):
                _require_string(self.turn_ref, "trajectory fork turn_ref")
            else:
                _require_index(self.turn_ref, "trajectory fork turn_ref")
        _require_string(self.node_id, "trajectory fork node_id", optional=True)
        _require_string(self.head_id, "trajectory fork head_id", optional=True)
        _require_string(self.branch_id, "trajectory fork branch_id")


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
    run_id: str | None = None
    run_step: int | None = None
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
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        _require_string(self.id, "trajectory node id")
        _require_string(self.session_id, "trajectory node session_id")
        _require_index(self.seq, "trajectory node seq")
        if self.kind not in {"message", "compact_boundary", "system_prompt"}:
            raise ValueError(f"invalid trajectory node kind: {self.kind!r}")
        for label, string_value in (
            ("root_session_id", self.root_session_id),
            ("parent_session_id", self.parent_session_id),
            ("parent_id", self.parent_id),
            ("logical_parent_id", self.logical_parent_id),
            ("turn_id", self.turn_id),
            ("run_id", self.run_id),
            ("agent_id", self.agent_id),
            ("cache_key", self.cache_key),
            ("content_ref", self.content_ref),
        ):
            _require_string(
                string_value,
                f"trajectory node {label}",
                optional=True,
            )
        _require_string(self.branch_id, "trajectory node branch_id")
        _require_string(self.head_id, "trajectory node head_id")
        if self.role not in {"user", "assistant", "tool_result", "control"}:
            raise ValueError(f"invalid trajectory node role: {self.role!r}")
        for label, index_value in (
            ("turn_index", self.turn_index),
            ("run_step", self.run_step),
            ("message_index", self.message_index),
        ):
            _require_index(
                index_value,
                f"trajectory node {label}",
                optional=True,
            )
        if not isinstance(self.is_sidechain, bool):
            raise TypeError("trajectory node is_sidechain must be a bool")
        object.__setattr__(
            self,
            "tool_call_ids",
            _string_tuple(self.tool_call_ids, "trajectory node tool_call_ids"),
        )
        object.__setattr__(
            self,
            "tool_names",
            _string_tuple(self.tool_names, "trajectory node tool_names"),
        )
        if self.visibility not in {"visible", "hidden", "replay_only"}:
            raise ValueError(f"invalid trajectory node visibility: {self.visibility!r}")
        if self.kind == "message":
            if self.message is None:
                raise ValueError("message trajectory nodes require a message")
            if self.message.role != self.role:
                raise ValueError("trajectory node role must match its message role")
        elif self.kind == "system_prompt":
            if self.message is not None:
                raise ValueError("system_prompt nodes cannot carry a message")
            if self.role != "control":
                raise ValueError("system_prompt nodes require the control role")
            if self.content_ref is None:
                raise ValueError("system_prompt nodes require a content_ref")
        else:
            if self.message is not None:
                raise ValueError("control trajectory nodes cannot carry a message")
            if self.role != "control":
                raise ValueError("control trajectory nodes require the control role")
            if self.content_ref is None:
                raise ValueError("compact boundary nodes require a content_ref")
            if self.turn_id is None or self.turn_index is None:
                raise ValueError(
                    "compact boundary nodes require a committed turn anchor"
                )
        object.__setattr__(
            self,
            "payload",
            _freeze_metadata(self.payload, "trajectory node payload"),
        )
        _require_finite(self.timestamp, "trajectory node timestamp")


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

    def __post_init__(self) -> None:
        _require_string(self.session_id, "trajectory leaf session_id")
        _require_string(self.node_id, "trajectory leaf node_id")
        _require_index(self.seq, "trajectory leaf seq")
        _require_string(self.branch_id, "trajectory leaf branch_id")
        _require_string(self.head_id, "trajectory leaf head_id")
        _require_string(self.agent_id, "trajectory leaf agent_id", optional=True)
        if not isinstance(self.is_sidechain, bool):
            raise TypeError("trajectory leaf is_sidechain must be a bool")
        if self.status not in {"active", "dead", "archived"}:
            raise ValueError(f"invalid trajectory leaf status: {self.status!r}")


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
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("total_input_tokens", self.total_input_tokens),
            ("total_output_tokens", self.total_output_tokens),
            ("cache_read_tokens", self.cache_read_tokens),
            ("cache_write_tokens", self.cache_write_tokens),
            ("duration_ns", self.duration_ns),
        ):
            _require_index(value, f"turn meta {label}")
        _require_string(self.model_id, "turn meta model_id", optional=True)
        if not isinstance(self.resource_mutations, tuple) or not all(
            isinstance(item, ResourceMutation) for item in self.resource_mutations
        ):
            raise TypeError(
                "turn meta resource_mutations must be a tuple of ResourceMutation"
            )
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise TypeError("turn meta system_prompt must be a string or None")


def _validate_resource_transaction_anchors(
    meta: TurnMeta,
    *,
    turn_id: str,
    turn_index: int,
    label: str,
) -> None:
    for mutation in meta.resource_mutations:
        transaction = mutation.transaction
        if transaction is None:
            continue
        if transaction.turn_id != turn_id or transaction.turn_index != turn_index:
            raise ValueError(
                f"{label} resource transaction {transaction.id!r} does not "
                "match its owning turn"
            )


@dataclass(frozen=True, slots=True)
class ToolRecord:
    """One tool call and its final result (post all bus hooks)."""

    call: ToolCallBlock
    result: ToolResultBlock
    backgrounded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.call, ToolCallBlock):
            raise TypeError("tool record call must be ToolCallBlock")
        if not isinstance(self.result, ToolResultBlock):
            raise TypeError("tool record result must be ToolResultBlock")
        if self.result.tool_call_id != self.call.id:
            raise ValueError("tool record result must reference its call id")
        if not isinstance(self.backgrounded, bool):
            raise TypeError("tool record backgrounded must be a bool")


def _validate_turn_payload(
    response: AssistantMessage | None,
    tool_results: tuple[ToolRecord, ...],
    *,
    label: str,
) -> None:
    if response is not None and not isinstance(response, AssistantMessage):
        raise TypeError(f"{label} response must be AssistantMessage or None")
    if not isinstance(tool_results, tuple) or not all(
        isinstance(item, ToolRecord) for item in tool_results
    ):
        raise TypeError(f"{label} tool_results must be a tuple of ToolRecord")
    if response is None:
        if tool_results:
            raise ValueError(f"{label} without a response cannot have tool results")
        return
    call_ids = {
        block.id for block in response.content if isinstance(block, ToolCallBlock)
    }
    result_ids = [record.call.id for record in tool_results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError(f"{label} tool results contain duplicate call ids")
    if not set(result_ids).issubset(call_ids):
        raise ValueError(f"{label} tool results must reference calls from the response")


def _validate_injected(
    injected: tuple[AgentMessage, ...],
    *,
    label: str,
) -> None:
    if not isinstance(injected, tuple) or not all(
        isinstance(item, (UserMessage, AssistantMessage, ToolResultMessage))
        for item in injected
    ):
        raise TypeError(f"{label} must be a tuple of AgentMessage")


@dataclass(frozen=True, slots=True)
class Outcome:
    """Why one durable turn ended and messages injected after it."""

    cause: TerminationCause
    injected: tuple[AgentMessage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cause, TerminationCause):
            raise TypeError("outcome cause must be a TerminationCause")
        _validate_injected(self.injected, label="outcome injected")


@dataclass(frozen=True, slots=True)
class TurnCheckpoint:
    """Latest durable state for one incomplete model/tool transaction."""

    index: int
    id: str
    run_id: str
    run_step: int
    trigger: Trigger
    response: AssistantMessage | None
    tool_results: tuple[ToolRecord, ...]
    updated_at: float
    meta: TurnMeta = field(default_factory=TurnMeta)
    injected: tuple[AgentMessage, ...] = ()
    trigger_metadata: TriggerMetadata | None = None

    def __post_init__(self) -> None:
        from agentm.core.abi.trigger import Trigger, TriggerMetadata

        _require_index(self.index, "turn checkpoint index")
        _require_string(self.id, "turn checkpoint id")
        _require_string(self.run_id, "turn checkpoint run_id")
        _require_index(self.run_step, "turn checkpoint run_step")
        if not isinstance(self.trigger, Trigger):
            raise TypeError("turn checkpoint trigger must implement Trigger")
        _require_string(self.trigger.source, "turn checkpoint trigger source")
        _validate_turn_payload(
            self.response,
            self.tool_results,
            label="turn checkpoint",
        )
        _validate_injected(self.injected, label="turn checkpoint injected")
        _require_finite(self.updated_at, "turn checkpoint updated_at")
        if not isinstance(self.meta, TurnMeta):
            raise TypeError("turn checkpoint meta must be TurnMeta")
        _validate_resource_transaction_anchors(
            self.meta,
            turn_id=self.id,
            turn_index=self.index,
            label="turn checkpoint",
        )
        if self.trigger_metadata is not None and not isinstance(
            self.trigger_metadata,
            TriggerMetadata,
        ):
            raise TypeError(
                "turn checkpoint trigger_metadata must be TriggerMetadata or None"
            )


@dataclass(frozen=True, slots=True)
class Turn:
    """One committed assistant response and its complete tool-result set."""

    index: int
    id: str
    run_id: str
    run_step: int
    trigger: Trigger
    response: AssistantMessage | None
    tool_results: tuple[ToolRecord, ...]
    outcome: Outcome
    timestamp: float
    meta: TurnMeta = field(default_factory=TurnMeta)
    trigger_metadata: TriggerMetadata | None = None

    def __post_init__(self) -> None:
        from agentm.core.abi.trigger import Trigger, TriggerMetadata

        _require_index(self.index, "turn index")
        _require_string(self.id, "turn id")
        _require_string(self.run_id, "turn run_id")
        _require_index(self.run_step, "turn run_step")
        if not isinstance(self.trigger, Trigger):
            raise TypeError("turn trigger must implement Trigger")
        _require_string(self.trigger.source, "turn trigger source")
        _validate_turn_payload(self.response, self.tool_results, label="turn")
        if not isinstance(self.outcome, Outcome):
            raise TypeError("turn outcome must be Outcome")
        _require_finite(self.timestamp, "turn timestamp")
        if not isinstance(self.meta, TurnMeta):
            raise TypeError("turn meta must be TurnMeta")
        _validate_resource_transaction_anchors(
            self.meta,
            turn_id=self.id,
            turn_index=self.index,
            label="turn",
        )
        if self.trigger_metadata is not None and not isinstance(
            self.trigger_metadata,
            TriggerMetadata,
        ):
            raise TypeError("turn trigger_metadata must be TriggerMetadata or None")


def _validate_head_fields(
    *,
    session_id: str,
    head_id: str,
    branch_id: str,
    node_id: str | None,
    seq: int | None,
    root_session_id: str | None,
    parent_session_id: str | None,
    logical_parent_id: str | None,
    agent_id: str | None,
    is_sidechain: bool,
    status: TrajectoryHeadStatus,
    updated_at: float,
) -> None:
    _require_string(session_id, "trajectory head session_id")
    _require_string(head_id, "trajectory head head_id")
    _require_string(branch_id, "trajectory head branch_id")
    _require_string(node_id, "trajectory head node_id", optional=True)
    _require_index(seq, "trajectory head seq", optional=True)
    if (node_id is None) != (seq is None):
        raise ValueError("trajectory head node_id and seq must be set together")
    if node_id is not None and logical_parent_id is not None:
        raise ValueError(
            "trajectory head logical_parent_id is valid only for an empty head"
        )
    for label, value in (
        ("root_session_id", root_session_id),
        ("parent_session_id", parent_session_id),
        ("logical_parent_id", logical_parent_id),
        ("agent_id", agent_id),
    ):
        _require_string(value, f"trajectory head {label}", optional=True)
    if not isinstance(is_sidechain, bool):
        raise TypeError("trajectory head is_sidechain must be a bool")
    if status not in {"active", "dead", "archived"}:
        raise ValueError(f"invalid trajectory head status: {status!r}")
    _require_finite(updated_at, "trajectory head updated_at")


__all__ = [
    "ContentReplacementState",
    "DEFAULT_TRAJECTORY_BRANCH_ID",
    "DEFAULT_TRAJECTORY_HEAD_ID",
    "Outcome",
    "PromptCacheState",
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
    "Turn",
    "TurnCheckpoint",
    "TurnMeta",
    "TurnRef",
]
