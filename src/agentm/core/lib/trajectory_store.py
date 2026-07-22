# code-health: ignore-file[AM025] -- core helpers normalize serialization, schema, and stream boundary data
"""Backend-neutral invariants shared by trajectory store implementations."""

from __future__ import annotations

from collections.abc import Sequence

from agentm.core.abi.store import TrajectoryCompactionCommit
from agentm.core.abi.termination import PromptRunContinued
from agentm.core.abi.trajectory import (
    TrajectoryHead,
    TrajectoryHeadAdvance,
    TrajectoryNode,
    Turn,
    TurnCheckpoint,
    TurnRef,
)


def validate_initial_node_state(
    session_id: str,
    nodes: Sequence[TrajectoryNode],
    head: TrajectoryHead,
) -> None:
    """Require a complete initial node index and matching explicit head."""

    if head.session_id != session_id:
        raise ValueError("initial trajectory head must belong to the session")
    copied = tuple(nodes)
    _validate_node_batch(session_id, copied, expected_seq=0)
    if not copied:
        if head.node_id is not None:
            raise ValueError(
                "initial trajectory head cannot target a node outside its index"
            )
        return
    if copied[0].parent_id is not None or copied[0].logical_parent_id is not None:
        raise ValueError(
            "initial trajectory node index must start without a predecessor"
        )
    _validate_head_node_identity(
        copied[-1],
        head,
        label="initial trajectory",
    )


def validate_node_append_state(
    session_id: str,
    nodes: Sequence[TrajectoryNode],
    advance: TrajectoryHeadAdvance,
    *,
    current_head: TrajectoryHead | None,
    expected_seq: int,
) -> None:
    """Require one append batch to preserve its selected chain identity."""

    copied = tuple(nodes)
    if not copied:
        raise ValueError("trajectory head advance requires a non-empty node batch")
    _validate_node_batch(session_id, copied, expected_seq=expected_seq)
    _validate_head_node_identity(
        copied[-1],
        advance,
        label="trajectory append",
    )
    if advance.logical_parent_id is not None:
        raise ValueError("an advanced trajectory head cannot retain a logical parent")

    if current_head is not None:
        if current_head.status != "active":
            raise ValueError("cannot append to an inactive trajectory head")
        if current_head.node_id != advance.previous_node_id:
            raise ValueError(
                "trajectory head changed before append: "
                f"{current_head.node_id!r} != {advance.previous_node_id!r}"
            )
        if (
            current_head.head_id != advance.head_id
            or current_head.branch_id != advance.branch_id
            or current_head.root_session_id != advance.root_session_id
            or current_head.parent_session_id != advance.parent_session_id
            or current_head.agent_id != advance.agent_id
            or current_head.is_sidechain != advance.is_sidechain
        ):
            raise ValueError("trajectory append cannot change head chain identity")

    first = copied[0]
    predecessor = (
        current_head.node_id or current_head.logical_parent_id
        if current_head is not None
        else advance.previous_node_id
    )
    if first.kind == "compact_boundary":
        if first.parent_id is not None or first.logical_parent_id != predecessor:
            raise ValueError(
                "compact boundary must logically reference the current head"
            )
    elif current_head is not None and current_head.node_id is not None:
        if first.parent_id != predecessor or first.logical_parent_id is not None:
            raise ValueError(
                "trajectory message append must physically extend the current head"
            )
    elif predecessor is None:
        if first.parent_id is not None or first.logical_parent_id is not None:
            raise ValueError(
                "first trajectory message cannot reference an unknown predecessor"
            )
    elif first.parent_id is not None or first.logical_parent_id != predecessor:
        raise ValueError(
            "new trajectory branch must logically reference its predecessor"
        )


def _validate_node_batch(
    session_id: str,
    nodes: Sequence[TrajectoryNode],
    *,
    expected_seq: int,
) -> None:
    batch_ids: set[str] = set()
    for node in nodes:
        if node.session_id != session_id:
            raise ValueError("trajectory node session_id does not match its session")
        if node.seq != expected_seq:
            raise ValueError(f"node seq {node.seq} does not follow {expected_seq - 1}")
        if node.id in batch_ids:
            raise ValueError(f"duplicate trajectory node id: {node.id}")
        batch_ids.add(node.id)
        expected_seq += 1
    if nodes and (
        nodes[0].parent_id in batch_ids or nodes[0].logical_parent_id in batch_ids
    ):
        raise ValueError("first trajectory node cannot reference its own append batch")
    for previous, node in zip(nodes, nodes[1:], strict=False):
        if node.parent_id != previous.id or node.logical_parent_id is not None:
            raise ValueError(
                "trajectory append nodes must form one physical parent chain"
            )
        if (
            node.head_id != previous.head_id
            or node.branch_id != previous.branch_id
            or node.root_session_id != previous.root_session_id
            or node.parent_session_id != previous.parent_session_id
            or node.agent_id != previous.agent_id
            or node.is_sidechain != previous.is_sidechain
        ):
            raise ValueError("trajectory append nodes must share one chain identity")


def _validate_head_node_identity(
    node: TrajectoryNode,
    head: TrajectoryHead | TrajectoryHeadAdvance,
    *,
    label: str,
) -> None:
    if head.node_id != node.id or head.seq != node.seq:
        raise ValueError(f"{label} head must target its final node")
    if (
        head.session_id != node.session_id
        or head.head_id != node.head_id
        or head.branch_id != node.branch_id
        or head.root_session_id != node.root_session_id
        or head.parent_session_id != node.parent_session_id
        or head.agent_id != node.agent_id
        or head.is_sidechain != node.is_sidechain
    ):
        raise ValueError(f"{label} head identity must match its final node")


def validate_turn_sequence(turns: Sequence[Turn]) -> None:
    """Require a complete zero-based sequence suitable for session creation."""
    turn_ids: set[str] = set()
    for expected, turn in enumerate(turns):
        if turn.index != expected:
            raise ValueError(
                f"turn index {turn.index} does not match expected {expected}"
            )
        if turn.id in turn_ids:
            raise ValueError(f"duplicate turn id in session: {turn.id}")
        _validate_prompt_run_position(turns[:expected], turn.run_id, turn.run_step)
        turn_ids.add(turn.id)


def validate_turn_append(turns: Sequence[Turn], turn: Turn) -> None:
    """Require the next index and a session-unique durable turn id."""
    expected = len(turns)
    if turn.index != expected:
        raise ValueError(f"turn index {turn.index} does not follow {expected - 1}")
    if any(existing.id == turn.id for existing in turns):
        raise ValueError(f"duplicate turn id in session: {turn.id}")
    _validate_prompt_run_position(turns, turn.run_id, turn.run_step)


def validate_turn_checkpoint(
    turns: Sequence[Turn],
    checkpoint: TurnCheckpoint,
    *,
    existing: TurnCheckpoint | None = None,
) -> None:
    """Require one checkpoint for the next uncommitted turn."""
    expected = len(turns)
    if checkpoint.index != expected:
        raise ValueError(
            f"checkpoint index {checkpoint.index} does not follow {expected - 1}"
        )
    if any(turn.id == checkpoint.id for turn in turns):
        raise ValueError(f"checkpoint id duplicates a committed turn: {checkpoint.id}")
    _validate_prompt_run_position(turns, checkpoint.run_id, checkpoint.run_step)
    if existing is not None and (
        existing.index != checkpoint.index
        or existing.id != checkpoint.id
        or existing.run_id != checkpoint.run_id
        or existing.run_step != checkpoint.run_step
    ):
        raise ValueError(
            "checkpoint replacement must preserve turn and prompt-run identity"
        )


def validate_checkpoint_commit(
    checkpoint: TurnCheckpoint | None,
    turn: Turn,
) -> None:
    """Require a final turn to identify the checkpoint it supersedes."""
    if checkpoint is None:
        return
    if (
        checkpoint.index != turn.index
        or checkpoint.id != turn.id
        or checkpoint.run_id != turn.run_id
        or checkpoint.run_step != turn.run_step
    ):
        raise ValueError("committed turn does not match the active checkpoint")


def _validate_prompt_run_position(
    turns: Sequence[Turn],
    run_id: str,
    run_step: int,
) -> None:
    if not turns:
        if run_step != 0:
            raise ValueError("first prompt-run Turn must have run_step 0")
        return

    previous = turns[-1]
    if previous.run_id == run_id:
        if not isinstance(previous.outcome.cause, PromptRunContinued):
            raise ValueError("a terminal Turn cannot continue the same prompt run")
        if run_step != previous.run_step + 1:
            raise ValueError("prompt-run steps must be contiguous")
        return

    if any(turn.run_id == run_id for turn in turns):
        raise ValueError("a prompt run cannot resume after another run started")
    if run_step != 0:
        raise ValueError("a new prompt run must start at run_step 0")


def validate_checkpoint_discard(
    current: TurnCheckpoint | None,
    expected: TurnCheckpoint,
) -> None:
    """Require compare-and-discard to target the current checkpoint."""

    if not isinstance(expected, TurnCheckpoint):
        raise TypeError("checkpoint discard target must be a TurnCheckpoint")
    if current is not None and current != expected:
        raise ValueError(
            "checkpoint discard target does not match the active checkpoint"
        )


def validate_compaction_commit(
    turns: Sequence[Turn],
    commit: TrajectoryCompactionCommit,
) -> None:
    """Require every control node to anchor the latest committed turn."""

    boundary = commit.boundary
    if boundary.turn_index is None or boundary.turn_id is None:
        raise ValueError("trajectory compact boundary requires a committed turn anchor")
    if not turns:
        raise ValueError("trajectory compact boundary requires committed history")
    latest = turns[-1]
    if latest.index != boundary.turn_index or latest.id != boundary.turn_id:
        raise ValueError(
            "trajectory compact boundary does not match the latest committed turn"
        )


def turn_prefix_cut(turns: Sequence[Turn], up_to: TurnRef) -> int:
    """Return the list index of a turn identified by index or durable id."""
    if isinstance(up_to, int):
        for position, turn in enumerate(turns):
            if turn.index == up_to:
                return position
        raise KeyError(up_to)
    for position, turn in enumerate(turns):
        if turn.id == up_to:
            return position
    raise KeyError(up_to)


__all__ = [
    "turn_prefix_cut",
    "validate_checkpoint_commit",
    "validate_checkpoint_discard",
    "validate_compaction_commit",
    "validate_initial_node_state",
    "validate_node_append_state",
    "validate_turn_append",
    "validate_turn_checkpoint",
    "validate_turn_sequence",
]
