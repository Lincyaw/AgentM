"""Backend-neutral invariants shared by trajectory store implementations."""

from __future__ import annotations

from collections.abc import Sequence

from agentm.core.abi.trajectory import Turn, TurnCheckpoint, TurnRef


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
        turn_ids.add(turn.id)


def validate_turn_append(turns: Sequence[Turn], turn: Turn) -> None:
    """Require the next index and a session-unique durable turn id."""
    expected = len(turns)
    if turn.index != expected:
        raise ValueError(
            f"turn index {turn.index} does not follow {expected - 1}"
        )
    if any(existing.id == turn.id for existing in turns):
        raise ValueError(f"duplicate turn id in session: {turn.id}")


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
        raise ValueError(
            f"checkpoint id duplicates a committed turn: {checkpoint.id}"
        )
    if existing is not None and (
        existing.index != checkpoint.index or existing.id != checkpoint.id
    ):
        raise ValueError(
            "checkpoint replacement must preserve turn index and id"
        )


def validate_checkpoint_commit(
    checkpoint: TurnCheckpoint | None,
    turn: Turn,
) -> None:
    """Require a final turn to identify the checkpoint it supersedes."""
    if checkpoint is None:
        return
    if checkpoint.index != turn.index or checkpoint.id != turn.id:
        raise ValueError(
            "committed turn does not match the active checkpoint"
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
    "validate_turn_append",
    "validate_turn_checkpoint",
    "validate_turn_sequence",
]
