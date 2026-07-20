"""In-memory TrajectoryStore — non-persistent, for tests and ephemeral runs."""

from __future__ import annotations


from collections.abc import Sequence

from agentm.core.abi.store import (
    SessionMeta,
    TrajectoryCommit,
    TrajectoryCompactionCommit,
)
from agentm.core.abi.trajectory import (
    TrajectoryHead,
    TrajectoryNode,
    Turn,
    TurnCheckpoint,
    TurnRef,
)
from agentm.core.lib.trajectory_nodes import (
    TrajectoryIndexState,
    _synchronized_trajectory_state,
)
from agentm.core.lib.trajectory_store import (
    turn_prefix_cut,
    validate_checkpoint_commit,
    validate_checkpoint_discard,
    validate_compaction_commit,
    validate_turn_append,
    validate_turn_checkpoint,
    validate_turn_sequence,
)


class InMemoryTrajectoryStore(TrajectoryIndexState):
    """A ``TrajectoryStore`` backed by a plain dict.

    Non-persistent: all state is lost on process exit.  Returns shallow
    copies of the stored Turn lists so callers cannot add/remove turns
    from internal state.  Turn objects themselves are frozen dataclasses
    and shared between copies.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sessions: dict[str, tuple[SessionMeta, list[Turn]]] = {}
        self._checkpoints: dict[str, TurnCheckpoint] = {}

    @_synchronized_trajectory_state
    def create_session(
        self,
        meta: SessionMeta,
        *,
        turns: Sequence[Turn] = (),
        nodes: Sequence[TrajectoryNode] = (),
        head: TrajectoryHead,
    ) -> None:
        if meta.id in self._sessions:
            raise ValueError(f"session already exists: {meta.id}")
        copied = list(turns)
        validate_turn_sequence(copied)
        if head.session_id != meta.id:
            raise ValueError("initial trajectory head must belong to the session")
        turn_ids = {turn.id for turn in copied}
        if any(node.turn_id not in turn_ids for node in nodes):
            raise ValueError(
                "initial trajectory nodes must belong to an initial committed turn"
            )
        self._initialize_index(
            meta.id,
            nodes,
            head=head,
        )
        self._sessions[meta.id] = (meta, copied)

    @_synchronized_trajectory_state
    def save_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        existing = self._checkpoints.get(session_id)
        validate_turn_checkpoint(record[1], checkpoint, existing=existing)
        self._checkpoints[session_id] = checkpoint

    @_synchronized_trajectory_state
    def load_checkpoint(self, session_id: str) -> TurnCheckpoint | None:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._checkpoints.get(session_id)

    @_synchronized_trajectory_state
    def discard_checkpoint(
        self,
        session_id: str,
        checkpoint: TurnCheckpoint,
    ) -> None:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        current = self._checkpoints.get(session_id)
        validate_checkpoint_discard(current, checkpoint)
        if current is not None:
            self._checkpoints.pop(session_id)

    @_synchronized_trajectory_state
    def commit_turn(self, session_id: str, commit: TrajectoryCommit) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        turn = commit.turn
        validate_turn_append(record[1], turn)
        validate_checkpoint_commit(self._checkpoints.get(session_id), turn)
        if any(node.session_id != session_id for node in commit.nodes):
            raise ValueError("trajectory commit nodes must belong to the session")
        self._commit_nodes(
            session_id,
            commit.nodes,
            advance_head=commit.advance_head,
        )
        record[1].append(turn)
        self._checkpoints.pop(session_id, None)

    @_synchronized_trajectory_state
    def commit_compaction(
        self,
        session_id: str,
        commit: TrajectoryCompactionCommit,
    ) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        if commit.boundary.session_id != session_id:
            raise ValueError("trajectory compact boundary must belong to the session")
        validate_compaction_commit(record[1], commit)
        self._commit_nodes(
            session_id,
            (commit.boundary,),
            advance_head=commit.advance_head,
        )
        self.save_content_replacement_state(
            session_id,
            commit.content_replacement_state,
        )

    @_synchronized_trajectory_state
    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        meta, turns = record
        return (meta, list(turns))

    @_synchronized_trajectory_state
    def load_prefix(
        self,
        session_id: str,
        up_to: TurnRef,
    ) -> tuple[SessionMeta, list[Turn]]:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        meta, turns = record
        cut = turn_prefix_cut(turns, up_to)
        return (meta, list(turns[: cut + 1]))

    @_synchronized_trajectory_state
    def session_children(self, session_id: str) -> list[str]:
        return [
            sid
            for sid, (meta, _) in self._sessions.items()
            if meta.parent_id == session_id
        ]

    @_synchronized_trajectory_state
    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    @_synchronized_trajectory_state
    def list_sessions(self) -> list[SessionMeta]:
        return [meta for meta, _ in self._sessions.values()]


__all__ = ["InMemoryTrajectoryStore"]
