"""SessionGraph — queryable session DAG with typed edges.

Sessions form a directed acyclic graph, not a tree.  A main agent can
spawn sub-agents whose results flow back; a workflow is a session that
orchestrates multiple child sessions.  Edges carry a ``kind`` that
distinguishes spawn/fork/workflow relationships.

Cross-trajectory data flow (which turn in session A consumes session B's
result) is captured in the Turn data itself — the ``SubagentResult``
trigger carries the child session id, and tool calls carry dispatch
metadata.  The graph here is about STRUCTURAL relationships; data-flow
queries are trajectory queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.trajectory import TurnRef

EdgeKind = Literal["spawned", "forked", "workflow_member"]


@dataclass(slots=True)
class SessionEdge:
    """A directed edge from parent to child in the session graph."""

    parent_id: str
    child_id: str
    kind: EdgeKind = "spawned"
    fork_point: TurnRef | None = None


@dataclass(slots=True)
class SessionNode:
    """One node in the session DAG."""

    session_id: str
    parent_id: str | None = None
    fork_point: TurnRef | None = None
    purpose: str = "root"
    edge_kind: EdgeKind | None = None
    children: list[str] = field(default_factory=list)


@runtime_checkable
class SessionGraphProtocol(Protocol):
    """Read/write interface over the session DAG.

    The runtime implementation holds this in memory; the store can
    reconstruct it from persisted SessionMeta records.
    """

    def register(
        self,
        session_id: str,
        *,
        parent_id: str | None = None,
        fork_point: TurnRef | None = None,
        purpose: str = "root",
        edge_kind: EdgeKind = "spawned",
    ) -> SessionNode:
        """Add a node.  Idempotent — re-registering the same id updates."""
        ...

    def get(self, session_id: str) -> SessionNode | None: ...

    def children(self, session_id: str, kind: EdgeKind | None = None) -> list[str]:
        """Direct children.  Filter by edge kind when given."""
        ...

    def ancestors(self, session_id: str) -> list[str]:
        """Walk parent chain to root: [parent, grandparent, ...]."""
        ...

    def root(self, session_id: str) -> str:
        """Root of this node's connected component."""
        ...

    def descendants(self, session_id: str) -> list[str]:
        """All transitive children (BFS order)."""
        ...

    def edges(self, session_id: str | None = None) -> list[SessionEdge]:
        """Edges from ``session_id``, or all edges when None."""
        ...

    def all_nodes(self) -> dict[str, SessionNode]: ...


__all__ = [
    "EdgeKind",
    "SessionEdge",
    "SessionGraphProtocol",
    "SessionNode",
]
