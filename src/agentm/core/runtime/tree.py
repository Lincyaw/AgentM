"""InMemorySessionGraph — queryable session DAG held in runtime memory."""

from __future__ import annotations

from collections import deque

from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.tree import (
    EdgeKind,
    SessionEdge,
    SessionNode,
)
from agentm.core.abi.trajectory import TurnRef


class InMemorySessionGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, SessionNode] = {}
        self._edges: list[SessionEdge] = []

    def register(
        self,
        session_id: str,
        *,
        parent_id: str | None = None,
        fork_point: TurnRef | None = None,
        purpose: str = "root",
        edge_kind: EdgeKind = "spawned",
    ) -> SessionNode:
        node = self._nodes.get(session_id)
        if node is None:
            node = SessionNode(session_id=session_id)
            self._nodes[session_id] = node
        old_parent_id = node.parent_id
        if old_parent_id is not None and old_parent_id != parent_id:
            old_parent = self._nodes.get(old_parent_id)
            if old_parent is not None and session_id in old_parent.children:
                old_parent.children.remove(session_id)
        self._edges = [edge for edge in self._edges if edge.child_id != session_id]
        node.parent_id = parent_id
        node.fork_point = fork_point
        node.purpose = purpose
        node.edge_kind = edge_kind
        if parent_id is not None:
            parent = self._nodes.get(parent_id)
            if parent is None:
                parent = SessionNode(session_id=parent_id)
                self._nodes[parent_id] = parent
            if session_id not in parent.children:
                parent.children.append(session_id)
            self._edges.append(
                SessionEdge(
                    parent_id=parent_id,
                    child_id=session_id,
                    kind=edge_kind,
                    fork_point=fork_point,
                )
            )
        return node

    def get(self, session_id: str) -> SessionNode | None:
        return self._nodes.get(session_id)

    def children(self, session_id: str, kind: EdgeKind | None = None) -> list[str]:
        node = self._nodes.get(session_id)
        if node is None:
            return []
        if kind is None:
            return list(node.children)
        return [
            e.child_id
            for e in self._edges
            if e.parent_id == session_id and e.kind == kind
        ]

    def ancestors(self, session_id: str) -> list[str]:
        result: list[str] = []
        node = self._nodes.get(session_id)
        if node is None:
            return result
        current = node.parent_id
        seen: set[str] = {session_id}
        while current is not None and current not in seen:
            result.append(current)
            seen.add(current)
            parent = self._nodes.get(current)
            if parent is None:
                break
            current = parent.parent_id
        return result

    def root(self, session_id: str) -> str:
        node = self._nodes.get(session_id)
        if node is None or node.parent_id is None:
            return session_id
        chain = self.ancestors(session_id)
        return chain[-1] if chain else session_id

    def descendants(self, session_id: str) -> list[str]:
        result: list[str] = []
        queue: deque[str] = deque()
        node = self._nodes.get(session_id)
        if node is None:
            return result
        queue.extend(node.children)
        seen: set[str] = {session_id}
        while queue:
            sid = queue.popleft()
            if sid in seen:
                continue
            seen.add(sid)
            result.append(sid)
            child_node = self._nodes.get(sid)
            if child_node is not None:
                queue.extend(child_node.children)
        return result

    def edges(self, session_id: str | None = None) -> list[SessionEdge]:
        if session_id is None:
            return list(self._edges)
        return [e for e in self._edges if e.parent_id == session_id]

    def all_nodes(self) -> dict[str, SessionNode]:
        return dict(self._nodes)

    def rebuild_from_store(self, store: TrajectoryStore) -> None:
        """Reconstruct the graph from a TrajectoryStore's session metadata.

        Called after a process restart to populate the in-memory graph
        from persisted session records.
        """
        for meta in store.list_sessions():
            self.register(
                meta.id,
                parent_id=meta.parent_id,
                fork_point=meta.fork_point,
                purpose=meta.purpose,
                edge_kind="forked" if meta.fork_point is not None else "spawned",
            )


__all__ = [
    "InMemorySessionGraph",
]
