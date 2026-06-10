"""Pure :func:`fold_graph` — collapse an op log into a live :class:`Graph`.

The persistent audit graph is the **fold** of an ordered op list. Given
the initial empty graph plus a list of :mod:`graph_ops`-typed ops,
:func:`fold_graph` returns the current view. Replayable by construction:
the op list is the source of truth and the graph is a cached projection.

Semantics:

- ``NodeUpsert`` — ``nodes[op.id] = Event(...)``.
- ``NodeDelete`` — drop ``nodes[op.id]`` and every incident edge (cascade).
- ``EdgeUpsert`` — ``edges[(src, dst, kind)] = Edge(...)``.
- ``EdgeDelete`` — drop ``edges[(src, dst, kind)]``.

The fold uses insertion-ordered :class:`dict` so iteration order matches
op order. Downstream consumers (auditor compose, replay sidecar) expect
stable ordering; helpers :meth:`Graph.nodes_list` / :meth:`Graph.edges_list`
expose that as sequences.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from llmharness.schema import Edge, EdgeKind, Event, EventKind

from .ops import (
    EdgeDelete,
    EdgeUpsert,
    GraphOp,
    NodeDelete,
    NodeUpsert,
)


@dataclass(frozen=True)
class Graph:
    """Folded view of an op log.

    ``nodes`` is keyed by event id; ``edges`` by the ``(src, dst, kind)``
    triple (kind as the string value — the same string the op carries).
    Both are insertion-ordered.
    """

    nodes: dict[int, Event] = field(default_factory=dict)
    edges: dict[tuple[int, int, str], Edge] = field(default_factory=dict)

    def nodes_list(self) -> list[Event]:
        """Events in insertion order — what most downstream code wants."""
        return list(self.nodes.values())

    def edges_list(self) -> list[Edge]:
        """Edges in insertion order — what most downstream code wants."""
        return list(self.edges.values())


def fold_graph(ops: Iterable[GraphOp]) -> Graph:
    """Fold an op iterable into a :class:`Graph`. Pure, no I/O."""
    nodes: dict[int, Event] = {}
    edges: dict[tuple[int, int, str], Edge] = {}
    for op in ops:
        if isinstance(op, NodeUpsert):
            nodes[op.id] = Event(
                id=op.id,
                kind=EventKind(op.kind),
                summary=op.summary,
                source_turns=list(op.source_turns),
                external_refs=op.external_refs,
            )
        elif isinstance(op, NodeDelete):
            nodes.pop(op.id, None)
            # Cascade: drop every edge whose src or dst is op.id.
            for key in [k for k in edges if k[0] == op.id or k[1] == op.id]:
                del edges[key]
        elif isinstance(op, EdgeUpsert):
            key = (op.src, op.dst, op.kind)
            edges[key] = Edge(
                src=op.src,
                dst=op.dst,
                kind=EdgeKind(op.kind),
                reason=op.reason,
                src_turns=op.src_turns,
                dst_turns=op.dst_turns,
                cited_entities=op.cited_entities,
                cited_quote=op.cited_quote,
            )
        elif isinstance(op, EdgeDelete):
            edges.pop((op.src, op.dst, op.kind), None)
        else:  # pragma: no cover - exhaustiveness guard
            raise TypeError(f"unknown graph op type: {type(op).__name__}")
    return Graph(nodes=nodes, edges=edges)


__all__ = ["Graph", "fold_graph"]
