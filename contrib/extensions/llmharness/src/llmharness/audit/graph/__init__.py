"""Audit-graph model: ops, fold, phase merge."""

from __future__ import annotations

from .fold import Graph, fold_graph
from .ops import (
    EdgeDelete,
    EdgeUpsert,
    GraphOp,
    NodeDelete,
    NodeUpsert,
    op_from_edge,
    op_from_event,
    parse_op,
)
from .phase import merge_to_phases

__all__ = [
    "EdgeDelete",
    "EdgeUpsert",
    "Graph",
    "GraphOp",
    "NodeDelete",
    "NodeUpsert",
    "fold_graph",
    "merge_to_phases",
    "op_from_edge",
    "op_from_event",
    "parse_op",
]
