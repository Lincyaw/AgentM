"""Compatibility exports for the pre-index extractor store names.

New code should import from :mod:`llmharness.agents.extractor.index_store`.
This module remains only so old sidecars and external imports fail softly.
"""

from __future__ import annotations

from .index_store import (
    Index,
    IndexOp,
    LinkDelete,
    LinkUpsert,
    RecordDelete,
    RecordUpsert,
    fold_index,
    merge_to_phases,
    parse_op,
)

Graph = Index
GraphOp = IndexOp
NodeUpsert = RecordUpsert
NodeDelete = RecordDelete
EdgeUpsert = LinkUpsert
EdgeDelete = LinkDelete
fold_graph = fold_index

__all__ = [
    "EdgeDelete",
    "EdgeUpsert",
    "Graph",
    "GraphOp",
    "NodeDelete",
    "NodeUpsert",
    "fold_graph",
    "merge_to_phases",
    "parse_op",
]
