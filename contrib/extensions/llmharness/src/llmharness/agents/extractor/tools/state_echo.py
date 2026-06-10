"""State echo for extractor error messages."""

from __future__ import annotations

from ..state import ExtractionState


def state_echo(state: ExtractionState) -> str:
    """One-line summary of the currently-folded graph for this firing."""
    if not state.pending_ops:
        return ""
    nodes = state.pending_graph.nodes
    edges = state.pending_graph.edges
    if not nodes:
        return ""
    last_id = max(nodes)
    last = nodes[last_id]
    return (
        f"{len(nodes)} node(s), {len(edges)} edge(s); "
        f"last accepted: id={last.id} kind={last.kind.value}"
    )


__all__ = ["state_echo"]
