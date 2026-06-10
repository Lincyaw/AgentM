"""Build the state-echo line for the extractor's three-section error /
advisory templates.

Kept beside the tool files (not promoted to the shared ``audit/``
namespace) because the shape here is extractor-specific:
``ExtractionState`` internals. Used for ADVISORY rendering only — never
to gate finalize.

Not an atom — pure functions, no MANIFEST, no install hook.
"""

from __future__ import annotations

from .state import ExtractionState


def state_echo(state: ExtractionState) -> str:
    """One-line summary of the currently-folded graph for this firing.

    Reads the folded view (recent prefix + this firing's ``pending_ops``)
    so the model can locate itself in the trajectory of edits: node
    count, edge count, and the highest-id node currently in the graph.
    Returns an empty string when this firing has applied no ops — the
    helper renders that as ``(empty)`` per the three-section template.
    """
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
