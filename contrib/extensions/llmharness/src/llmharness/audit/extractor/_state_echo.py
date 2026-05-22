"""Small helpers that build the state-echo + option suggestions for the
extractor's three-section error template.

Kept beside the tool files (not promoted to the shared ``audit/``
namespace) because every shape here is extractor-specific:
``ExtractionState`` internals, ``EdgeKind`` enum text, the passthrough
recovery recipe. The auditor side reuses ``format_witness_error`` but
not these specifics.

Not an atom — pure functions, no MANIFEST, no install hook.
"""

from __future__ import annotations

from .state import ExtractionState


def state_echo(state: ExtractionState) -> str:
    """One-line summary of the currently-accepted (pending) graph.

    Lists pending node count, pending edge count, and the last accepted
    node so the model can locate itself in the trajectory of edits.
    Returns an empty string when nothing has been accepted yet — the
    helper renders that as ``(empty)`` per the three-section template.
    """
    n_nodes = len(state._events_pending)
    n_edges = len(state._edges_pending)
    if n_nodes == 0:
        return ""
    last = state._events_pending[-1]
    return (
        f"{n_nodes} node(s), {n_edges} edge(s); "
        f"last accepted: id={last.id} kind={last.kind.value}"
    )


def chain_neighbours(state: ExtractionState, event_id: int) -> tuple[int | None, int | None]:
    """Return ``(in_neighbour_id, out_neighbour_id)`` for a passthrough.

    A passthrough event has exactly one predecessor and one successor in
    the pending in-firing graph; this returns those ids so the error
    message can name them in the options. Returns ``(None, None)`` when
    the event is not in the pending graph or is not a passthrough.
    """
    in_neighbour: int | None = None
    out_neighbour: int | None = None
    in_count = 0
    out_count = 0
    for ed in state._edges_pending:
        if ed.dst == event_id:
            in_count += 1
            in_neighbour = ed.src
        if ed.src == event_id:
            out_count += 1
            out_neighbour = ed.dst
    if in_count == 1 and out_count == 1:
        return in_neighbour, out_neighbour
    return None, None


__all__ = ["chain_neighbours", "state_echo"]
