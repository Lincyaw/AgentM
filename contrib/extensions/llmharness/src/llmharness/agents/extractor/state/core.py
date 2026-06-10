"""Per-firing in-memory ``ExtractionState`` for the v4 extractor.

Event-sourced incremental flow: each firing edits the graph through the
``apply_node_upsert`` / ``apply_node_delete`` / ``apply_edge_upsert`` /
``apply_edge_delete`` surface, which appends a :class:`GraphOp` to
``pending_ops`` and refolds ``pending_graph``. The state IS the output:
the adapter constructs one ``ExtractionState`` per firing, binds it into
the extractor tools, and reads ``events`` / ``edges`` / ``dropped_edges``
back after :meth:`ExtractionState.finalize` (called explicitly by the
``finalize_extraction`` terminator or on commit-on-stop).

Per-op validation:

1. **node shape** (:func:`_validate_event_shape`): ``id`` is an int >= 1,
   resolving against the folded view to an existing node, a node deleted
   in this firing, or the next free id. ``kind`` is a valid ``EventKind``,
   ``summary`` non-empty, ``source_turns`` non-empty.
2. **edge shape**: endpoints must exist in the folded view; ``kind`` is a
   valid ``EdgeKind``; ``data`` requires non-empty ``cited_entities``;
   ``ref`` requires a ``cited_quote`` that passes witness.
3. **witness** (:func:`witness_ref`): a ``ref`` edge's quote must appear
   (case+ws normalized substring) in at least one endpoint's source-turn
   text.

A failing op is rejected (the LLM gets the error in the tool result and
may retry, bounded by the caller's budget); accepted ops accumulate in
the op log. ``finalize`` reads this firing's emitted nodes + edges back
out of the folded view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmharness.graph.fold import Graph, fold_graph
from llmharness.graph.ops import (
    EdgeDelete,
    EdgeUpsert,
    GraphOp,
    NodeDelete,
    NodeUpsert,
)
from llmharness.schema import Edge, EdgeKind, Event
from llmharness.validation.enum_schema import EDGE_KIND_VALUES

from ..witness import witness_ref
from .validate import (
    _coerce_int,
    _compute_degree_warning,
    _validate_event_shape,
)


@dataclass
class ExtractionState:
    """Per-firing scratch space for the v3 extractor tool flow."""

    # turn_index -> raw turn text used for witness substring checks. The
    # adapter populates this from the trajectory window before spawning
    # the extractor child. Keys are absolute trajectory indices; values
    # are the rendered text content for that turn. Includes BOTH this
    # firing's window AND every turn referenced by ``recent_graph``
    # source_turns — required so external_refs can be witnessed.
    turn_texts: dict[int, str] = field(default_factory=dict)

    # The recent-graph slice presented to the extractor this firing.
    # ``external_refs[].to_recent_event_id`` cites one of these entries
    # by its global ``id``. Empty when no prior events exist (the
    # extractor must then emit only in-firing refs).
    recent_graph: tuple[Event, ...] = ()

    # The next available global event id. The adapter computes this as
    # ``max(branch_state.graph) + 1`` before spawning the extractor and
    # the prompt instructs the LLM to start numbering here. The
    # validator enforces ``ev.id >= next_event_id`` so this firing's
    # events occupy a fresh contiguous slice of the global id space.
    # Defaults to ``1`` for the very first firing (or for tests with no
    # prior history).
    next_event_id: int = 1

    # Frozen results — populated by ``finalize`` from the op log.
    events: tuple[Event, ...] = ()
    edges: tuple[Edge, ...] = ()
    dropped_edges: tuple[dict[str, Any], ...] = ()
    committed: bool = False

    # ------------------------------------------------------------------
    # Event-sourcing op log (2026-05-22 refactor).
    #
    # ``pending_ops`` is the authoritative record of every edit this
    # firing applied to the persistent graph; finalize hands the list
    # to the adapter, which persists each entry as an ``AUDIT_GRAPH_OP``
    # session entry. ``pending_graph`` caches the fold of
    # (recent_graph + recent_edges) + pending_ops so each ``apply_*``
    # call can validate against the *current* view (the LLM may edit
    # nodes that were emitted by prior firings, not just this firing's
    # work).
    #
    # ``recent_graph_dict`` / ``recent_edges_dict`` carry the prior-
    # firings view; the adapter populates them in branch order from the
    # scanned op log. Defaults are empty for the first firing, which has
    # no cumulative graph to start from.
    recent_graph_dict: dict[int, Event] = field(default_factory=dict)
    recent_edges_dict: dict[tuple[int, int, str], Edge] = field(default_factory=dict)
    pending_ops: list[GraphOp] = field(default_factory=list)
    pending_graph: Graph = field(default_factory=Graph)

    def __post_init__(self) -> None:
        # Seed ``recent_graph_dict`` from the ``recent_graph`` tuple when
        # callers passed only the tuple (tests do this; the adapter
        # populates the dict directly). Keeps the two views consistent so
        # the ``apply_*`` surface sees the same nodes.
        if self.recent_graph and not self.recent_graph_dict:
            self.recent_graph_dict = {ev.id: ev for ev in self.recent_graph}
        # First fold: the only ops at this point are the recent prefix,
        # so the result equals the recent graph view.
        if self.recent_graph_dict or self.recent_edges_dict or self.pending_ops:
            self._refold()
        # ``next_event_id`` intentionally NOT auto-derived from
        # ``recent_graph_dict``: it defaults to 1 even when
        # ``recent_graph`` is non-empty. The adapter sets next_event_id
        # explicitly for live sessions; tests can override per case.

    # ------------------------------------------------------------------
    # Public mutators

    def finalize(self) -> str | None:
        """Freeze pending state and commit.

        V4 (2026-05-24): finalize ALWAYS commits on a witness-valid,
        id-monotonic graph. The previous "no passthrough" hard reject
        is gone — a model that produces a chain shape no longer gets
        stuck retrying. Use :meth:`compute_degree_warning` to retrieve
        the soft advisory string (or ``None``) attached to the
        successful commit; the caller surfaces that to the model so it
        has feedback for the NEXT firing.

        Returns ``None`` on success; ``ExtractionState.committed`` is
        then ``True`` and ``events`` / ``edges`` / ``dropped_edges``
        are populated. The only remaining failure mode is calling
        ``finalize`` after it already succeeded.
        """
        if self.committed:
            return "finalize: firing already finalized"

        # The tool surface (apply_node_upsert / apply_edge_upsert) writes
        # only to the op log. Read this firing's emitted nodes + edges
        # back out of the folded view, scoped to ops the firing actually
        # appended (so prior-firing nodes that arrived via
        # ``recent_graph_dict`` don't get re-emitted as fresh audit
        # entries). An empty op log commits an empty graph.
        firing_node_ids = {op.id for op in self.pending_ops if isinstance(op, NodeUpsert)}
        firing_node_ids -= {op.id for op in self.pending_ops if isinstance(op, NodeDelete)}
        nodes, edges_view = self._folded_view()
        firing_events = [nodes[nid] for nid in sorted(firing_node_ids) if nid in nodes]
        firing_edges = [
            ed
            for (src, dst, _kind), ed in edges_view.items()
            if src in firing_node_ids or dst in firing_node_ids
        ]
        self.events = tuple(firing_events)
        self.edges = tuple(firing_edges)
        self.dropped_edges = ()
        self.committed = True
        return None

    def compute_degree_warning(self) -> str | None:
        """V4 soft advisory: return chain-link warning for the committed graph.

        Call after :meth:`finalize` succeeds. Inspects the FROZEN
        ``self.events`` / ``self.edges`` (not the pending lists) and
        returns either ``None`` or a short advisory string the caller
        attaches to the success result so the model gets feedback for
        the next firing. Never raises; never blocks.
        """
        return _compute_degree_warning(list(self.events), list(self.edges))

    def reset_pending(self) -> None:
        """Drop this firing's ops so the LLM can re-submit from scratch.

        Used by the ``reset_extraction`` tool when the LLM decides its
        accumulated graph is unrecoverable. Clears ``pending_ops`` and
        refolds back to the recent-graph prefix; prior-firing state
        (``recent_graph_dict`` / ``recent_edges_dict``) is untouched.
        """
        self.pending_ops = []
        self._refold()

    # ------------------------------------------------------------------
    # Event-sourcing apply_* surface.
    #
    # Each ``apply_*`` validates against the folded view
    # ``recent (union pending_graph)``, appends a :class:`GraphOp` to
    # ``pending_ops``, refolds ``pending_graph``, and returns a digest
    # dict on success or a string error on rejection. Edits may target
    # nodes from prior firings (anything in ``recent_graph_dict``): the
    # graph maintainer revises stale nodes and merges duplicates across
    # firings.

    def _refold(self) -> None:
        """Recompute ``pending_graph`` from recent + pending_ops.

        Recent state is materialised as a synthetic op prefix so the
        same :func:`fold_graph` semantics apply uniformly (insertion
        order, NodeDelete cascade, etc.). Building the prefix is O(R+E)
        where R is recent_graph_dict size and E is recent_edges_dict
        size; ``pending_ops`` is typically small per firing so the
        refold cost is dominated by recent.
        """
        prefix: list[GraphOp] = []
        for nid, ev in self.recent_graph_dict.items():
            prefix.append(
                NodeUpsert(
                    id=nid,
                    kind=ev.kind.value,
                    summary=ev.summary,
                    source_turns=tuple(ev.source_turns),
                    external_refs=ev.external_refs,
                )
            )
        for (src, dst, kind), ed in self.recent_edges_dict.items():
            prefix.append(
                EdgeUpsert(
                    src=src,
                    dst=dst,
                    kind=kind,
                    reason=ed.reason,
                    cited_entities=ed.cited_entities,
                    cited_quote=ed.cited_quote,
                    src_turns=ed.src_turns,
                    dst_turns=ed.dst_turns,
                )
            )
        self.pending_graph = fold_graph(prefix + self.pending_ops)

    def _ops_digest(self, op: str) -> dict[str, Any]:
        return {
            "ok": True,
            "op": op,
            "pending_ops": len(self.pending_ops),
            "graph_nodes": len(self.pending_graph.nodes),
            "graph_edges": len(self.pending_graph.edges),
        }

    def _folded_view(self) -> tuple[dict[int, Event], dict[tuple[int, int, str], Edge]]:
        """Convenience: return ``(nodes, edges)`` of the current folded graph.

        Always reflects ``recent union pending_ops`` because ``pending_graph``
        is kept in sync via :meth:`_refold` after every accepted op.
        """
        return self.pending_graph.nodes, self.pending_graph.edges

    def apply_node_upsert(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one ``NodeUpsert`` against the folded view.

        Accepts ids that are (a) already present in the folded graph
        (edit), (b) the next free id ``= max(folded_ids) + 1`` (append),
        or (c) any id previously removed in this firing via
        ``apply_node_delete`` — re-use after delete is allowed because
        the op log preserves the deletion order. The constraint is
        enforced against ``(folded union deleted-in-this-firing)``.
        """
        if self.committed:
            return "apply_node_upsert: firing already finalized"
        err, ev = _validate_event_shape(0, raw)
        if err is not None:
            return err
        assert ev is not None

        nodes, _edges = self._folded_view()
        deleted_in_firing = {op.id for op in self.pending_ops if isinstance(op, NodeDelete)}
        # Largest seen id includes recent_graph_dict (which seeded
        # pending_graph via _refold), pending NodeUpsert ids, and the
        # explicitly-deleted ids — so the LLM cannot "skip" over a
        # deleted slot to make the validator forget about it.
        max_seen = max(
            [0]
            + list(nodes.keys())
            + [op.id for op in self.pending_ops if isinstance(op, NodeUpsert)]
            + list(deleted_in_firing)
            + [self.next_event_id - 1]
        )
        allowed_ids = set(nodes.keys()) | deleted_in_firing | {max_seen + 1}
        if ev.id not in allowed_ids:
            return (
                f"apply_node_upsert: id={ev.id} is neither an existing node "
                f"in the folded graph (recent + pending), a node deleted in "
                f"this firing, nor the next available id ({max_seen + 1}). "
                "Pick one: edit an existing id, re-use a just-deleted id, or "
                "append at the next free slot."
            )

        op = NodeUpsert(
            id=ev.id,
            kind=ev.kind.value,
            summary=ev.summary,
            source_turns=tuple(ev.source_turns),
        )
        self.pending_ops.append(op)
        self._refold()
        return self._ops_digest("node_upsert")

    def apply_node_delete(self, node_id: int) -> dict[str, Any] | str:
        """Validate + apply one ``NodeDelete`` against the folded view.

        The node must exist in the current folded graph (recent +
        pending); deleting a node not present is a programmer error
        from the LLM's perspective. Edge cascade happens at fold time.
        """
        if self.committed:
            return "apply_node_delete: firing already finalized"
        nodes, _edges = self._folded_view()
        if node_id not in nodes:
            return (
                f"apply_node_delete: unknown node_id {node_id}. "
                f"Existing ids in the folded graph: {sorted(nodes.keys())}"
            )
        self.pending_ops.append(NodeDelete(id=node_id))
        self._refold()
        return self._ops_digest("node_delete")

    def apply_edge_upsert(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one ``EdgeUpsert`` against the folded view.

        Endpoint nodes are looked up in the folded view (recent +
        pending). Kind-specific witness rules:

        - ``kind='data'``: ``cited_entities`` is non-empty (validator
          rejects empty list; the witness pipeline at edge-construction
          time has historically been lax for this case because the
          downstream auditor still has the entities to work with).
        - ``kind='ref'``: ``cited_quote`` must appear (case+ws
          normalised) as a substring of at least one endpoint's
          source-turns text — same rule as the batch path
          (:func:`witness_ref`).
        """
        if self.committed:
            return "apply_edge_upsert: firing already finalized"

        src = _coerce_int(raw.get("src"))
        dst = _coerce_int(raw.get("dst"))
        kind_raw = raw.get("kind")
        if src is None or dst is None:
            return "apply_edge_upsert: 'src' and 'dst' must be integers"
        try:
            kind = EdgeKind(kind_raw)
        except ValueError:
            return f"apply_edge_upsert: kind {kind_raw!r} not in {EDGE_KIND_VALUES}"

        nodes, _edges = self._folded_view()
        if src not in nodes or dst not in nodes:
            return (
                f"apply_edge_upsert: src={src} and dst={dst} must both exist "
                f"in the folded graph. Existing ids: {sorted(nodes.keys())}"
            )
        src_event = nodes[src]
        dst_event = nodes[dst]

        cited_entities_raw = raw.get("cited_entities", [])
        cited_quote = str(raw.get("cited_quote", "") or "")
        if kind is EdgeKind.DATA:
            if not isinstance(cited_entities_raw, list) or not cited_entities_raw:
                return "apply_edge_upsert: kind='data' requires non-empty cited_entities"
            if any(not isinstance(e, str) or not e for e in cited_entities_raw):
                return "apply_edge_upsert: cited_entities must be non-empty strings"
            cited_entities = tuple(str(e) for e in cited_entities_raw)
        else:
            if not cited_quote:
                return "apply_edge_upsert: kind='ref' requires non-empty cited_quote"
            # Witness validation for ref: route through the same helper
            # used by the batch path so both contracts stay identical.
            src_text = self._concat_turn_texts(src_event.source_turns)
            dst_text = self._concat_turn_texts(dst_event.source_turns)
            werr = witness_ref(cited_quote, src_text, dst_text)
            if werr is not None:
                return f"apply_edge_upsert: {werr}"
            cited_entities = tuple(str(e) for e in (cited_entities_raw or []))

        reason = raw.get("reason", "")
        if not isinstance(reason, str):
            return "apply_edge_upsert: reason must be a string"

        op = EdgeUpsert(
            src=src,
            dst=dst,
            kind=kind.value,
            reason=reason,
            cited_entities=cited_entities,
            cited_quote=cited_quote,
            src_turns=tuple(src_event.source_turns),
            dst_turns=tuple(dst_event.source_turns),
        )
        self.pending_ops.append(op)
        self._refold()
        return self._ops_digest("edge_upsert")

    def apply_edge_delete(self, selector: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one ``EdgeDelete`` against the folded view.

        Selector MUST carry src, dst, and kind — the op-log contract
        requires the full key because ``(src, dst)`` alone is ambiguous
        across ``kind=data`` / ``kind=ref``.
        """
        if self.committed:
            return "apply_edge_delete: firing already finalized"

        src = _coerce_int(selector.get("src"))
        dst = _coerce_int(selector.get("dst"))
        kind_raw = selector.get("kind")
        if src is None or dst is None:
            return "apply_edge_delete: 'src' and 'dst' must be integers"
        if not isinstance(kind_raw, str) or not kind_raw:
            return "apply_edge_delete: 'kind' is required and must be a string"
        try:
            EdgeKind(kind_raw)
        except ValueError:
            return f"apply_edge_delete: kind {kind_raw!r} not in {EDGE_KIND_VALUES}"

        _nodes, edges = self._folded_view()
        if (src, dst, kind_raw) not in edges:
            return (
                f"apply_edge_delete: edge ({src}, {dst}, {kind_raw}) not found in the folded graph"
            )
        self.pending_ops.append(EdgeDelete(src=src, dst=dst, kind=kind_raw))
        self._refold()
        return self._ops_digest("edge_delete")

    def _concat_turn_texts(self, turn_indices: list[int] | tuple[int, ...]) -> str:
        # Missing turn texts contribute the empty string — the witness
        # check will then naturally fail rather than KeyError out.
        return " ".join(self.turn_texts.get(idx, "") for idx in turn_indices)


__all__ = ["ExtractionState"]
