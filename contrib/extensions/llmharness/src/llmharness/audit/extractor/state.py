"""Per-firing in-memory ``ExtractionState`` for the v3 extractor.

V3.1 (events-only single-tool flow): one ``submit_events`` call carries
the entire graph as a list of events with embedded ``refs[]``. The
state IS the output: the adapter constructs one ``ExtractionState`` per
firing, hands it to ``build_extractor_tools`` so the tool callback
closes over it, and reads ``events`` / ``edges`` / ``dropped_edges``
back after the child loop terminates.

The validation pipeline runs inside :meth:`ExtractionState.commit`:

1. **events shape**: ``id`` is an int >= ``next_event_id`` (the global
   cursor the adapter passed in), strictly increasing in submission
   order, and disjoint from any ``recent_graph`` entry's id. Each
   ``kind`` is a valid ``EventKind``, ``summary`` non-empty,
   ``source_turns`` non-empty.
2. **refs shape**: ``to`` must reference an earlier event id (``< self.id``,
   guaranteeing no cycles + time-order); ``kind`` is a valid ``EdgeKind``;
   ``data`` requires non-empty ``cited_entities``; ``ref`` requires
   non-empty ``cited_quote``.
3. **witness**: each ref's witnesses must appear (case+ws normalized
   substring) in BOTH the source-turn text of the referenced event and
   the source-turn text of the citing event.

If any **event-shape** check fails the whole submission is rejected
(LLM gets the error in the tool result and may retry, bounded by the
caller's attempt budget). If shape is fine but some **refs** fail
witness, those refs are recorded into ``dropped_edges`` and the events
+ surviving refs are accepted (design §4.f partial-success path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...schema import Edge, EdgeKind, Event, EventKind, ExternalRef
from .._enum_schema import EDGE_KIND_VALUES, EVENT_KIND_VALUES
from ..graph_fold import Graph, fold_graph
from ..graph_ops import (
    EdgeDelete,
    EdgeUpsert,
    GraphOp,
    NodeDelete,
    NodeUpsert,
)
from .witness import witness_data, witness_ref


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

    # Frozen results — populated by ``finalize`` (or by the legacy
    # one-shot ``commit`` for backwards compatibility with the v17 tests).
    events: tuple[Event, ...] = ()
    edges: tuple[Edge, ...] = ()
    dropped_edges: tuple[dict[str, Any], ...] = ()
    committed: bool = False

    # Pending accumulators for the multi-batch v18 flow. Each successful
    # ``commit_batch`` appends to these; ``finalize`` runs the
    # cross-graph degree check on the accumulated pending state and then
    # freezes them into the public ``events`` / ``edges`` tuples.
    _events_pending: list[Event] = field(default_factory=list)
    _edges_pending: list[Edge] = field(default_factory=list)
    _external_refs_pending: dict[int, list[ExternalRef]] = field(default_factory=dict)
    _dropped_pending: list[dict[str, Any]] = field(default_factory=list)

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
    # scanned op log. Defaults are empty for first-firing tests and the
    # legacy ``commit`` / ``commit_batch`` callers that never had a
    # cumulative graph to start from.
    recent_graph_dict: dict[int, Event] = field(default_factory=dict)
    recent_edges_dict: dict[tuple[int, int, str], Edge] = field(
        default_factory=dict
    )
    pending_ops: list[GraphOp] = field(default_factory=list)
    pending_graph: Graph = field(default_factory=Graph)

    def __post_init__(self) -> None:
        # Seed ``recent_graph_dict`` from the legacy ``recent_graph``
        # tuple when callers passed only the tuple (every existing test
        # does this; the adapter populates the dict directly). Keeps
        # the two views consistent so the new ``apply_*`` surface sees
        # the same nodes the legacy ``commit`` path does.
        if self.recent_graph and not self.recent_graph_dict:
            self.recent_graph_dict = {ev.id: ev for ev in self.recent_graph}
        # First fold: the only ops at this point are the recent prefix,
        # so the result equals the recent graph view.
        if (
            self.recent_graph_dict
            or self.recent_edges_dict
            or self.pending_ops
        ):
            self._refold()
        # ``next_event_id`` intentionally NOT auto-derived from
        # ``recent_graph_dict``: legacy callers and tests rely on the
        # default of 1 even when ``recent_graph`` is non-empty (the
        # legacy validation rule didn't reject id collisions with
        # recent). The adapter sets next_event_id explicitly for live
        # sessions; tests can override per case.

    # ------------------------------------------------------------------
    # Public mutators

    def commit_batch(self, events_payload: list[dict[str, Any]]) -> str | None:
        """Validate one batch and append to pending on success.

        The batch is validated against (already-accepted pending events
        + this batch). Hard errors (event-shape, id-sequence, ref-shape)
        leave the pending state untouched — the LLM may retry the batch.
        Witness failures drop the offending refs into ``_dropped_pending``
        and accept the events as in v3.1.

        The cross-graph degree invariant is NOT checked here; that runs
        in :meth:`finalize`. This lets the LLM submit events in chunks
        without paying for a global re-validation per chunk.
        """
        if self.committed:
            return "submit_events_batch: firing already finalized; no further batches accepted"
        return self._validate_and_append(events_payload)

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

        # v19 path. The new tool surface (apply_node_upsert /
        # apply_edge_upsert) only writes to the op log; ``_events_pending``
        # / ``_edges_pending`` stay empty. Read this firing's emitted
        # nodes + edges back out of the folded view, scoped to ops the
        # firing actually appended (so prior-firing nodes that arrived
        # via ``recent_graph_dict`` don't get re-emitted as fresh audit
        # entries).
        if not self._events_pending and self.pending_ops:
            firing_node_ids = {
                op.id for op in self.pending_ops if isinstance(op, NodeUpsert)
            }
            firing_node_ids -= {
                op.id for op in self.pending_ops if isinstance(op, NodeDelete)
            }
            nodes, edges_view = self._folded_view()
            firing_events = [
                nodes[nid] for nid in sorted(firing_node_ids) if nid in nodes
            ]
            firing_edges = [
                ed for (src, dst, _kind), ed in edges_view.items()
                if src in firing_node_ids or dst in firing_node_ids
            ]
            self.events = tuple(firing_events)
            self.edges = tuple(firing_edges)
            self.dropped_edges = ()
            self.committed = True
            return None

        if not self._events_pending:
            self.events = ()
            self.edges = ()
            self.dropped_edges = ()
            self.committed = True
            return None
        finalized: list[Event] = [
            Event(
                id=w.id,
                kind=w.kind,
                summary=w.summary,
                source_turns=w.source_turns,
                external_refs=tuple(self._external_refs_pending.get(w.id, [])),
            )
            for w in self._events_pending
        ]
        self.events = tuple(finalized)
        self.edges = tuple(self._edges_pending)
        self.dropped_edges = tuple(self._dropped_pending)
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
        """Drop pending batches so the LLM can re-submit from scratch.

        Used by the ``reset_extraction`` tool when the LLM decides its
        accumulated graph is unrecoverable (e.g. a finalize rejection
        on degree check that can't be fixed by appending more events).
        """
        self._events_pending = []
        self._edges_pending = []
        self._external_refs_pending = {}
        self._dropped_pending = []
        self.pending_ops = []
        self._refold()

    def upsert_node(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Insert or replace one pending event node by id.

        Legacy in-firing-only contract: ids must occupy a fresh slice
        starting at ``next_event_id`` — no cross-firing edits. Use
        :meth:`apply_node_upsert` for the event-sourcing surface that
        accepts edits targeting prior-firing nodes.
        """
        if self.committed:
            return "upsert_node: firing already finalized"
        err, ev = _validate_event_shape(len(self._events_pending), raw)
        if err is not None:
            return err.replace("submit_events", "upsert_node", 1)
        assert ev is not None
        idx, current = self._pending_event(ev.id)
        if current is None:
            min_id = (
                self._events_pending[-1].id + 1
                if self._events_pending
                else self.next_event_id
            )
            if ev.id < min_id:
                return f"upsert_node: node id {ev.id} must be >= {min_id}"
            self._events_pending.append(ev)
        else:
            self._events_pending[idx] = ev
        # Mirror into the op log so commit() can emit AUDIT_GRAPH_OP
        # entries for the firing.
        self.pending_ops.append(
            NodeUpsert(
                id=ev.id,
                kind=ev.kind.value,
                summary=ev.summary,
                source_turns=tuple(ev.source_turns),
            )
        )
        self._refold()
        return self._edit_digest("upsert_node")

    def delete_node(self, node_id: int) -> dict[str, Any] | str:
        """Delete one pending event node and all pending edges touching it."""
        if self.committed:
            return "delete_node: firing already finalized"
        idx, current = self._pending_event(node_id)
        if current is None:
            return f"delete_node: unknown node_id {node_id}"
        del self._events_pending[idx]
        self._edges_pending = [
            e for e in self._edges_pending if e.src != node_id and e.dst != node_id
        ]
        self._external_refs_pending.pop(node_id, None)
        self.pending_ops.append(NodeDelete(id=node_id))
        self._refold()
        return self._edit_digest("delete_node")

    def upsert_edge(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Insert or replace one pending edge by (src, dst, kind).

        Witness rule: kind='ref' requires ``cited_quote`` to appear in
        BOTH endpoints' source_turns text — same gate as the batch
        path and :meth:`apply_edge_upsert`. The legacy path used to
        skip this check, letting fabricated quotes through; closed
        here so all three surfaces share one contract.
        """
        if self.committed:
            return "upsert_edge: firing already finalized"
        err, edge = self._build_pending_edge(raw)
        if err is not None:
            return err
        assert edge is not None
        if edge.kind is EdgeKind.REF:
            src_text = self._concat_turn_texts(edge.src_turns)
            dst_text = self._concat_turn_texts(edge.dst_turns)
            werr = witness_ref(edge.cited_quote, src_text, dst_text)
            if werr is not None:
                return f"upsert_edge: {werr}"
        selector = {"src": edge.src, "dst": edge.dst, "kind": edge.kind.value}
        idx = self._pending_edge_index(selector)
        if idx is None:
            self._edges_pending.append(edge)
        else:
            self._edges_pending[idx] = edge
        self.pending_ops.append(
            EdgeUpsert(
                src=edge.src,
                dst=edge.dst,
                kind=edge.kind.value,
                reason=edge.reason,
                cited_entities=edge.cited_entities,
                cited_quote=edge.cited_quote,
                src_turns=edge.src_turns,
                dst_turns=edge.dst_turns,
            )
        )
        self._refold()
        return self._edit_digest("upsert_edge")

    def delete_edge(self, selector: dict[str, Any]) -> dict[str, Any] | str:
        """Delete one pending edge selected by ``(src, dst, kind)``.

        ``kind`` is **mandatory** — the op-log contract requires the
        full triple because the same ``(src, dst)`` pair may carry
        both a ``data`` and a ``ref`` edge. The original kind-less
        behaviour (first-match-in-pending-order) lost replay
        determinism: which edge got the EdgeDelete depended on
        insertion order, not on the selector.
        """
        if self.committed:
            return "delete_edge: firing already finalized"
        kind_raw = selector.get("kind")
        if not isinstance(kind_raw, str) or not kind_raw:
            return (
                "delete_edge: 'kind' is required. The op-log selector "
                "needs the full (src, dst, kind) triple — (src, dst) "
                "alone is ambiguous when both 'data' and 'ref' edges "
                "exist between the same pair."
            )
        idx = self._pending_edge_index(selector)
        if idx is None:
            return "delete_edge: edge not found"
        edge = self._edges_pending[idx]
        del self._edges_pending[idx]
        self.pending_ops.append(
            EdgeDelete(src=edge.src, dst=edge.dst, kind=edge.kind.value)
        )
        self._refold()
        return self._edit_digest("delete_edge")

    # ------------------------------------------------------------------
    # Event-sourcing apply_* surface (2026-05-22 refactor).
    #
    # Each ``apply_*`` validates against the folded view
    # ``recent (union pending_graph)``, appends a :class:`GraphOp` to
    # ``pending_ops``, refolds ``pending_graph``, and returns a digest
    # dict on success or a string error on rejection. Unlike the legacy
    # ``upsert_node`` / ``delete_node`` / ``upsert_edge`` / ``delete_edge``
    # methods, these accept edits that target nodes from prior firings
    # (anything in ``recent_graph_dict``) — that is the whole point of
    # the event-sourcing refactor: the graph maintainer can revise
    # stale nodes and merge duplicates across firings.

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

    def _witness_in_turn_texts(
        self, quote: str, turn_indices: tuple[int, ...] | list[int]
    ) -> bool:
        """Substring-check ``quote`` against concatenated ``turn_texts``.

        Uses the same case+whitespace normalisation as :mod:`witness` so
        the gate here is identical to the one applied to in-firing refs
        in :meth:`_validate_and_append`. Missing turn texts contribute
        the empty string — the check then naturally fails.
        """
        from .witness import normalize as _normalize

        concat = " ".join(self.turn_texts.get(t, "") for t in turn_indices)
        return _normalize(quote) in _normalize(concat)

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
            return err.replace("submit_events", "apply_node_upsert", 1)
        assert ev is not None

        nodes, _edges = self._folded_view()
        deleted_in_firing = {
            op.id for op in self.pending_ops if isinstance(op, NodeDelete)
        }
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
          normalised) as a substring of BOTH the src node's source-
          turns text AND the dst node's source-turns text — same rule
          as the batch path (:func:`witness_ref`). The atomic path
          originally checked src only, letting a quote that exists
          only in src text slip through; that divergence was a hole
          the batch contract closed and we close here too.
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
                return (
                    "apply_edge_upsert: kind='data' requires non-empty "
                    "cited_entities"
                )
            if any(not isinstance(e, str) or not e for e in cited_entities_raw):
                return (
                    "apply_edge_upsert: cited_entities must be non-empty strings"
                )
            cited_entities = tuple(str(e) for e in cited_entities_raw)
        else:
            if not cited_quote:
                return (
                    "apply_edge_upsert: kind='ref' requires non-empty cited_quote"
                )
            # Witness validation for ref: the verbatim quote must appear
            # in BOTH the src and dst nodes' source_turns text (same
            # rule as :func:`witness_ref` in the batch path). Routing
            # through that helper keeps the two contracts byte-identical
            # so a quote accepted by one path is accepted by the other.
            src_text = self._concat_turn_texts(src_event.source_turns)
            dst_text = self._concat_turn_texts(dst_event.source_turns)
            werr = witness_ref(cited_quote, src_text, dst_text)
            if werr is not None:
                return f"apply_edge_upsert: {werr}"
            cited_entities = tuple(
                str(e) for e in (cited_entities_raw or [])
            )

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
                f"apply_edge_delete: edge ({src}, {dst}, {kind_raw}) not "
                "found in the folded graph"
            )
        self.pending_ops.append(EdgeDelete(src=src, dst=dst, kind=kind_raw))
        self._refold()
        return self._ops_digest("edge_delete")

    # ------------------------------------------------------------------
    # Legacy single-shot API (kept for v17 tests and direct callers).

    def commit(self, events_payload: list[dict[str, Any]]) -> str | None:
        """Legacy one-shot commit — validate, append, freeze in one call.

        Kept for the v17 test suite and direct callers. Skips the
        ``finalize`` degree check (passthrough rejection) — that's only
        enforced via the new ``commit_batch`` + ``finalize`` flow used
        by ``submit_events_batch``. Calling ``commit`` twice returns
        the "already committed" error as it did in v17.
        """
        if self.committed:
            return "submit_events: already committed; one submission per firing"
        err = self._validate_and_append(events_payload)
        if err is not None:
            return err
        # Freeze without the cross-graph degree check — preserves the
        # v17 contract that ``commit`` accepts any chain-shaped graph.
        finalized: list[Event] = [
            Event(
                id=w.id,
                kind=w.kind,
                summary=w.summary,
                source_turns=w.source_turns,
                external_refs=tuple(self._external_refs_pending.get(w.id, [])),
            )
            for w in self._events_pending
        ]
        self.events = tuple(finalized)
        self.edges = tuple(self._edges_pending)
        self.dropped_edges = tuple(self._dropped_pending)
        self.committed = True
        return None

    # ------------------------------------------------------------------
    # Shared validation core

    def _validate_and_append(
        self, events_payload: list[dict[str, Any]]
    ) -> str | None:
        """Validate one batch and (atomically) append to pending lists.

        Ref targets may point at events from previous batches (already
        in ``_events_pending``) OR earlier events in this same batch.
        Witness failures drop the offending ref into ``_dropped_pending``
        and accept the event. Any hard-reject error (shape, id-sequence,
        ref-shape) returns without mutating pending state — the LLM may
        resubmit only the rejected batch on retry.
        """
        # Pass 1: validate event shapes + collect into a working list.
        if not isinstance(events_payload, list):
            return "submit_events: 'events' must be an array"
        working: list[Event] = []
        for idx, raw in enumerate(events_payload):
            if not isinstance(raw, dict):
                return f"submit_events: events[{idx}] must be an object"
            err, ev = _validate_event_shape(idx, raw)
            if err is not None:
                return err
            assert ev is not None
            working.append(ev)

        # Pass 2: cross-event id check. Ids are global — must continue
        # the sequence from the highest id we've already accepted (or
        # from next_event_id if no prior batch landed), and strictly
        # increasing within the batch.
        cursor_start = (
            self._events_pending[-1].id
            if self._events_pending
            else self.next_event_id - 1
        )
        prev_id = cursor_start
        for idx, ev in enumerate(working):
            if ev.id <= cursor_start and not self._events_pending:
                return (
                    f"submit_events: events[{idx}].id={ev.id} is below "
                    f"next_event_id={self.next_event_id}. This firing's events "
                    "must continue the global id sequence — start at "
                    f"{self.next_event_id} and increment from there."
                )
            if ev.id <= prev_id:
                return (
                    f"submit_events: events[{idx}].id={ev.id} is not strictly "
                    f"greater than the previous event's id ({prev_id}). Ids "
                    "must be strictly increasing in submission order so that "
                    "refs.to references resolve unambiguously to earlier "
                    "events."
                )
            prev_id = ev.id

        # Events by id covers BOTH prior batches and this batch — refs
        # can point to either.
        events_by_id: dict[int, Event] = {ev.id: ev for ev in self._events_pending}
        for ev in working:
            events_by_id[ev.id] = ev

        # Pass 3: refs + external_refs.
        accepted_edges: list[Edge] = []
        accepted_external: dict[int, list[ExternalRef]] = {ev.id: [] for ev in working}
        dropped: list[dict[str, Any]] = []
        recent_n = len(self.recent_graph)
        recent_ids: set[int] = {e.id for e in self.recent_graph}
        for idx, (raw_event, ev) in enumerate(
            zip(events_payload, working, strict=True)
        ):
            refs_raw = raw_event.get("refs", [])
            if refs_raw is None:
                refs_raw = []
            if not isinstance(refs_raw, list):
                return f"submit_events: events[{idx}].refs must be an array"
            ext_raw = raw_event.get("external_refs", [])
            if ext_raw is None:
                ext_raw = []
            if not isinstance(ext_raw, list):
                return (
                    f"submit_events: events[{idx}].external_refs must be an array"
                )
            # Connection check: every non-genesis event must cite at
            # least one parent. "Genesis" means the very first event of
            # the whole case — no prior batches AND no recent_graph.
            has_priors = (
                idx >= 1
                or len(self._events_pending) > 0
                or len(self.recent_graph) > 0
            )
            if has_priors and not refs_raw and not ext_raw:
                candidates_list = list(self._events_pending) + [
                    w for w in working if w.id < ev.id
                ]
                candidates = ", ".join(
                    f"{{id:{c.id}, kind:{c.kind.value}, summary:{c.summary[:60]!r}}}"
                    for c in candidates_list
                )
                return (
                    f"submit_events: events[{idx}] (id={ev.id}) has no refs "
                    "and no external_refs. The genesis exemption only applies "
                    "to the very first event of the whole case (firing 1, "
                    "first event, empty recent_graph). Every other event must "
                    "cite at least one earlier event.\n"
                    f"Earlier-event candidates: [{candidates}].\n"
                    f"recent_graph has {recent_n} prior event(s) available via "
                    "external_refs[].to_recent_event_id (the .id field of a "
                    "recent_graph entry).\n"
                    f"Each ref needs: {{to:<earlier_id>, kind:'data'|'ref', "
                    "reason:<short>, and EITHER cited_entities:[...] OR "
                    "cited_quote:'...'}}. Witnesses must appear in BOTH the "
                    "cited event's source_turns text and this event's "
                    "source_turns text."
                )
            for ridx, raw_ref in enumerate(refs_raw):
                if not isinstance(raw_ref, dict):
                    return (
                        f"submit_events: events[id={ev.id}].refs[{ridx}] must be "
                        "an object"
                    )
                err = _validate_ref_shape(ev.id, ridx, raw_ref, events_by_id)
                if err is not None:
                    return err
                src_event = events_by_id[int(raw_ref["to"])]
                kind = EdgeKind(raw_ref["kind"])
                src_text = self._concat_turn_texts(src_event.source_turns)
                dst_text = self._concat_turn_texts(ev.source_turns)
                cited_entities = list(raw_ref.get("cited_entities", []) or [])
                cited_quote = str(raw_ref.get("cited_quote", "") or "")
                if kind is EdgeKind.DATA:
                    werr = witness_data(cited_entities, src_text, dst_text)
                else:
                    werr = witness_ref(cited_quote, src_text, dst_text)
                if werr is not None:
                    dropped.append(
                        {
                            "src": src_event.id,
                            "dst": ev.id,
                            "kind": kind.value,
                            "last_error": werr,
                        }
                    )
                    continue
                accepted_edges.append(
                    Edge(
                        src=src_event.id,
                        dst=ev.id,
                        kind=kind,
                        reason=str(raw_ref.get("reason", "")),
                        src_turns=tuple(src_event.source_turns),
                        dst_turns=tuple(ev.source_turns),
                        cited_entities=tuple(cited_entities),
                        cited_quote=cited_quote,
                    )
                )

            for ridx, raw_ref in enumerate(ext_raw):
                if not isinstance(raw_ref, dict):
                    return (
                        f"submit_events: events[id={ev.id}].external_refs"
                        f"[{ridx}] must be an object"
                    )
                err = _validate_external_ref_shape(ev.id, ridx, raw_ref, recent_ids)
                if err is not None:
                    return err
                ext_event_id = int(raw_ref["to_recent_event_id"])
                src_ext = next(
                    (e for e in self.recent_graph if e.id == ext_event_id),
                    None,
                )
                assert src_ext is not None  # validator above guarantees membership
                kind = EdgeKind(raw_ref["kind"])
                src_text = self._concat_turn_texts(src_ext.source_turns)
                dst_text = self._concat_turn_texts(ev.source_turns)
                cited_entities = list(raw_ref.get("cited_entities", []) or [])
                cited_quote = str(raw_ref.get("cited_quote", "") or "")
                if kind is EdgeKind.DATA:
                    werr = witness_data(cited_entities, src_text, dst_text)
                else:
                    werr = witness_ref(cited_quote, src_text, dst_text)
                if werr is not None:
                    dropped.append(
                        {
                            "src": f"recent_graph_event#{ext_event_id}",
                            "dst": ev.id,
                            "kind": kind.value,
                            "last_error": werr,
                        }
                    )
                    continue
                accepted_external[ev.id].append(
                    ExternalRef(
                        to_recent_event_id=ext_event_id,
                        kind=kind,
                        reason=str(raw_ref.get("reason", "")),
                        cited_entities=tuple(cited_entities),
                        cited_quote=cited_quote,
                    )
                )

        # Atomically extend pending state — all-or-nothing per batch.
        self._events_pending.extend(working)
        self._edges_pending.extend(accepted_edges)
        for eid, refs in accepted_external.items():
            self._external_refs_pending.setdefault(eid, []).extend(refs)
        self._dropped_pending.extend(dropped)
        # Mirror the batch into the op log so commit() can persist
        # AUDIT_GRAPH_OP entries even when the LLM used the legacy
        # ``submit_events_batch`` flow. external_refs are folded onto
        # each event's NodeUpsert so the op-log round-trip preserves
        # cross-firing connectivity — matches the eventual
        # ``finalize()`` Event(...) construction (line ~273 in this
        # file) that attaches them as event-level fields.
        for ev in working:
            self.pending_ops.append(
                NodeUpsert(
                    id=ev.id,
                    kind=ev.kind.value,
                    summary=ev.summary,
                    source_turns=tuple(ev.source_turns),
                    external_refs=tuple(accepted_external.get(ev.id, [])),
                )
            )
        for ed in accepted_edges:
            self.pending_ops.append(
                EdgeUpsert(
                    src=ed.src,
                    dst=ed.dst,
                    kind=ed.kind.value,
                    reason=ed.reason,
                    cited_entities=ed.cited_entities,
                    cited_quote=ed.cited_quote,
                    src_turns=ed.src_turns,
                    dst_turns=ed.dst_turns,
                )
            )
        self._refold()
        return None

    def _concat_turn_texts(self, turn_indices: list[int] | tuple[int, ...]) -> str:
        # Missing turn texts contribute the empty string — the witness
        # check will then naturally fail rather than KeyError out.
        return " ".join(self.turn_texts.get(idx, "") for idx in turn_indices)

    def _pending_event(self, event_id: int) -> tuple[int, Event | None]:
        for idx, event in enumerate(self._events_pending):
            if event.id == event_id:
                return idx, event
        return -1, None

    def _pending_edge_index(self, selector: dict[str, Any]) -> int | None:
        src = _coerce_int(selector.get("src"))
        dst = _coerce_int(selector.get("dst"))
        kind = selector.get("kind")
        for idx, edge in enumerate(self._edges_pending):
            if src is not None and edge.src != src:
                continue
            if dst is not None and edge.dst != dst:
                continue
            if kind is not None and edge.kind.value != str(kind):
                continue
            return idx
        return None

    def _build_pending_edge(self, raw: dict[str, Any]) -> tuple[str | None, Edge | None]:
        src = _coerce_int(raw.get("src"))
        dst = _coerce_int(raw.get("dst"))
        kind_raw = raw.get("kind")
        if src is None or dst is None:
            return "upsert_edge: 'src' and 'dst' must be integers", None
        src_event = self._pending_event(src)[1]
        dst_event = self._pending_event(dst)[1]
        if src_event is None or dst_event is None:
            return f"upsert_edge: src={src} and dst={dst} must both be pending nodes", None
        try:
            kind = EdgeKind(kind_raw)
        except ValueError:
            return f"upsert_edge: kind {kind_raw!r} not in {EDGE_KIND_VALUES}", None
        cited_entities = raw.get("cited_entities", [])
        cited_quote = str(raw.get("cited_quote", "") or "")
        if kind is EdgeKind.DATA:
            if not isinstance(cited_entities, list) or not cited_entities:
                return "upsert_edge: kind='data' requires non-empty cited_entities", None
            if any(not isinstance(e, str) or not e for e in cited_entities):
                return "upsert_edge: cited_entities must be non-empty strings", None
        else:
            if not cited_quote:
                return "upsert_edge: kind='ref' requires non-empty cited_quote", None
        reason = raw.get("reason", "")
        if not isinstance(reason, str):
            return "upsert_edge: reason must be a string", None
        return None, Edge(
            src=src,
            dst=dst,
            kind=kind,
            reason=reason,
            src_turns=tuple(src_event.source_turns),
            dst_turns=tuple(dst_event.source_turns),
            cited_entities=tuple(cited_entities or []),
            cited_quote=cited_quote,
        )

    def _edit_digest(self, op: str) -> dict[str, Any]:
        return {
            "ok": True,
            "op": op,
            "pending_nodes": len(self._events_pending),
            "pending_edges": len(self._edges_pending),
            "pending_dropped": len(self._dropped_pending),
        }


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _validate_event_shape(idx: int, raw: dict[str, Any]) -> tuple[str | None, Event | None]:
    eid_raw = raw.get("id")
    kind_raw = raw.get("kind")
    summary_raw = raw.get("summary")
    source_turns_raw = raw.get("source_turns")

    if isinstance(eid_raw, bool) or not isinstance(eid_raw, int):
        return f"submit_events: events[{idx}].id must be an integer", None
    if eid_raw < 1:
        return f"submit_events: events[{idx}].id must be >= 1; got {eid_raw}", None
    if not isinstance(kind_raw, str):
        return f"submit_events: events[{idx}].kind must be a string", None
    try:
        kind = EventKind(kind_raw)
    except ValueError:
        return (
            f"submit_events: events[{idx}].kind {kind_raw!r} not in {EVENT_KIND_VALUES}",
            None,
        )
    if not isinstance(summary_raw, str) or not summary_raw.strip():
        return f"submit_events: events[{idx}].summary must be a non-empty string", None
    if not isinstance(source_turns_raw, list) or not source_turns_raw:
        return (
            f"submit_events: events[{idx}].source_turns must be a non-empty "
            "array of integers",
            None,
        )
    source_turns: list[int] = []
    for t in source_turns_raw:
        if isinstance(t, bool) or not isinstance(t, int):
            return (
                f"submit_events: events[{idx}].source_turns contains "
                f"non-integer entry {t!r}",
                None,
            )
        source_turns.append(t)
    return None, Event(id=eid_raw, kind=kind, summary=summary_raw, source_turns=source_turns)


def _validate_ref_shape(
    self_event_id: int,
    ridx: int,
    raw: dict[str, Any],
    events_by_id: dict[int, Event],
) -> str | None:
    to_raw = raw.get("to")
    kind_raw = raw.get("kind")

    if isinstance(to_raw, bool) or not isinstance(to_raw, int):
        return (
            f"submit_events: events[id={self_event_id}].refs[{ridx}].to must be "
            "an integer"
        )
    if to_raw not in events_by_id:
        return (
            f"submit_events: events[id={self_event_id}].refs[{ridx}].to={to_raw} "
            "does not reference any submitted event id"
        )
    if to_raw >= self_event_id:
        return (
            f"submit_events: events[id={self_event_id}].refs[{ridx}].to={to_raw} "
            f"must reference an EARLIER event (< {self_event_id}); refs only flow "
            "forward in time"
        )
    if not isinstance(kind_raw, str):
        return (
            f"submit_events: events[id={self_event_id}].refs[{ridx}].kind must "
            "be a string"
        )
    try:
        kind = EdgeKind(kind_raw)
    except ValueError:
        return (
            f"submit_events: events[id={self_event_id}].refs[{ridx}].kind "
            f"{kind_raw!r} not in {EDGE_KIND_VALUES}"
        )

    cited_entities = raw.get("cited_entities", [])
    cited_quote = raw.get("cited_quote", "")
    if kind is EdgeKind.DATA:
        if not isinstance(cited_entities, list) or not cited_entities:
            return (
                f"submit_events: events[id={self_event_id}].refs[{ridx}] kind="
                "'data' requires non-empty cited_entities"
            )
        for e in cited_entities:
            if not isinstance(e, str) or not e:
                return (
                    f"submit_events: events[id={self_event_id}].refs[{ridx}]."
                    "cited_entities must be non-empty strings"
                )
    else:  # EdgeKind.REF
        if not isinstance(cited_quote, str) or not cited_quote:
            return (
                f"submit_events: events[id={self_event_id}].refs[{ridx}] kind="
                "'ref' requires non-empty cited_quote"
            )
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        return (
            f"submit_events: events[id={self_event_id}].refs[{ridx}].reason "
            "must be a string"
        )
    return None


def _validate_external_ref_shape(
    self_event_id: int,
    ridx: int,
    raw: dict[str, Any],
    recent_ids: set[int],
) -> str | None:
    to_raw = raw.get("to_recent_event_id")
    kind_raw = raw.get("kind")

    if isinstance(to_raw, bool) or not isinstance(to_raw, int):
        return (
            f"submit_events: events[id={self_event_id}].external_refs[{ridx}]"
            ".to_recent_event_id must be an integer"
        )
    if to_raw not in recent_ids:
        sorted_ids = sorted(recent_ids)
        return (
            f"submit_events: events[id={self_event_id}].external_refs[{ridx}]"
            f".to_recent_event_id={to_raw} not found in recent_graph "
            f"(available ids: {sorted_ids}). Copy the .id field of a "
            "recent_graph entry verbatim — not its array position."
        )
    if not isinstance(kind_raw, str):
        return (
            f"submit_events: events[id={self_event_id}].external_refs[{ridx}]"
            ".kind must be a string"
        )
    try:
        kind = EdgeKind(kind_raw)
    except ValueError:
        return (
            f"submit_events: events[id={self_event_id}].external_refs[{ridx}]"
            f".kind {kind_raw!r} not in {EDGE_KIND_VALUES}"
        )

    cited_entities = raw.get("cited_entities", [])
    cited_quote = raw.get("cited_quote", "")
    if kind is EdgeKind.DATA:
        if not isinstance(cited_entities, list) or not cited_entities:
            return (
                f"submit_events: events[id={self_event_id}].external_refs"
                f"[{ridx}] kind='data' requires non-empty cited_entities"
            )
        for e in cited_entities:
            if not isinstance(e, str) or not e:
                return (
                    f"submit_events: events[id={self_event_id}].external_refs"
                    f"[{ridx}].cited_entities must be non-empty strings"
                )
    else:
        if not isinstance(cited_quote, str) or not cited_quote:
            return (
                f"submit_events: events[id={self_event_id}].external_refs"
                f"[{ridx}] kind='ref' requires non-empty cited_quote"
            )
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        return (
            f"submit_events: events[id={self_event_id}].external_refs"
            f"[{ridx}].reason must be a string"
        )
    return None


def _compute_degree_warning(
    events: list[Event],
    edges: list[Edge],
) -> str | None:
    """V4 soft advisory: flag consecutive ``(in=1, out=1)`` chain links.

    Returns ``None`` when there are no chain-link events worth nudging
    the model about, or a short advisory string naming the offending
    event ids and suggesting a remediation. NEVER raises and is NEVER
    consulted to block finalize — :meth:`ExtractionState.finalize`
    always commits a witness-valid graph and the caller surfaces the
    warning (if any) on the SUCCESSFUL tool result so the model gets
    feedback for the next firing.

    Detection rule: an event whose in-degree is 1 AND out-degree is 1
    is a chain link. A natural, well-shaped trace
    (``task → act → hyp → act → concl``) has chain links in the
    middle and that's fine; this helper exists to nudge the model
    when chain links accumulate AND the linear stretch could be
    coalesced into one ``act`` or split by a branch event the model
    forgot to emit.

    Only in-firing edges count toward degree. External refs are
    intentionally excluded — the aggregator stitches them later.
    """
    if len(events) <= 1:
        return None
    in_deg: dict[int, int] = {ev.id: 0 for ev in events}
    out_deg: dict[int, int] = {ev.id: 0 for ev in events}
    for ed in edges:
        if ed.dst in in_deg:
            in_deg[ed.dst] += 1
        if ed.src in out_deg:
            out_deg[ed.src] += 1
    chain_links: list[Event] = [
        ev for ev in events if in_deg[ev.id] == 1 and out_deg[ev.id] == 1
    ]
    if not chain_links:
        return None
    lines = [
        f"  event[{ev.id}] kind={ev.kind.value} "
        f"'{ev.summary[:70]}': in=1, out=1"
        for ev in chain_links
    ]
    return (
        f"Soft warning: {len(chain_links)} chain-link event(s) "
        "(in-degree=1 AND out-degree=1) detected. If two adjacent "
        "``act`` nodes have nothing branching between them, consider "
        "merging them into one coalesced ``act`` (record every probe "
        "and result in time order in the summary). If a real "
        "``hyp`` / ``dec`` reasoning move was made between them but "
        "you didn't emit a node for it, add one in a follow-up "
        "firing. Aim for compact graphs but do NOT fabricate refs "
        "just to satisfy this heuristic.\n"
        "Chain-link events:\n"
        + "\n".join(lines)
    )


__all__ = ["ExtractionState", "_compute_degree_warning"]
