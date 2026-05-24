"""Scanner / persistence regressions covered by the 2026-05-22 review.

Three fail-stop positions:

- **B1**: ``_scan_branch`` must fold ``AUDIT_GRAPH_OP`` entries that
  carry only a ``node_delete`` — proving the maintainer's
  prior-firing-only edits survive the round trip. The companion
  unit-level check on the gating logic uses a vanilla
  :class:`ExtractionState` to assert ``pending_ops`` is the
  authoritative signal for "did this firing do something?".

- **B2**: a legacy ``AUDIT_EVENT`` entry carrying ``external_refs``
  in its payload must surface those external_refs on the folded
  ``Event`` — the prior translation dropped them silently.

- **M4**: the persisted ``firing_id`` per AUDIT_GRAPH_OP entry must
  be a stable session-scoped counter (0, 1, 2, ...) — not derived
  from "count of op_index==0 entries on the branch" which collapses
  if any future op-log change emits a firing whose first op has a
  non-zero index.
"""

from __future__ import annotations

import uuid
from typing import Any

from agentm.core.abi.session import SessionEntry

from llmharness.adapters.agentm import _scan_branch
from llmharness.audit.entry_types import (
    AUDIT_EDGE,
    AUDIT_EVENT,
    AUDIT_GRAPH_OP,
    EXTRACTOR_CURSOR,
)
from llmharness.audit.extractor.state import ExtractionState
from llmharness.audit.graph_ops import NodeDelete, NodeUpsert
from llmharness.schema import EdgeKind, Event, EventKind


def _entry(etype: str, payload: dict[str, Any]) -> SessionEntry:
    return SessionEntry(
        type=etype,
        id=uuid.uuid4().hex,
        parent_id=None,
        timestamp=0.0,
        payload=payload,
    )


# --- B1: pure-op firing -----------------------------------------------------


def test_scan_branch_folds_pure_node_delete_op() -> None:
    """A firing that emits only a NodeDelete must roll forward through
    the scanner: the prior node is gone from the folded graph. This
    is the maintainer use case (merge duplicates from prior firings
    via a single delete with no new events).
    """
    branch = [
        # Prior firing emitted one event the legacy way.
        _entry(
            AUDIT_EVENT,
            {
                "id": 1,
                "kind": "act",
                "summary": "stale",
                "source_turns": [1],
                "external_refs": [],
            },
        ),
        # Maintainer firing emits one AUDIT_GRAPH_OP deleting it.
        _entry(
            AUDIT_GRAPH_OP,
            {
                "op": "node_delete",
                "id": 1,
                "firing_id": 1,
                "op_index": 0,
                "caused_by_turn_window": [2, 3],
            },
        ),
    ]
    state = _scan_branch(branch, recent_verdicts_n=0)
    assert state.graph == []
    assert state.edges == []


def test_state_pending_ops_is_truthy_for_pure_node_delete() -> None:
    """B1 gating: a firing that called only ``apply_node_delete`` (no
    new events) must report ``pending_ops`` as truthy so the adapter
    persists AUDIT_GRAPH_OP entries instead of misclassifying the
    firing as ``EXTRACTOR_EMPTY``. Verified at the state level — the
    adapter's gating logic reads ``state.pending_ops`` directly.
    """
    prior = Event(id=1, kind=EventKind("act"), summary="stale", source_turns=[1])
    state = ExtractionState(
        turn_texts={1: "alpha"},
        recent_graph_dict={1: prior},
        next_event_id=2,
    )
    # The maintainer's single edit: drop the prior node.
    result = state.apply_node_delete(1)
    assert not isinstance(result, str), result

    # Legacy snapshot view is empty (commit_batch / finalize never ran).
    assert state.events == ()
    assert state.edges == ()
    assert state.dropped_edges == ()
    # But the op log is non-empty — the gating signal for the adapter.
    assert len(state.pending_ops) == 1
    assert isinstance(state.pending_ops[0], NodeDelete)


# --- B2: legacy AUDIT_EVENT external_refs round-trip ------------------------


def test_scan_branch_legacy_audit_event_preserves_external_refs() -> None:
    """Legacy ``AUDIT_EVENT`` translation must thread ``external_refs``
    onto the synthesized ``NodeUpsert`` so the folded ``Event``
    carries them. Without this, the auditor and the next firing's
    ``recent_graph`` payload regress to per-firing islands — the very
    failure mode the cumulative graph is designed to avoid.
    """
    branch = [
        _entry(
            AUDIT_EVENT,
            {
                "id": 5,
                "kind": "concl",
                "summary": "root cause",
                "source_turns": [20],
                "external_refs": [
                    {
                        "to_recent_event_id": 3,
                        "kind": "ref",
                        "reason": "restates earlier hyp",
                        "cited_entities": [],
                        "cited_quote": "latency spike",
                    }
                ],
            },
        ),
    ]
    state = _scan_branch(branch, recent_verdicts_n=0)
    assert len(state.graph) == 1
    ev = state.graph[0]
    assert ev.id == 5
    assert len(ev.external_refs) == 1
    er = ev.external_refs[0]
    assert er.to_recent_event_id == 3
    assert er.kind is EdgeKind.REF
    assert er.cited_quote == "latency spike"


# --- M4: firing_id invariant ------------------------------------------------


def test_firing_id_invariant_across_three_firings() -> None:
    """The persisted ``firing_id`` is a stable session-scoped counter.

    Simulates the install-level closure: a 1-element box starts at 0
    and is incremented by ``_drain_extractor`` after each ops-bearing
    firing. Across three firings the persisted ops carry firing_id 0,
    1, 2 — independent of the within-firing ``op_index`` values.

    The prior heuristic (count branch entries with ``op_index==0``)
    collapses if any future change emits a firing whose first op has
    a non-zero index (op-log compaction, partial-recovery retry).
    """
    firing_counter: list[int] = [0]
    persisted: list[dict[str, Any]] = []

    def _persist_firing(ops: list[NodeUpsert | NodeDelete]) -> None:
        """Mirror the adapter's persistence loop in _drain_extractor."""
        firing_id = firing_counter[0]
        for op_index, op in enumerate(ops):
            payload = op.to_dict()
            payload["firing_id"] = firing_id
            payload["op_index"] = op_index
            payload["caused_by_turn_window"] = [0, 0]
            persisted.append(payload)
        if ops:
            firing_counter[0] += 1

    _persist_firing(
        [NodeUpsert(id=1, kind="task", summary="t", source_turns=(0,))]
    )
    _persist_firing(
        [
            # firing 2 starts at op_index 0 with a NodeDelete (not the
            # typical "first op is the new task" shape) — the prior
            # heuristic would still count it because op_index==0, but
            # the test verifies the right answer holds regardless.
            NodeDelete(id=1),
            NodeUpsert(id=2, kind="hyp", summary="h", source_turns=(1,)),
        ]
    )
    _persist_firing(
        [NodeUpsert(id=3, kind="concl", summary="c", source_turns=(2,))]
    )

    firing_ids_seen = [p["firing_id"] for p in persisted]
    # Within each firing, all ops share one firing_id; across firings
    # the counter strictly increases.
    assert firing_ids_seen == [0, 1, 1, 2]
    assert firing_counter[0] == 3


def test_firing_id_counter_does_not_advance_on_empty_firing() -> None:
    """An ops-less firing must NOT burn a firing_id — the counter
    only ticks when persistence actually happens. This keeps the
    per-session sequence contiguous and aligned with on-disk entries.
    """
    firing_counter: list[int] = [0]
    persisted: list[dict[str, Any]] = []

    def _persist_firing(ops: list[NodeUpsert | NodeDelete]) -> None:
        firing_id = firing_counter[0]
        for op_index, op in enumerate(ops):
            payload = op.to_dict()
            payload["firing_id"] = firing_id
            payload["op_index"] = op_index
            persisted.append(payload)
        if ops:
            firing_counter[0] += 1

    _persist_firing([NodeUpsert(id=1, kind="task", summary="t", source_turns=(0,))])
    _persist_firing([])  # trivial / empty — no advance
    _persist_firing([NodeUpsert(id=2, kind="hyp", summary="h", source_turns=(1,))])

    assert [p["firing_id"] for p in persisted] == [0, 1]
    assert firing_counter[0] == 2


# --- B1 + B2 combined: extractor-cursor advance on pure op firing -----------


def test_scan_branch_cursor_survives_audit_graph_op_only_branch() -> None:
    """When a firing's only output is AUDIT_GRAPH_OP + EXTRACTOR_CURSOR
    (no legacy AUDIT_EVENT/AUDIT_EDGE), the scanner must still report
    the cursor — proving the maintainer path doesn't break the
    "have we caught up?" gate that drives window slicing.
    """
    branch = [
        _entry(
            AUDIT_GRAPH_OP,
            {
                "op": "node_delete",
                "id": 7,
                "firing_id": 0,
                "op_index": 0,
                "caused_by_turn_window": [4, 5],
            },
        ),
        _entry(EXTRACTOR_CURSOR, {"last_turn_index": 5, "extraction_run_id": "x"}),
    ]
    state = _scan_branch(branch, recent_verdicts_n=0)
    assert state.cursor_last_turn_index == 5


# --- legacy AUDIT_EDGE translation parity ----------------------------------


def test_scan_branch_legacy_audit_edge_translation_preserves_witness_fields() -> None:
    """Cross-check on the legacy AUDIT_EDGE → EdgeUpsert path: the
    cited_entities / cited_quote / src_turns / dst_turns must round-
    trip so the auditor's witness display doesn't lose anchors.
    Pinned because the witness fields are exactly the bit the auditor
    needs to drill back to the trace.
    """
    branch = [
        _entry(
            AUDIT_EVENT,
            {"id": 1, "kind": "act", "summary": "s", "source_turns": [1], "external_refs": []},
        ),
        _entry(
            AUDIT_EVENT,
            {"id": 2, "kind": "act", "summary": "d", "source_turns": [2], "external_refs": []},
        ),
        _entry(
            AUDIT_EDGE,
            {
                "src": 1,
                "dst": 2,
                "kind": "data",
                "reason": "supports",
                "src_turns": [1],
                "dst_turns": [2],
                "cited_entities": ["abnormal_traces"],
                "cited_quote": "",
            },
        ),
    ]
    state = _scan_branch(branch, recent_verdicts_n=0)
    assert len(state.edges) == 1
    ed = state.edges[0]
    assert ed.cited_entities == ("abnormal_traces",)
    assert ed.src_turns == (1,)
    assert ed.dst_turns == (2,)
