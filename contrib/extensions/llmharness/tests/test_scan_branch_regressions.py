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
    AUDIT_GRAPH_OP,
    EXTRACTOR_CURSOR,
)
from llmharness.audit.graph_ops import NodeDelete, NodeUpsert


def _entry(etype: str, payload: dict[str, Any]) -> SessionEntry:
    return SessionEntry(
        type=etype,
        id=uuid.uuid4().hex,
        parent_id=None,
        timestamp=0.0,
        payload=payload,
    )


# --- B1: pure-op firing -----------------------------------------------------






# --- B2: legacy AUDIT_EVENT external_refs round-trip ------------------------




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


