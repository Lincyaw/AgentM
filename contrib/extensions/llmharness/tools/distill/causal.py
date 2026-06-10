"""Causal masking of audit graph state to turn t.

This is the load-bearing piece of the labeling pipeline (design D2 in
README.md). Without it, the oracle uses post-hoc evidence (events whose
``source_turns`` extend beyond t) to decide what to flag at turn t,
producing a non-causal selection function that the student model — which
only sees past turns — cannot reproduce.

Rules (kept deliberately simple):

* **Event** kept iff ``max(source_turns) <= t``. (Open-cycle events with
  ``source_turns = []`` are kept regardless — they are not future
  information.)
* **Edge** kept iff both endpoint events are kept AND
  ``max(src_turns | dst_turns) <= t``.
* **Finding** kept iff every ``related_event_ids`` entry refers to a
  kept event. (A finding citing a future event is itself non-causal.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schema import Edge, Event, Finding


@dataclass(frozen=True)
class CausalSnapshot:
    """The view of the graph that existed at turn ``turn_index``."""

    turn_index: int
    events: tuple[Event, ...]
    edges: tuple[Edge, ...]
    findings: tuple[Finding, ...]
    trajectory_turns: tuple[dict[str, Any], ...]  # serialized, indices ≤ turn_index

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "events": [e.to_dict() for e in self.events],
            "edges": [ed.to_dict() for ed in self.edges],
            "findings": [
                {
                    "index": i,
                    "category": f.category,
                    "description": f.description,
                    "related_event_ids": list(f.related_event_ids),
                }
                for i, f in enumerate(self.findings)
            ],
            "trajectory": list(self.trajectory_turns),
        }


def _event_max_turn(ev: Event) -> int:
    return max(ev.source_turns) if ev.source_turns else -1


def _edge_max_turn(ed: Edge) -> int:
    pool = list(ed.src_turns) + list(ed.dst_turns)
    return max(pool) if pool else -1


def causal_mask(
    *,
    turn_index: int,
    events: list[Event] | tuple[Event, ...],
    edges: list[Edge] | tuple[Edge, ...],
    findings: list[Finding] | tuple[Finding, ...],
    trajectory: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> CausalSnapshot:
    kept_events = tuple(ev for ev in events if _event_max_turn(ev) <= turn_index)
    kept_event_ids = {ev.id for ev in kept_events}

    kept_edges = tuple(
        ed
        for ed in edges
        if ed.src in kept_event_ids
        and ed.dst in kept_event_ids
        and _edge_max_turn(ed) <= turn_index
    )

    kept_findings = tuple(
        f for f in findings if all(eid in kept_event_ids for eid in f.related_event_ids)
    )

    kept_trajectory = tuple(t for t in trajectory if int(t.get("index", -1)) <= turn_index)

    return CausalSnapshot(
        turn_index=turn_index,
        events=kept_events,
        edges=kept_edges,
        findings=kept_findings,
        trajectory_turns=kept_trajectory,
    )


__all__ = ["CausalSnapshot", "causal_mask"]
