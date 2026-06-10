"""Check: flag dec/hyp events with no outgoing data edge.

A ``dec`` or ``hyp`` event without any outgoing
:class:`~llmharness.schema.EdgeKind.DATA` edge is the graph approximation
of "discarded alternative without closing evidence" — the agent registered
a choice or claim but produced no evidence-flow follow-up linking back to it.

Output: one :class:`~llmharness.schema.Finding` per such open event.
Category: ``"open_branches"``.
"""

from __future__ import annotations

from llmharness.schema import Edge, EdgeKind, Event, EventKind, Finding

_OPEN_KINDS = frozenset({EventKind.DEC, EventKind.HYP})


def check_open_branches(events: list[Event], edges: list[Edge]) -> list[Finding]:
    """Flag dec/hyp events with no outgoing data edge."""
    outgoing_data: set[int] = {edge.src for edge in edges if edge.kind is EdgeKind.DATA}
    findings: list[Finding] = []
    for ev in events:
        if ev.kind not in _OPEN_KINDS:
            continue
        if ev.id in outgoing_data:
            continue
        findings.append(
            Finding(
                category="open_branches",
                description=(
                    f"{ev.kind.value} event #{ev.id} {ev.summary!r}: "
                    "no downstream data edge found"
                ),
                related_event_ids=(ev.id,),
            )
        )
    return findings


__all__ = ["check_open_branches"]
