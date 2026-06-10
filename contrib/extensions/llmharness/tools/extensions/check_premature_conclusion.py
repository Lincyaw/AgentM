"""Check: flag concl events with thin incoming evidence.

Heuristic — a ``concl`` event whose total incoming edge count
(``data`` + ``ref`` combined, from any earlier event) is strictly
less than 2 is flagged as a premature conclusion. The threshold
matches "thin evidence": a single supporting edge into a conclusion
is the canonical premature-conclusion shape.

Output: one :class:`~llmharness.schema.Finding` per such concl event.
Category: ``"premature_conclusion"``.
"""

from __future__ import annotations

from llmharness.schema import Edge, Event, EventKind, Finding

_PREMATURE_THRESHOLD = 2


def check_premature_conclusion(events: list[Event], edges: list[Edge]) -> list[Finding]:
    """Flag concl events with incoming-edge count below threshold."""
    incoming: dict[int, int] = {}
    for edge in edges:
        incoming[edge.dst] = incoming.get(edge.dst, 0) + 1

    findings: list[Finding] = []
    for ev in events:
        if ev.kind is not EventKind.CONCL:
            continue
        count = incoming.get(ev.id, 0)
        if count >= _PREMATURE_THRESHOLD:
            continue
        findings.append(
            Finding(
                category="premature_conclusion",
                description=(
                    f"concl event #{ev.id} {ev.summary!r}: "
                    f"{count} incoming edge(s), below threshold "
                    f"{_PREMATURE_THRESHOLD}"
                ),
                related_event_ids=(ev.id,),
            )
        )
    return findings


__all__ = ["check_premature_conclusion"]
