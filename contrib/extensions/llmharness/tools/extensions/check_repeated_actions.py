"""Check: flag repeated tool-call signatures.

Heuristic — two distinct ``act`` events sharing an identical
``summary`` count as a repeat. Equality on the summary string is the
most defensible signature available without an inline arg-hash field
on :class:`~llmharness.schema.Event`.

Output: one :class:`~llmharness.schema.Finding` per group of
>=2 events sharing an act-summary, with the full event-id tuple in
``related_event_ids``. Category: ``"repeated_actions"``.
"""

from __future__ import annotations

from llmharness.schema import Edge, Event, EventKind, Finding


def check_repeated_actions(events: list[Event], edges: list[Edge]) -> list[Finding]:
    """Flag groups of act events that share an identical summary."""
    del edges  # unused but kept in signature for interface consistency
    # Group act events by their summary, preserving first-seen order
    # so output is deterministic.
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for ev in events:
        if ev.kind is not EventKind.ACT:
            continue
        key = ev.summary
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ev.id)

    findings: list[Finding] = []
    for summary in order:
        ids = groups[summary]
        if len(ids) < 2:
            continue
        findings.append(
            Finding(
                category="repeated_actions",
                description=(
                    f"act-summary {summary!r} appears {len(ids)} times (events: {list(ids)})"
                ),
                related_event_ids=tuple(ids),
            )
        )
    return findings


__all__ = ["check_repeated_actions"]
