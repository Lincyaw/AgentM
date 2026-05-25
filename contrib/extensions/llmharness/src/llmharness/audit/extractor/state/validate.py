"""Module-level shape + degree validators extracted from extractor state.

These were defined as private helpers next to :class:`ExtractionState`
in the pre-reorg ``extractor/state.py``. They are still private to the
extractor — no behavior change, only physical relocation so the
1200-line god-file shrinks. The class's commit / apply methods import
them from here.
"""

from __future__ import annotations

from typing import Any

from ....schema import Edge, EdgeKind, Event, EventKind
from ...validation.enum_schema import EDGE_KIND_VALUES, EVENT_KIND_VALUES


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


__all__ = [
    "_coerce_int",
    "_compute_degree_warning",
    "_validate_event_shape",
    "_validate_external_ref_shape",
    "_validate_ref_shape",
]
