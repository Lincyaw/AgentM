"""Module-level shape + degree validators for the extractor state.

Used by the event-sourcing ``apply_*`` surface in
:class:`llmharness.audit.extractor.state.ExtractionState` and by
:meth:`ExtractionState.compute_degree_warning`.
"""

from __future__ import annotations

from typing import Any

from llmharness.schema import Edge, Event, EventKind
from llmharness.validation.enum_schema import EVENT_KIND_VALUES


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _validate_event_shape(idx: int, raw: dict[str, Any]) -> tuple[str | None, Event | None]:
    """Validate one node payload from ``upsert_node`` / ``apply_node_upsert``.

    ``idx`` is the per-firing node-index used only for human-readable
    error messages; the v4 tools call this with ``0`` because they
    submit one node at a time.
    """
    eid_raw = raw.get("id")
    kind_raw = raw.get("kind")
    summary_raw = raw.get("summary")
    source_turns_raw = raw.get("source_turns")

    if isinstance(eid_raw, bool) or not isinstance(eid_raw, int):
        return f"upsert_node: events[{idx}].id must be an integer", None
    if eid_raw < 1:
        return f"upsert_node: events[{idx}].id must be >= 1; got {eid_raw}", None
    if not isinstance(kind_raw, str):
        return f"upsert_node: events[{idx}].kind must be a string", None
    try:
        kind = EventKind(kind_raw)
    except ValueError:
        return (
            f"upsert_node: events[{idx}].kind {kind_raw!r} not in {EVENT_KIND_VALUES}",
            None,
        )
    if not isinstance(summary_raw, str) or not summary_raw.strip():
        return f"upsert_node: events[{idx}].summary must be a non-empty string", None
    if not isinstance(source_turns_raw, list) or not source_turns_raw:
        return (
            f"upsert_node: events[{idx}].source_turns must be a non-empty array of integers",
            None,
        )
    source_turns: list[int] = []
    for t in source_turns_raw:
        if isinstance(t, bool) or not isinstance(t, int):
            return (
                f"upsert_node: events[{idx}].source_turns contains non-integer entry {t!r}",
                None,
            )
        source_turns.append(t)
    return None, Event(id=eid_raw, kind=kind, summary=summary_raw, source_turns=source_turns)


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
    chain_links: list[Event] = [ev for ev in events if in_deg[ev.id] == 1 and out_deg[ev.id] == 1]
    if not chain_links:
        return None
    lines = [
        f"  event[{ev.id}] kind={ev.kind.value} '{ev.summary[:70]}': in=1, out=1"
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
        "Chain-link events:\n" + "\n".join(lines)
    )


__all__ = [
    "_coerce_int",
    "_compute_degree_warning",
    "_validate_event_shape",
]
