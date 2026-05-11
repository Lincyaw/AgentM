"""Per-firing in-memory ``ExtractionState`` for the v3 extractor.

V3.1 (events-only single-tool flow): one ``submit_events`` call carries
the entire graph as a list of events with embedded ``refs[]``. The
state IS the output: the adapter constructs one ``ExtractionState`` per
firing, hands it to ``build_extractor_tools`` so the tool callback
closes over it, and reads ``events`` / ``edges`` / ``dropped_edges``
back after the child loop terminates.

The validation pipeline runs inside :meth:`ExtractionState.commit`:

1. **events shape**: ``id`` must be 1..N strictly increasing, each
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

from ...schema import Edge, EdgeKind, Event, EventKind
from .._enum_schema import EDGE_KIND_VALUES, EVENT_KIND_VALUES
from .witness import witness_data, witness_ref


@dataclass
class ExtractionState:
    """Per-firing scratch space for the v3 extractor tool flow."""

    # turn_index -> raw turn text used for witness substring checks. The
    # adapter populates this from the trajectory window before spawning
    # the extractor child. Keys are absolute trajectory indices; values
    # are the rendered text content for that turn.
    turn_texts: dict[int, str] = field(default_factory=dict)

    # Frozen results filled in by ``commit``.
    events: tuple[Event, ...] = ()
    edges: tuple[Edge, ...] = ()
    dropped_edges: tuple[dict[str, Any], ...] = ()
    committed: bool = False

    # ------------------------------------------------------------------
    # Public mutator: single-shot commit

    def commit(self, events_payload: list[dict[str, Any]]) -> str | None:
        """Validate ``events_payload`` and populate the frozen results.

        Returns ``None`` on (full-or-partial) success; ``ExtractionState``
        is then populated with ``events`` / ``edges`` / ``dropped_edges``
        and ``committed=True``.

        Returns an error string on hard rejection (event-shape errors).
        The state is NOT mutated on hard reject — the LLM may retry by
        re-calling ``submit_events`` with a corrected payload, subject
        to the caller's attempt budget.
        """

        if self.committed:
            return "submit_events: already committed; one submission per firing"

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

        # Pass 2: cross-event id check — strictly 1..N, no gaps, no
        # repeats. The shape check above guarantees ``id`` is an int.
        for idx, ev in enumerate(working, start=1):
            if ev.id != idx:
                return (
                    "submit_events: event ids must be 1, 2, 3, ... in submission "
                    f"order; got events[{idx - 1}].id={ev.id} (expected {idx})"
                )

        events_by_id = {ev.id: ev for ev in working}

        # Pass 3: validate refs and accumulate edges + dropped.
        accepted_edges: list[Edge] = []
        dropped: list[dict[str, Any]] = []
        for raw_event, ev in zip(events_payload, working, strict=True):
            refs_raw = raw_event.get("refs", [])
            if refs_raw is None:
                refs_raw = []
            if not isinstance(refs_raw, list):
                return f"submit_events: events[{ev.id - 1}].refs must be an array"
            # Genesis exception: id=1 (first event of this firing) has no
            # in-window predecessor and may have empty refs. Every other
            # event MUST cite at least one earlier event — without refs
            # the auditor cannot trace causal structure across this
            # firing's window.
            if ev.id >= 2 and not refs_raw:
                return (
                    f"submit_events: events[{ev.id - 1}].refs must be "
                    "non-empty for non-genesis events (id>=2). The "
                    f"event id={ev.id} must cite at least one earlier "
                    "event in this firing with a witness."
                )
            for ridx, raw_ref in enumerate(refs_raw):
                if not isinstance(raw_ref, dict):
                    return (
                        f"submit_events: events[{ev.id - 1}].refs[{ridx}] must be "
                        "an object"
                    )
                err = _validate_ref_shape(ev.id, ridx, raw_ref, events_by_id)
                if err is not None:
                    return err
                # Witness: hard fields are present, validate substrings.
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

        self.events = tuple(working)
        self.edges = tuple(accepted_edges)
        self.dropped_edges = tuple(dropped)
        self.committed = True
        return None

    def _concat_turn_texts(self, turn_indices: list[int] | tuple[int, ...]) -> str:
        # Missing turn texts contribute the empty string — the witness
        # check will then naturally fail rather than KeyError out.
        return " ".join(self.turn_texts.get(idx, "") for idx in turn_indices)


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
            f"submit_events: events[{self_event_id - 1}].refs[{ridx}].to must be "
            "an integer"
        )
    if to_raw not in events_by_id:
        return (
            f"submit_events: events[{self_event_id - 1}].refs[{ridx}].to={to_raw} "
            "does not reference any submitted event id"
        )
    if to_raw >= self_event_id:
        return (
            f"submit_events: events[{self_event_id - 1}].refs[{ridx}].to={to_raw} "
            f"must reference an EARLIER event (< {self_event_id}); refs only flow "
            "forward in time"
        )
    if not isinstance(kind_raw, str):
        return (
            f"submit_events: events[{self_event_id - 1}].refs[{ridx}].kind must "
            "be a string"
        )
    try:
        kind = EdgeKind(kind_raw)
    except ValueError:
        return (
            f"submit_events: events[{self_event_id - 1}].refs[{ridx}].kind "
            f"{kind_raw!r} not in {EDGE_KIND_VALUES}"
        )

    cited_entities = raw.get("cited_entities", [])
    cited_quote = raw.get("cited_quote", "")
    if kind is EdgeKind.DATA:
        if not isinstance(cited_entities, list) or not cited_entities:
            return (
                f"submit_events: events[{self_event_id - 1}].refs[{ridx}] kind="
                "'data' requires non-empty cited_entities"
            )
        for e in cited_entities:
            if not isinstance(e, str) or not e:
                return (
                    f"submit_events: events[{self_event_id - 1}].refs[{ridx}]."
                    "cited_entities must be non-empty strings"
                )
    else:  # EdgeKind.REF
        if not isinstance(cited_quote, str) or not cited_quote:
            return (
                f"submit_events: events[{self_event_id - 1}].refs[{ridx}] kind="
                "'ref' requires non-empty cited_quote"
            )
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        return (
            f"submit_events: events[{self_event_id - 1}].refs[{ridx}].reason "
            "must be a string"
        )
    return None


__all__ = ["ExtractionState"]
