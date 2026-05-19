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

    # The LLM's basic-block partition of the new-turn window, submitted
    # alongside ``events`` in the same ``submit_events`` payload. JSON
    # field ordering in the schema places this BEFORE ``events`` so the
    # model commits to a partition before token-generating events,
    # turning the basic-block discipline (Principle 3) from a soft
    # instruction into a generation-order constraint. Stored verbatim
    # for offline inspection; v15 keeps validation soft (no rejection
    # on plan/event mismatch).
    block_plan: tuple[dict[str, Any], ...] = ()

    # Plan-committed flag — set by ``commit_plan``. Allows the v18+ two-
    # tool flow (``submit_plan`` then ``submit_events_batch``); the
    # legacy single-shot ``commit`` bypasses this and skips the flag.
    plan_committed: bool = False

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
    # Public mutators

    def commit_plan(self, plan_payload: list[dict[str, Any]]) -> str | None:
        """Validate + capture the block plan; called once per firing.

        The plan is informational — the structural invariant (every
        internal event must be a true branch point) is enforced on the
        emitted graph in :meth:`finalize`, not on the plan. Returns
        ``None`` on success; an error string on a malformed payload
        (state unchanged).
        """
        if self.plan_committed:
            return "submit_plan: already committed; one plan per firing"
        if not isinstance(plan_payload, list):
            return "submit_plan: 'block_plan' must be an array"
        sanitized: list[dict[str, Any]] = []
        for idx, entry in enumerate(plan_payload):
            if not isinstance(entry, dict):
                return f"submit_plan: block_plan[{idx}] must be an object"
            kind = entry.get("kind")
            if kind not in ("linear", "branch"):
                return (
                    f"submit_plan: block_plan[{idx}].kind must be "
                    f"'linear' or 'branch'; got {kind!r}"
                )
            turns = entry.get("turns")
            if not isinstance(turns, list) or not turns:
                return (
                    f"submit_plan: block_plan[{idx}].turns must be a "
                    "non-empty array of integers"
                )
            for t in turns:
                if isinstance(t, bool) or not isinstance(t, int):
                    return (
                        f"submit_plan: block_plan[{idx}].turns contains "
                        f"non-integer entry {t!r}"
                    )
            note = entry.get("note")
            if not isinstance(note, str) or not note.strip():
                return (
                    f"submit_plan: block_plan[{idx}].note must be a "
                    "non-empty string"
                )
            sanitized.append(entry)
        self.block_plan = tuple(sanitized)
        self.plan_committed = True
        return None

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
        """Run the cross-graph degree check and freeze pending state.

        Returns ``None`` on success; ``ExtractionState.committed`` is
        then ``True`` and ``events`` / ``edges`` / ``dropped_edges`` are
        populated. Returns an error string when the accumulated graph
        has any passthrough event (in-degree=1 AND out-degree=1) — see
        :func:`_validate_event_degrees`. On error the pending state
        stays intact so the LLM can submit additional batches that
        promote the offending events to true branch points (e.g. add a
        later event with an extra ref back to the passthrough node).
        """
        if self.committed:
            return "finalize: firing already finalized"
        if not self._events_pending:
            self.events = ()
            self.edges = ()
            self.dropped_edges = ()
            self.committed = True
            return None
        degree_err = _validate_event_degrees(
            self._events_pending, self._edges_pending
        )
        if degree_err is not None:
            return degree_err
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


def _validate_event_degrees(
    events: list[Event],
    edges: list[Edge],
) -> str | None:
    """Reject any event whose (in_deg, out_deg) is (1, 1) — passthroughs.

    A passthrough event sits on a linear chain with no branching role:
    it adds no graph-level structure beyond its single predecessor and
    successor. Such events are the v17 basic-block coalescence failure
    mode in disguise — instead of merging the linear stretch into one
    act+evid pair, the LLM emitted one event per tool_call/tool_result
    pair and threaded a chain through them.

    The valid shapes are:

    * (0, 0): a single-node firing — rare; only legal when the firing
      has exactly one event with no parents and no children.
    * (0, k>=1): a starting node — typically the task event or the
      genesis event of the very first firing.
    * (k>=1, 0): a terminal node — typically a concl event or the last
      observation in this firing's window.
    * (in>=1, out>=2) / (in>=2, out>=1): a true branch / merge point.

    Returns ``None`` on a clean graph; an error string on violation,
    listing every offending event so the LLM can fix them all in one
    retry (rather than bouncing on the first).

    Note: only in-firing edges count toward degree. External refs are
    intentionally excluded — they connect to prior firings, which the
    aggregator stitches later, and counting them would let an event
    "look like" a branch point in this firing while still being a
    passthrough in the cumulative graph.
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
    passthrough: list[Event] = [
        ev for ev in events if in_deg[ev.id] == 1 and out_deg[ev.id] == 1
    ]
    if not passthrough:
        return None
    lines = [
        f"  event[{ev.id}] kind={ev.kind.value} "
        f"'{ev.summary[:70]}': in=1, out=1"
        for ev in passthrough
    ]
    return (
        "finalize: graph has passthrough events (in-degree=1 AND "
        "out-degree=1). These events sit on a linear chain with no "
        "branching role — they should be merged with their neighbour "
        "into a single basic block (one act + one evid per linear "
        "stretch), or promoted to true branch points by adding another "
        "ref. A new event submitted in a later batch CAN boost the "
        "out-degree of an existing event by ref-ing it again; that's "
        "the recovery path when you want to keep the event but make "
        "it a branch.\n"
        f"Passthrough events ({len(passthrough)}):\n"
        + "\n".join(lines)
    )


__all__ = ["ExtractionState"]
