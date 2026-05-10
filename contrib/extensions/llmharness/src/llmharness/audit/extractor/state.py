"""Per-firing in-memory ``ExtractionState`` for the v3 extractor.

This is mechanism (not an atom): the adapter constructs one
``ExtractionState`` per extractor firing, hands it to
``build_extractor_tools`` so the tool callbacks close over it, and
freezes it at the end to drive ``RawExtractorOutput``.

Validation order in ``add_edge`` mirrors design §4.f:
existence → src≠dst → turns subset → cycle → per-kind witness.
The retry budget (``max_retry = 2`` per ``(src, dst, kind)`` tuple,
i.e. 3 attempts including the initial one) is enforced INSIDE
``add_edge``: after the third failure on the same tuple the edge is
recorded into ``dropped_edges`` and subsequent attempts on the same
tuple short-circuit with the terminal sentinel
``"giving up on this edge"`` without re-validating.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ...schema import Edge, EdgeKind, Event, EventKind
from .witness import witness_data, witness_ref

_GIVING_UP_SENTINEL = "giving up on this edge"
_MAX_RETRY = 2  # plus the initial attempt = up to 3 attempts per tuple


@dataclass
class ExtractionState:
    """Per-firing scratch space for the v3 extractor tool flow."""

    # turn_index -> raw turn text used for witness substring checks. The
    # adapter populates this from the trajectory window before spawning
    # the extractor child. Keys are absolute trajectory indices; values
    # are the rendered text content for that turn.
    turn_texts: dict[int, str] = field(default_factory=dict)

    # event_id -> Event (insertion order, monotonic ids starting at 1).
    events: dict[int, Event] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    next_event_id: int = 1

    # Per-(src, dst, kind) attempt counter. After the third failure the
    # tuple is frozen and `add_edge` short-circuits on subsequent
    # attempts.
    failure_counts: dict[tuple[int, int, str], int] = field(default_factory=dict)
    # Tuples that have reached the terminal "giving up" state. Keeps
    # `add_edge` deterministic on repeat calls without re-validating.
    terminal_tuples: set[tuple[int, int, str]] = field(default_factory=set)
    dropped_edges: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public mutators

    def register_event(
        self,
        *,
        kind: EventKind,
        summary: str,
        source_turns: Iterable[int],
    ) -> int:
        """Register a new event. Returns its monotonic id."""

        event_id = self.next_event_id
        self.events[event_id] = Event(
            id=event_id,
            kind=kind,
            summary=summary,
            source_turns=list(source_turns),
        )
        self.next_event_id += 1
        return event_id

    def add_edge(
        self,
        *,
        src_event_id: int,
        dst_event_id: int,
        kind: EdgeKind,
        reason: str,
        src_turns: Iterable[int],
        dst_turns: Iterable[int],
        cited_entities: Iterable[str] = (),
        cited_quote: str = "",
    ) -> str | None:
        """Validate and append an edge. Returns None on success, error string on failure.

        The retry budget is enforced here: after the third failure on
        the same ``(src, dst, kind)`` tuple, the edge is recorded into
        ``dropped_edges`` and the sentinel ``"giving up on this edge"``
        error is returned. Subsequent attempts on the same tuple return
        the same terminal error WITHOUT re-validating.
        """

        src_turns_list = list(src_turns)
        dst_turns_list = list(dst_turns)
        cited_entities_list = list(cited_entities)
        tuple_key = (src_event_id, dst_event_id, kind.value)

        # Short-circuit terminal tuples — design §4.f retry policy.
        if tuple_key in self.terminal_tuples:
            return f"{_GIVING_UP_SENTINEL}: ({src_event_id}, {dst_event_id}, {kind.value}) frozen"

        err = self._validate_edge(
            src_event_id=src_event_id,
            dst_event_id=dst_event_id,
            kind=kind,
            src_turns=src_turns_list,
            dst_turns=dst_turns_list,
            cited_entities=cited_entities_list,
            cited_quote=cited_quote,
        )
        if err is None:
            self.edges.append(
                Edge(
                    src=src_event_id,
                    dst=dst_event_id,
                    kind=kind,
                    reason=reason,
                    src_turns=tuple(src_turns_list),
                    dst_turns=tuple(dst_turns_list),
                    cited_entities=tuple(cited_entities_list),
                    cited_quote=cited_quote,
                )
            )
            return None

        # Record failure and check retry budget.
        prev = self.failure_counts.get(tuple_key, 0)
        attempts_used = prev + 1
        self.failure_counts[tuple_key] = attempts_used
        if attempts_used > _MAX_RETRY:
            # Third (or later) failure -> terminal.
            self.terminal_tuples.add(tuple_key)
            self.dropped_edges.append(
                {
                    "src": src_event_id,
                    "dst": dst_event_id,
                    "kind": kind.value,
                    "last_error": err,
                }
            )
            return f"{_GIVING_UP_SENTINEL}: {err}"
        return err

    def freeze(
        self,
    ) -> tuple[tuple[Event, ...], tuple[Edge, ...], list[dict[str, Any]]]:
        """Snapshot the final state for the adapter."""

        events = tuple(self.events[i] for i in sorted(self.events))
        edges = tuple(self.edges)
        return events, edges, list(self.dropped_edges)

    # ------------------------------------------------------------------
    # Internals

    def _validate_edge(
        self,
        *,
        src_event_id: int,
        dst_event_id: int,
        kind: EdgeKind,
        src_turns: list[int],
        dst_turns: list[int],
        cited_entities: list[str],
        cited_quote: str,
    ) -> str | None:
        # 1. Existence + src != dst.
        if src_event_id not in self.events:
            return f"add_edge: src_event_id {src_event_id} not registered"
        if dst_event_id not in self.events:
            return f"add_edge: dst_event_id {dst_event_id} not registered"
        if src_event_id == dst_event_id:
            return "add_edge: src_event_id must differ from dst_event_id"

        # 2. Turns are subsets of the corresponding event source_turns.
        src_event = self.events[src_event_id]
        dst_event = self.events[dst_event_id]
        src_allowed = set(src_event.source_turns)
        dst_allowed = set(dst_event.source_turns)
        bad_src = [t for t in src_turns if t not in src_allowed]
        if bad_src:
            return (
                f"add_edge: src_turns {bad_src} not subset of "
                f"events[{src_event_id}].source_turns {sorted(src_allowed)}"
            )
        bad_dst = [t for t in dst_turns if t not in dst_allowed]
        if bad_dst:
            return (
                f"add_edge: dst_turns {bad_dst} not subset of "
                f"events[{dst_event_id}].source_turns {sorted(dst_allowed)}"
            )

        # 3. Cycle check.
        if self.cycle_check(src_event_id, dst_event_id):
            return f"add_edge: edge {src_event_id}->{dst_event_id} would introduce a cycle"

        # 4. Per-kind witness.
        src_text = self._concat_turn_texts(src_turns)
        dst_text = self._concat_turn_texts(dst_turns)
        if kind is EdgeKind.DATA:
            return witness_data(cited_entities, src_text, dst_text)
        # EdgeKind.REF
        return witness_ref(cited_quote, src_text, dst_text)

    def _concat_turn_texts(self, turn_indices: list[int]) -> str:
        # Missing turn texts contribute the empty string — the witness
        # check will then naturally fail rather than KeyError out.
        return " ".join(self.turn_texts.get(idx, "") for idx in turn_indices)

    def cycle_check(self, src: int, dst: int) -> bool:
        """Return True iff adding ``src->dst`` would create a cycle.

        Treats both ``data`` and ``ref`` as directed edges (design §4.f
        step 3). Implementation: DFS from ``dst`` in the existing edge
        set; if we can reach ``src`` then adding ``src->dst`` closes a
        cycle.
        """

        if src == dst:
            return True
        adj: dict[int, list[int]] = {}
        for e in self.edges:
            adj.setdefault(e.src, []).append(e.dst)
        stack: list[int] = [dst]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adj.get(node, ()))
        return False


__all__ = ["ExtractionState"]
