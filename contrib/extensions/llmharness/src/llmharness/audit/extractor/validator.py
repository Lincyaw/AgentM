"""Phase 1 graph validator — pure Python checks (design §5.4).

Runs in the adapter after each ``submit_events`` call, before any events are
committed to the session entry tree. All checks operate on plain Python data;
no I/O, no LLM calls.

Public surface::

    validate(
        *,
        new_events: list[Event],
        existing_events: list[Event],
        turn_index_to_kinds: dict[int, set[str]],
    ) -> list[str]

Returns an empty list when the graph is valid. Returns a non-empty list of
human-readable violation strings on any failure — one string per violation;
the adapter writes all violations to the ``llmharness.extractor_invalid``
entry so they are visible for debugging.

Five checks (accumulated — all violations reported, not just the first):

1. Ref resolution — every ``event.refs`` id resolves to a prior event
   (existing or a strictly earlier event in the same new batch).
2. No cycles — the directed ref graph has no cycles.
3. kind↔source_turns rules:
   - ``evidence``: at least one source turn must carry a ``tool_result``,
     be a ``user`` message, or carry a ``thinking`` block.
   - ``action``: at least one source turn must carry a ``tool_call``.
4. Task reachability — every non-task event traces back to a ``task``
   event via the ref chain.
5. Conclusion reachability — every ``conclusion`` event traces back to
   a ``task`` event via the ref chain.

Checks 4 and 5 both use the same ref-chain walk; check 5 is a targeted
specialisation of check 4 for conclusions.
"""

from __future__ import annotations

from collections import deque

from ...schema import Event, EventKind


def validate(
    *,
    new_events: list[Event],
    existing_events: list[Event],
    turn_index_to_kinds: dict[int, set[str]],
) -> list[str]:
    """Run the five Phase 1 graph checks (design §5.4).

    Parameters
    ----------
    new_events:
        Events submitted in the current extractor firing, in the order
        emitted (monotonic id, earlier events have smaller list indices).
    existing_events:
        Events already committed to the session entry tree from prior
        firings.  These may be referenced by ``new_events`` but are
        themselves not re-validated.
    turn_index_to_kinds:
        Maps absolute trajectory-message index → set of content kinds
        present in that turn.  Possible kinds: ``"tool_call"``,
        ``"tool_result"``, ``"thinking"``, ``"text"``, plus the role
        (``"user"`` / ``"assistant"``).  Built by the adapter from the
        serialized message slice the extractor received.

    Returns
    -------
    list[str]
        Empty if the graph is valid.  One human-readable string per
        violation otherwise.
    """
    # Build a unified id → Event map.  For new_events, we also need their
    # positional index within the batch to enforce "no forward refs within
    # the same batch".
    existing_by_id: dict[int, Event] = {e.id: e for e in existing_events}
    new_by_position: dict[int, int] = {}  # event id → batch position (0-based)
    new_by_id: dict[int, Event] = {}
    for pos, ev in enumerate(new_events):
        new_by_position[ev.id] = pos
        new_by_id[ev.id] = ev

    all_events_by_id: dict[int, Event] = {**existing_by_id, **new_by_id}

    violations: list[str] = []

    # ------------------------------------------------------------------
    # Check 1 — Ref resolution (including no forward refs in same batch)
    # ------------------------------------------------------------------
    violations.extend(
        _check_ref_resolution(
            new_events=new_events,
            new_by_position=new_by_position,
            all_events_by_id=all_events_by_id,
        )
    )

    # ------------------------------------------------------------------
    # Check 2 — No cycles
    # ------------------------------------------------------------------
    violations.extend(_check_no_cycles(all_events_by_id=all_events_by_id))

    # ------------------------------------------------------------------
    # Check 3 — kind↔source_turns rules
    # ------------------------------------------------------------------
    violations.extend(
        _check_kind_source_rules(
            new_events=new_events,
            turn_index_to_kinds=turn_index_to_kinds,
        )
    )

    # ------------------------------------------------------------------
    # Check 4 — Task reachability for all non-task events
    # Check 5 — Conclusion reachability (subset of check 4)
    # ------------------------------------------------------------------
    violations.extend(
        _check_task_reachability(
            new_events=new_events,
            all_events_by_id=all_events_by_id,
        )
    )

    return violations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_ref_resolution(
    *,
    new_events: list[Event],
    new_by_position: dict[int, int],
    all_events_by_id: dict[int, Event],
) -> list[str]:
    """Check 1: every ref resolves; no forward refs within the same batch."""
    violations: list[str] = []
    for ev in new_events:
        my_pos = new_by_position[ev.id]
        for ref_id in ev.refs:
            if ref_id not in all_events_by_id:
                violations.append(
                    f"ref-resolution: event {ev.id} refs id {ref_id} which does not exist "
                    f"in existing_events or prior new_events"
                )
            elif ref_id in new_by_position and new_by_position[ref_id] >= my_pos:
                # The ref points at a later (or same) event in the same batch.
                violations.append(
                    f"forward-ref: event {ev.id} at batch-position {my_pos} refs "
                    f"event {ref_id} at batch-position {new_by_position[ref_id]} "
                    f"(later in the same batch — monotonic order violated)"
                )
    return violations


def _check_no_cycles(*, all_events_by_id: dict[int, Event]) -> list[str]:
    """Check 2: directed ref graph has no cycles.

    Uses iterative DFS with a 'gray/black' colouring.  Accumulates all
    back-edges found (naming the two endpoints that form the back edge).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[int, int] = {eid: WHITE for eid in all_events_by_id}
    cycles: list[str] = []

    def _dfs(start: int) -> None:
        # Each entry: (node_id, mutable children list to visit next).
        iter_stack: list[tuple[int, list[int]]] = [(start, list(all_events_by_id[start].refs))]
        color[start] = GRAY

        while iter_stack:
            node, children = iter_stack[-1]
            if not children:
                color[node] = BLACK
                iter_stack.pop()
                continue
            child = children.pop(0)
            if child not in all_events_by_id:
                # Unresolved ref — already caught by check 1.
                continue
            if color[child] == GRAY:
                # Back edge → cycle detected.
                cycles.append(f"cycle: event {node} and event {child} are part of a ref cycle")
                # Do not recurse into this child; continue visiting siblings.
                continue
            if color[child] == WHITE:
                color[child] = GRAY
                iter_stack.append((child, list(all_events_by_id[child].refs)))

    for eid in all_events_by_id:
        if color[eid] == WHITE:
            _dfs(eid)

    return cycles


def _check_kind_source_rules(
    *,
    new_events: list[Event],
    turn_index_to_kinds: dict[int, set[str]],
) -> list[str]:
    """Check 3: kind↔source_turns rules for evidence and action events.

    For evidence: at least one source turn must carry "tool_result", be
    a "user" turn, or carry "thinking" content.

    For action: at least one source turn must carry "tool_call".
    """
    violations: list[str] = []
    for ev in new_events:
        if ev.kind == EventKind.EVIDENCE and not _any_turn_has_kind(
            ev.source_turns, turn_index_to_kinds, {"tool_result", "user", "thinking"}
        ):
            violations.append(
                f"evidence-source: event {ev.id} (evidence) has no source_turn "
                f"with tool_result, user, or thinking content "
                f"(source_turns={ev.source_turns})"
            )
        elif ev.kind == EventKind.ACTION and not _any_turn_has_kind(
            ev.source_turns, turn_index_to_kinds, {"tool_call"}
        ):
            violations.append(
                f"action-source: event {ev.id} (action) has no source_turn "
                f"with a tool_call content block "
                f"(source_turns={ev.source_turns})"
            )
    return violations


def _any_turn_has_kind(
    source_turns: list[int],
    turn_index_to_kinds: dict[int, set[str]],
    required_kinds: set[str],
) -> bool:
    """True iff at least one turn in source_turns has any of required_kinds."""
    for idx in source_turns:
        kinds = turn_index_to_kinds.get(idx, set())
        if kinds & required_kinds:
            return True
    return False


def _check_task_reachability(
    *,
    new_events: list[Event],
    all_events_by_id: dict[int, Event],
) -> list[str]:
    """Checks 4 & 5: task reachability for all new non-task events.

    Check 4: every non-task event must have a ref path back to a task event.
    Check 5: every conclusion event must have a ref path back to a task event
             (specialisation of check 4).

    Both checks use the same BFS from each non-task event; we combine them
    to avoid traversing the graph twice.
    """
    violations: list[str] = []
    task_ids: set[int] = {eid for eid, ev in all_events_by_id.items() if ev.kind == EventKind.TASK}

    for ev in new_events:
        if ev.kind == EventKind.TASK:
            continue  # task events don't need to reach a task

        reachable = _reachable_from(ev.id, all_events_by_id)
        if not (reachable & task_ids):
            if ev.kind == EventKind.CONCLUSION:
                violations.append(
                    f"conclusion-reachability: conclusion event {ev.id} has no ref "
                    f"path back to any task event"
                )
            else:
                violations.append(
                    f"task-reachability: event {ev.id} ({ev.kind.value}) has no ref "
                    f"path back to any task event"
                )

    return violations


def _reachable_from(start_id: int, all_events_by_id: dict[int, Event]) -> set[int]:
    """BFS from start_id following refs; returns all reachable event ids
    (excluding start_id itself).
    """
    visited: set[int] = set()
    queue: deque[int] = deque()

    start_event = all_events_by_id.get(start_id)
    if start_event is None:
        return visited

    for ref_id in start_event.refs:
        if ref_id not in visited:
            visited.add(ref_id)
            queue.append(ref_id)

    while queue:
        current_id = queue.popleft()
        current = all_events_by_id.get(current_id)
        if current is None:
            continue
        for ref_id in current.refs:
            if ref_id not in visited:
                visited.add(ref_id)
                queue.append(ref_id)

    return visited


__all__ = ["validate"]
