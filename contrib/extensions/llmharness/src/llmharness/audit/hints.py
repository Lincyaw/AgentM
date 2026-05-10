"""Advisory hints module — pure deterministic signals over the event graph.

All functions are pure Python, synchronous, and operate on the in-memory
``list[Event]`` the adapter already has at auditor-firing time.

The ``compute(graph)`` function is the single entry point the adapter calls:
it renders all signals into a compact textual block ready for the auditor
JSON payload's ``hints`` key.

Phrasing follows design §7.5 exactly: each signal is a "consider …" directive,
never a "concern: …" statement, to avoid anchoring the auditor to a binary
triage stance. The auditor may ignore any or all hints, or flag concerns
they missed.

Design reference: ``.claude/designs/llmharness-cognitive-audit.md`` §7.5.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..schema import Event, EventKind

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _all_referenced_ids(graph: list[Event]) -> set[int]:
    """Return the set of all event ids that appear in any event's refs."""
    referenced: set[int] = set()
    for ev in graph:
        referenced.update(ev.refs)
    return referenced


def _transitive_refs(
    event_id: int,
    id_to_event: dict[int, Event],
) -> set[int]:
    """BFS/DFS over refs to collect the transitive closure of events reachable
    from ``event_id`` via the ref graph (i.e., everything the event cites,
    and everything those events cite, etc.).

    This is the ``event_id``'s *ancestry* in the ref DAG — the set of events
    it transitively depends on.
    """
    visited: set[int] = set()
    stack = [event_id]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        ev = id_to_event.get(current)
        if ev is None:
            continue
        stack.extend(ev.refs)
    # Exclude the start node itself.
    visited.discard(event_id)
    return visited


def _uf_find(parent: list[int], x: int) -> int:
    """Union-Find path-compressed find."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _count_independent_branches(parent: list[int], ref_ancestors: list[set[int]], n: int) -> int:
    """Union-Find grouping: merge refs whose ancestor sets overlap.

    Returns the number of distinct independent groups (branches).
    """
    for i in range(n):
        for j in range(i + 1, n):
            if ref_ancestors[i] & ref_ancestors[j]:
                ri, rj = _uf_find(parent, i), _uf_find(parent, j)
                if ri != rj:
                    parent[ri] = rj
    return len({_uf_find(parent, i) for i in range(n)})


def _canonical_summary(summary: str) -> str:
    """Canonical form for comparing action summaries.

    Transformation: lowercase → collapse whitespace runs → strip trailing
    punctuation. Two summaries are considered the "same action" iff their
    canonical forms are equal.
    """
    s = summary.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".!?,;:")
    return s


# ---------------------------------------------------------------------------
# signal functions
# ---------------------------------------------------------------------------


def repeated_actions(graph: list[Event]) -> list[tuple[int, ...]]:
    """Detect action events with identical canonical summaries.

    Canonical form: lowercase + collapse whitespace + strip trailing
    punctuation. Two action events are the "same action" iff their
    canonical summaries are equal.

    Returns a list of tuples; each tuple contains the event ids that share
    a canonical summary (groups of size ≥ 2 only).  Signals a potential
    stuck-loop pattern.
    """
    by_canonical: dict[str, list[int]] = defaultdict(list)
    for ev in graph:
        if ev.kind is EventKind.ACTION:
            by_canonical[_canonical_summary(ev.summary)].append(ev.id)

    result: list[tuple[int, ...]] = []
    for ids in by_canonical.values():
        if len(ids) >= 2:
            result.append(tuple(sorted(ids)))
    return result


def convergence_ratio(graph: list[Event]) -> float:
    """Fraction of unresolved hypotheses and open decisions vs total events.

    Formula: (unresolved_hypotheses + open_decisions) / max(1, total_events)

    - "Unresolved hypothesis": a ``hypothesis`` event whose id never appears
      in any other event's ``refs``.
    - "Open decision": a ``decision`` event whose id never appears in any
      other event's ``refs``.

    Returns a float in [0, 1]. A value near 1 means most hypotheses and
    decisions were never followed up; near 0 means all were addressed.
    """
    if not graph:
        return 0.0

    referenced = _all_referenced_ids(graph)
    unresolved = sum(
        1
        for ev in graph
        if ev.kind in (EventKind.HYPOTHESIS, EventKind.DECISION) and ev.id not in referenced
    )
    return unresolved / max(1, len(graph))


def reachability_gaps(graph: list[Event]) -> list[int]:
    """Evidence events not reached by any conclusion event via transitive refs.

    Definition: an ``evidence`` event is "in a gap" when no ``conclusion``
    event's transitive ref ancestry includes it.

    If no ``conclusion`` events exist in the graph, returns an empty list —
    the graph is still in progress, which is not a gap.
    """
    conclusions = [ev for ev in graph if ev.kind is EventKind.CONCLUSION]
    if not conclusions:
        return []

    id_to_event = {ev.id: ev for ev in graph}

    # Collect all event ids reachable from any conclusion via transitive refs.
    reachable: set[int] = set()
    for conclusion in conclusions:
        reachable.add(conclusion.id)
        reachable.update(_transitive_refs(conclusion.id, id_to_event))

    gaps: list[int] = []
    for ev in graph:
        if ev.kind is EventKind.EVIDENCE and ev.id not in reachable:
            gaps.append(ev.id)
    return sorted(gaps)


def open_branches(graph: list[Event]) -> list[int]:
    """Decision events whose id is never referenced by any later event.

    A ``decision`` event is "open" when no other event in the graph has
    this id in its ``refs``. This signals that the branch was created but
    never followed up by evidence or a conclusion — a discarded or forgotten
    branch.

    Returns sorted list of open decision event ids.
    """
    referenced = _all_referenced_ids(graph)
    open_ids: list[int] = []
    for ev in graph:
        if ev.kind is EventKind.DECISION and ev.id not in referenced:
            open_ids.append(ev.id)
    return sorted(open_ids)


def multi_branch_syntheses(graph: list[Event]) -> list[tuple[int, int]]:
    """Conclusion events whose refs reach multiple independent root paths.

    For each conclusion event, we count the number of its direct refs whose
    transitive ancestor sets are disjoint (share no intermediate events,
    excluding task events themselves).  When two ref's ancestries overlap in
    a non-task event, they share a chain and are not independent.

    Returns list of (conclusion_id, branch_count) for conclusions with
    branch_count ≥ 2.

    Operational definition: for each pair of direct refs of the conclusion,
    compute their transitive ancestor sets (excluding task events). Two refs
    are "independent" if their ancestor sets share no non-task event AND neither
    ref appears in the other's ancestry.  Count independent groups using a
    greedy partition.
    """
    task_ids = {ev.id for ev in graph if ev.kind is EventKind.TASK}
    id_to_event = {ev.id: ev for ev in graph}

    results: list[tuple[int, int]] = []

    for ev in graph:
        if ev.kind is not EventKind.CONCLUSION:
            continue
        if len(ev.refs) < 2:
            continue

        # Compute ancestor sets for each direct ref, excluding task nodes.
        ref_ancestors: list[set[int]] = []
        for ref_id in ev.refs:
            ancestors = _transitive_refs(ref_id, id_to_event) - task_ids
            ancestors.add(ref_id)  # include the ref itself
            ref_ancestors.append(ancestors)

        # Group into independent branches: two refs are in the same group
        # if their ancestor sets overlap.  Use an explicit union-find
        # implemented without a nested closure so ruff B023 is satisfied.
        n = len(ref_ancestors)
        union_parent = list(range(n))

        branch_count = _count_independent_branches(union_parent, ref_ancestors, n)
        if branch_count >= 2:
            results.append((ev.id, branch_count))

    return results


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------


def compute(graph: list[Event]) -> str:
    """Render all advisory signals into a compact prompt-ready block.

    Returns the empty string when no anomaly is found. Otherwise returns
    a block starting with a one-line header followed by one bullet per
    anomaly category that fired.  Always ends with a trailing newline when
    non-empty, so the JSON-payload concatenation doesn't run into other text.

    Output is deterministic: event ids are sorted, categories appear in
    stable order.
    """
    bullets: list[str] = []

    # 1. Repeated actions.
    repeated = repeated_actions(graph)
    for group in repeated:
        ids_str = ", ".join(str(i) for i in sorted(group))
        bullets.append(f"- consider: repeated action detected (event ids: {ids_str})")

    # 2. Convergence ratio — only surface when meaningfully low.
    ratio = convergence_ratio(graph)
    # Threshold: flag when ≥ 30 % of events are unresolved hyp/decisions
    # and the graph has at least 3 events (avoids noise on tiny graphs).
    if ratio >= 0.30 and len(graph) >= 3:
        pct = int(ratio * 100)
        bullets.append(
            f"- consider: low convergence — {pct}% of events are unresolved"
            " hypotheses or open decisions"
        )

    # 3. Reachability gaps.
    gaps = reachability_gaps(graph)
    if gaps:
        ids_str = ", ".join(str(i) for i in gaps)
        bullets.append(
            f"- consider: reachability gap — evidence event(s) not reached"
            f" by any conclusion (ids: {ids_str})"
        )

    # 4. Open branches.
    open_ids = open_branches(graph)
    if open_ids:
        ids_str = ", ".join(str(i) for i in open_ids)
        bullets.append(
            f"- consider: open branch — decision event(s) with no closing evidence (ids: {ids_str})"
        )

    # 5. Multi-branch syntheses.
    syntheses = multi_branch_syntheses(graph)
    for conc_id, branch_count in sorted(syntheses):
        bullets.append(
            f"- consider: multi-branch synthesis — conclusion event {conc_id}"
            f" merges {branch_count} independent branches; verify each branch"
            " carries sufficient evidence"
        )

    if not bullets:
        return ""

    lines = ["Advisory hints (consider — not directives):", *bullets, ""]
    return "\n".join(lines) + "\n"


__all__ = [
    "compute",
    "convergence_ratio",
    "multi_branch_syntheses",
    "open_branches",
    "reachability_gaps",
    "repeated_actions",
]
