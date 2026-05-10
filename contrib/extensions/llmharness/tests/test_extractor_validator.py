"""Fail-stop unit tests for the Phase 1 graph validator (design §5.4).

Each test protects a specific fail-stop position. Fixtures are built from
plain ``list[Event]`` — no LLM, no spawned sessions.

The validator signature under test:

    validate(
        *,
        new_events: list[Event],
        existing_events: list[Event],
        turn_index_to_kinds: dict[int, set[str]],
    ) -> list[str]

Returns an empty list on a valid graph; otherwise a list of violation
strings, each naming the offending event id(s) and the rule violated.
"""

from __future__ import annotations

from llmharness.audit.extractor import validate_graph
from llmharness.schema import Event, EventKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    id: int,
    kind: EventKind,
    refs: list[int] | None = None,
    source_turns: list[int] | None = None,
) -> Event:
    return Event(
        id=id,
        kind=kind,
        summary=f"synthetic {kind.value} event {id}",
        refs=refs or [],
        source_turns=source_turns or [],
    )


def _kinds(*kind_strs: str) -> set[str]:
    """Convenience: build a content-kind set for turn_index_to_kinds."""
    return set(kind_strs)


# ---------------------------------------------------------------------------
# Test 1: valid graph — 5-event chain passes all checks
# ---------------------------------------------------------------------------


def test_valid_graph_returns_no_violations() -> None:
    """A well-formed 5-event graph produces zero violations.

    Graph: task(0) → hypothesis(1, refs=[0]) →
           action(2, refs=[1], source_turns=[5]) →
           evidence(3, refs=[2], source_turns=[6]) →
           conclusion(4, refs=[3])

    Fail-stop: valid graph must not be incorrectly rejected — a false
    positive here would block all audit events from being committed.
    """
    task = _make_event(id=0, kind=EventKind.TASK)
    hyp = _make_event(id=1, kind=EventKind.HYPOTHESIS, refs=[0])
    action = _make_event(id=2, kind=EventKind.ACTION, refs=[1], source_turns=[5])
    evidence = _make_event(id=3, kind=EventKind.EVIDENCE, refs=[2], source_turns=[6])
    conclusion = _make_event(id=4, kind=EventKind.CONCLUSION, refs=[3])

    # Turn 5 has a tool_call (for action), turn 6 has a tool_result (for evidence).
    turn_kinds = {
        5: _kinds("tool_call", "assistant"),
        6: _kinds("tool_result", "assistant"),
    }
    violations = validate_graph(
        new_events=[task, hyp, action, evidence, conclusion],
        existing_events=[],
        turn_index_to_kinds=turn_kinds,
    )
    assert violations == [], f"expected no violations, got: {violations}"


# ---------------------------------------------------------------------------
# Test 2: unresolved ref — names the offending event and missing ref id
# ---------------------------------------------------------------------------


def test_unresolved_ref_reports_event_and_missing_id() -> None:
    """An event that refs a non-existent id produces a violation.

    Fail-stop: unresolved refs would let the LLM auditor consume a graph
    with dangling pointers, producing incoherent analysis.
    """
    task = _make_event(id=0, kind=EventKind.TASK)
    # hyp refs id=999 which does not exist anywhere.
    hyp = _make_event(id=1, kind=EventKind.HYPOTHESIS, refs=[0, 999])

    violations = validate_graph(
        new_events=[task, hyp],
        existing_events=[],
        turn_index_to_kinds={},
    )
    assert violations, "expected at least one violation for unresolved ref"
    combined = " ".join(violations)
    assert "1" in combined, f"violation should name offending event id 1: {violations}"
    assert "999" in combined, f"violation should name missing ref id 999: {violations}"


# ---------------------------------------------------------------------------
# Test 3: cycle in refs — names the cycle
# ---------------------------------------------------------------------------


def test_cycle_in_refs_reports_cycle() -> None:
    """A.refs=[B] and B.refs=[A] (both non-task) produces a violation.

    Fail-stop: a cycle makes ref-chain traversal infinite, breaking
    task-reachability and conclusion-reachability checks.
    """
    # We need a task so task-reachability doesn't mask the cycle violation.
    # A and B each ref the task AND each other; the task itself is not in
    # the cycle.
    task = _make_event(id=0, kind=EventKind.TASK)
    a = _make_event(id=1, kind=EventKind.HYPOTHESIS, refs=[0, 2])
    b = _make_event(id=2, kind=EventKind.HYPOTHESIS, refs=[0, 1])

    violations = validate_graph(
        new_events=[task, a, b],
        existing_events=[],
        turn_index_to_kinds={},
    )
    assert violations, "expected cycle violation"
    combined = " ".join(violations)
    assert "cycle" in combined.lower(), f"violation should mention cycle: {violations}"


# ---------------------------------------------------------------------------
# Test 4: evidence missing proper source — no tool_result/user/thinking
# ---------------------------------------------------------------------------


def test_evidence_missing_proper_source_names_event_id() -> None:
    """An evidence event whose source_turns only has assistant/text turns
    (no tool_result, no user, no thinking) produces a violation.

    The validator uses turn_index_to_kinds to check what kinds of content
    were present in the referenced turns.

    Fail-stop: evidence backed only by assistant text is epistemically
    ungrounded — the auditor would be misled about what the agent actually
    observed.
    """
    task = _make_event(id=0, kind=EventKind.TASK)
    # evidence refs task and claims source_turn=5, which only has assistant+text.
    evidence = _make_event(id=1, kind=EventKind.EVIDENCE, refs=[0], source_turns=[5])

    turn_kinds = {
        5: _kinds("text", "assistant"),  # no tool_result, no user, no thinking
    }
    violations = validate_graph(
        new_events=[task, evidence],
        existing_events=[],
        turn_index_to_kinds=turn_kinds,
    )
    assert violations, "expected violation for evidence missing proper source"
    combined = " ".join(violations)
    assert "1" in combined, f"violation should name event id 1: {violations}"


# ---------------------------------------------------------------------------
# Test 5: action missing tool_call source
# ---------------------------------------------------------------------------


def test_action_missing_tool_call_source_names_event_id() -> None:
    """An action event whose source_turns lacks any tool_call produces a
    violation.

    Fail-stop: an action without a tool_call source can't be traced back
    to what the agent actually did, making causality analysis unreliable.
    """
    task = _make_event(id=0, kind=EventKind.TASK)
    # action refs task, source_turn=7 has only assistant text — no tool_call.
    action = _make_event(id=1, kind=EventKind.ACTION, refs=[0], source_turns=[7])

    turn_kinds = {
        7: _kinds("text", "assistant"),  # no tool_call
    }
    violations = validate_graph(
        new_events=[task, action],
        existing_events=[],
        turn_index_to_kinds=turn_kinds,
    )
    assert violations, "expected violation for action missing tool_call source"
    combined = " ".join(violations)
    assert "1" in combined, f"violation should name event id 1: {violations}"


# ---------------------------------------------------------------------------
# Test 6: task unreachable — orphan event not connected to any task
# ---------------------------------------------------------------------------


def test_task_unreachable_names_orphan_event() -> None:
    """A non-task event that has no ref path back to any task event produces
    a violation naming the orphan.

    Fail-stop: an unanchored event gives the auditor a floating narrative
    fragment disconnected from the stated task.
    """
    task = _make_event(id=0, kind=EventKind.TASK)
    # orphan has no refs at all — not connected to anything.
    orphan = _make_event(id=1, kind=EventKind.HYPOTHESIS, refs=[])

    violations = validate_graph(
        new_events=[task, orphan],
        existing_events=[],
        turn_index_to_kinds={},
    )
    assert violations, "expected violation for orphan event"
    combined = " ".join(violations)
    assert "1" in combined, f"violation should name orphan event id 1: {violations}"


# ---------------------------------------------------------------------------
# Test 7: conclusion unreachable from task
# ---------------------------------------------------------------------------


def test_conclusion_unreachable_from_task_names_conclusion_id() -> None:
    """A conclusion event whose ref chain doesn't reach any task produces a
    violation.

    Fail-stop: a conclusion that doesn't close the loop to the task is a
    non-sequitur in the audit graph.

    Judgment call: this test uses an orphan chain (conclusion → hypothesis,
    neither connected to task) so the orphan violation fires first; we also
    get a conclusion-reachability violation. We assert both event ids appear
    in the combined violations.
    """
    task = _make_event(id=0, kind=EventKind.TASK)
    # Hypothesis is not connected to task either (orphan chain).
    orphan_hyp = _make_event(id=1, kind=EventKind.HYPOTHESIS, refs=[])
    conclusion = _make_event(id=2, kind=EventKind.CONCLUSION, refs=[1])

    violations = validate_graph(
        new_events=[task, orphan_hyp, conclusion],
        existing_events=[],
        turn_index_to_kinds={},
    )
    assert violations, "expected violation for conclusion not reachable from task"
    combined = " ".join(violations)
    assert "2" in combined, f"violation should name conclusion event id 2: {violations}"


# ---------------------------------------------------------------------------
# Test 8: batch refs prior in same batch — VALID (forward-only allowed)
# ---------------------------------------------------------------------------


def test_batch_refs_prior_in_same_batch_is_valid() -> None:
    """A new event may ref a prior new event from the same batch.

    Design §5.4: 'a new event may ref a prior new event from the same batch
    but never a later one.'

    Fail-stop: rejecting valid intra-batch refs would block normal extraction
    where a hypothesis references the task event submitted in the same batch.
    """
    task = _make_event(id=0, kind=EventKind.TASK)
    # hyp refs task (id=0) which is earlier in the same batch — valid.
    hyp = _make_event(id=1, kind=EventKind.HYPOTHESIS, refs=[0])

    violations = validate_graph(
        new_events=[task, hyp],
        existing_events=[],
        turn_index_to_kinds={},
    )
    assert violations == [], f"intra-batch ref to prior event must be valid, got: {violations}"


# ---------------------------------------------------------------------------
# Test 9: batch refs later in same batch — INVALID
# ---------------------------------------------------------------------------


def test_batch_refs_later_in_same_batch_reports_violation() -> None:
    """An event that refs a later event in the same batch produces a violation.

    Design §5.4: monotonic order — a new event may ref a prior new event
    from the same batch but never a later one.

    Fail-stop: forward refs within a batch break topological ordering of
    the graph and would cause ambiguous traversals.
    """
    # task is id=0 (index 0 in batch), hyp is id=1 (index 1).
    # task refs hyp (id=1) which is LATER in the same batch — invalid.
    task = _make_event(id=0, kind=EventKind.TASK, refs=[1])
    hyp = _make_event(id=1, kind=EventKind.HYPOTHESIS, refs=[0])

    violations = validate_graph(
        new_events=[task, hyp],
        existing_events=[],
        turn_index_to_kinds={},
    )
    assert violations, "expected violation for ref to later event in same batch"
    combined = " ".join(violations)
    assert "0" in combined, f"violation should name offending event id 0: {violations}"
