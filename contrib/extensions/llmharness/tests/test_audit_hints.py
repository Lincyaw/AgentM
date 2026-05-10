"""Tests for audit/hints.py — advisory hint signals (PR 3, REQ-027).

Each test protects a specific signal; no framework-guarantee tests.
Fixtures are hand-crafted list[Event] — no LLM, no spawn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmharness.schema import Event, EventKind


def _ev(
    id: int,
    kind: EventKind,
    summary: str,
    refs: list[int] | None = None,
    source_turns: list[int] | None = None,
) -> Event:
    """Convenience constructor for test fixtures."""
    return Event(
        id=id,
        kind=kind,
        summary=summary,
        refs=refs or [],
        source_turns=source_turns or [],
    )


# ---------------------------------------------------------------------------
# repeated_actions
# ---------------------------------------------------------------------------


class TestRepeatedActions:
    """repeated_actions(graph) -> list[tuple[int, ...]]

    An "action" is identified by the canonical form of its summary.
    Canonicalization: lowercase + collapse runs of whitespace + strip
    trailing punctuation. Two action events with the same canonical
    summary are considered identical actions (potential stuck loop).
    Returns event-id groups of size ≥ 2 (each group is a tuple of ids
    that share the same canonical signature).
    """

    def test_positive_three_identical_actions_among_four(self) -> None:
        """3 of 4 action events share the same summary → group returned."""
        from llmharness.audit.hints import repeated_actions

        graph = [
            _ev(0, EventKind.TASK, "initial task"),
            _ev(1, EventKind.ACTION, "read file README"),
            _ev(2, EventKind.ACTION, "read file README"),  # duplicate
            _ev(3, EventKind.ACTION, "read file README"),  # duplicate
            _ev(4, EventKind.ACTION, "write file output.txt"),  # different
        ]
        groups = repeated_actions(graph)
        # Expect one group containing the three repeated ids.
        assert len(groups) == 1
        group = groups[0]
        assert sorted(group) == [1, 2, 3]

    def test_negative_all_distinct_summaries(self) -> None:
        """4 action events with distinct summaries → no repeats."""
        from llmharness.audit.hints import repeated_actions

        graph = [
            _ev(0, EventKind.TASK, "initial task"),
            _ev(1, EventKind.ACTION, "read file README"),
            _ev(2, EventKind.ACTION, "write file output.txt"),
            _ev(3, EventKind.ACTION, "call api endpoint"),
            _ev(4, EventKind.ACTION, "run test suite"),
        ]
        groups = repeated_actions(graph)
        assert groups == []

    def test_canonicalization_ignores_case_and_whitespace(self) -> None:
        """Canonical form treats 'Read File README' == 'read  file  readme'."""
        from llmharness.audit.hints import repeated_actions

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.ACTION, "Read File README"),
            _ev(2, EventKind.ACTION, "read  file  readme"),  # same after canonicalization
        ]
        groups = repeated_actions(graph)
        assert len(groups) == 1
        assert sorted(groups[0]) == [1, 2]

    def test_non_action_events_ignored(self) -> None:
        """Only events with kind=action are checked; evidence/task ignored."""
        from llmharness.audit.hints import repeated_actions

        graph = [
            _ev(0, EventKind.TASK, "read file README"),
            _ev(1, EventKind.EVIDENCE, "read file README"),
            _ev(2, EventKind.ACTION, "read file README"),
        ]
        # Only one action event — no repeated group possible.
        groups = repeated_actions(graph)
        assert groups == []


# ---------------------------------------------------------------------------
# convergence_ratio
# ---------------------------------------------------------------------------


class TestConvergenceRatio:
    """convergence_ratio(graph) -> float

    Formula: (unresolved_hypotheses + open_decisions) / max(1, total_events)

    An "unresolved hypothesis" is a hypothesis event whose id never appears
    in any other event's refs. An "open decision" is a decision event whose
    id never appears in any other event's refs.

    Returns a float in [0, 1].
    """

    def test_positive_low_convergence_five_unresolved_hypotheses(self) -> None:
        """5 hypotheses, only 1 referenced by downstream → high ratio."""
        from llmharness.audit.hints import convergence_ratio

        # task(0), 5 hypotheses(1-5), 1 evidence(6) referencing only hyp 1.
        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.HYPOTHESIS, "hyp 1"),
            _ev(2, EventKind.HYPOTHESIS, "hyp 2"),
            _ev(3, EventKind.HYPOTHESIS, "hyp 3"),
            _ev(4, EventKind.HYPOTHESIS, "hyp 4"),
            _ev(5, EventKind.HYPOTHESIS, "hyp 5"),
            _ev(6, EventKind.EVIDENCE, "evidence for hyp 1", refs=[1]),
        ]
        ratio = convergence_ratio(graph)
        # 4 unresolved hypotheses, 0 open decisions → 4 / 7 ≈ 0.57
        assert isinstance(ratio, float)
        assert 0 < ratio <= 1
        expected = 4 / 7
        assert abs(ratio - expected) < 1e-9

    def test_negative_high_convergence_all_hypotheses_resolved(self) -> None:
        """3 hypotheses, all referenced → ratio = 0."""
        from llmharness.audit.hints import convergence_ratio

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.HYPOTHESIS, "hyp 1"),
            _ev(2, EventKind.HYPOTHESIS, "hyp 2"),
            _ev(3, EventKind.HYPOTHESIS, "hyp 3"),
            _ev(4, EventKind.EVIDENCE, "evidence 1", refs=[1]),
            _ev(5, EventKind.EVIDENCE, "evidence 2", refs=[2]),
            _ev(6, EventKind.CONCLUSION, "conclusion", refs=[3, 4, 5]),
        ]
        ratio = convergence_ratio(graph)
        assert ratio == 0.0

    def test_returns_float_in_unit_interval(self) -> None:
        """ratio is always in [0, 1]."""
        from llmharness.audit.hints import convergence_ratio

        graph = [_ev(0, EventKind.TASK, "task")]
        ratio = convergence_ratio(graph)
        assert 0.0 <= ratio <= 1.0

    def test_empty_graph_returns_zero(self) -> None:
        """Empty graph does not crash; returns 0.0."""
        from llmharness.audit.hints import convergence_ratio

        ratio = convergence_ratio([])
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# reachability_gaps
# ---------------------------------------------------------------------------


class TestReachabilityGaps:
    """reachability_gaps(graph) -> list[int]

    Returns event ids of evidence events that no conclusion event reaches
    via transitive refs. If no conclusion events exist in the graph, returns
    an empty list (graph still in progress — not a gap).
    """

    def test_positive_evidence_not_reached_by_any_conclusion(self) -> None:
        """An evidence event with no conclusion referencing it → gap."""
        from llmharness.audit.hints import reachability_gaps

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.EVIDENCE, "orphan evidence"),  # not ref'd by conclusion
            _ev(2, EventKind.EVIDENCE, "good evidence"),
            _ev(3, EventKind.CONCLUSION, "conclusion", refs=[2]),
        ]
        gaps = reachability_gaps(graph)
        assert 1 in gaps
        assert 2 not in gaps

    def test_negative_all_evidence_reached_by_conclusion(self) -> None:
        """Linear task → action → evidence → conclusion: no gaps."""
        from llmharness.audit.hints import reachability_gaps

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.ACTION, "do thing"),
            _ev(2, EventKind.EVIDENCE, "result", refs=[1]),
            _ev(3, EventKind.CONCLUSION, "conclusion", refs=[2]),
        ]
        gaps = reachability_gaps(graph)
        assert gaps == []

    def test_no_conclusion_returns_empty(self) -> None:
        """Graph with no conclusion events: return [] (in progress, not a gap)."""
        from llmharness.audit.hints import reachability_gaps

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.EVIDENCE, "evidence"),
        ]
        gaps = reachability_gaps(graph)
        assert gaps == []

    def test_transitive_reach_via_chain(self) -> None:
        """Conclusion → evidence2 → evidence1: evidence1 is reachable transitively."""
        from llmharness.audit.hints import reachability_gaps

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.EVIDENCE, "base evidence"),
            _ev(2, EventKind.EVIDENCE, "derived evidence", refs=[1]),
            _ev(3, EventKind.CONCLUSION, "conclusion", refs=[2]),
        ]
        gaps = reachability_gaps(graph)
        # evidence 1 is reachable transitively: conclusion→ev2→ev1
        assert gaps == []


# ---------------------------------------------------------------------------
# open_branches
# ---------------------------------------------------------------------------


class TestOpenBranches:
    """open_branches(graph) -> list[int]

    Returns ids of decision events whose id never appears in any later
    event's refs (i.e., no evidence or conclusion closes the branch).
    """

    def test_positive_decision_with_no_closing_reference(self) -> None:
        """A decision event never referenced downstream → open branch."""
        from llmharness.audit.hints import open_branches

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.DECISION, "chose approach A"),  # never ref'd
            _ev(2, EventKind.EVIDENCE, "some evidence"),  # refs nothing
        ]
        open_ids = open_branches(graph)
        assert 1 in open_ids

    def test_negative_decision_followed_by_referencing_evidence(self) -> None:
        """A decision event referenced by an evidence → branch is closed."""
        from llmharness.audit.hints import open_branches

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.DECISION, "chose approach A"),
            _ev(2, EventKind.EVIDENCE, "evidence for A", refs=[1]),
        ]
        open_ids = open_branches(graph)
        assert open_ids == []

    def test_multiple_decisions_mixed(self) -> None:
        """Two decisions: one closed, one open."""
        from llmharness.audit.hints import open_branches

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.DECISION, "decision A"),  # closed by ev 3
            _ev(2, EventKind.DECISION, "decision B"),  # open
            _ev(3, EventKind.CONCLUSION, "conclusion", refs=[1]),
        ]
        open_ids = open_branches(graph)
        assert 2 in open_ids
        assert 1 not in open_ids

    def test_no_decisions_returns_empty(self) -> None:
        """Graph with no decision events: no open branches."""
        from llmharness.audit.hints import open_branches

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.EVIDENCE, "evidence"),
            _ev(2, EventKind.CONCLUSION, "conclusion", refs=[1]),
        ]
        assert open_branches(graph) == []


# ---------------------------------------------------------------------------
# multi_branch_syntheses
# ---------------------------------------------------------------------------


class TestMultiBranchSyntheses:
    """multi_branch_syntheses(graph) -> list[tuple[int, int]]

    Returns (conclusion_id, branch_count) for each conclusion event whose
    transitive refs reach two or more independent root paths from task.

    Two paths are "independent" if they share no intermediate events
    (events that are not task itself). The branch count is the number of
    distinct direct-ref sub-graphs of the conclusion that share no
    intermediate events.

    Simplified operational definition: count the direct refs of the
    conclusion that have disjoint ancestor sets (excluding task itself).
    """

    def test_positive_conclusion_reaches_two_independent_branches(self) -> None:
        """Conclusion directly refs two evidence events from independent chains."""
        from llmharness.audit.hints import multi_branch_syntheses

        # task(0) → ev1(1) → ev2(2) (chain A)
        # task(0) → ev3(3) → ev4(4) (chain B)
        # conclusion(5) refs [2, 4] — merges both chains
        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.HYPOTHESIS, "hyp A", refs=[0]),
            _ev(2, EventKind.EVIDENCE, "evidence A", refs=[1]),
            _ev(3, EventKind.HYPOTHESIS, "hyp B", refs=[0]),
            _ev(4, EventKind.EVIDENCE, "evidence B", refs=[3]),
            _ev(5, EventKind.CONCLUSION, "merged conclusion", refs=[2, 4]),
        ]
        results = multi_branch_syntheses(graph)
        assert len(results) == 1
        conc_id, branch_count = results[0]
        assert conc_id == 5
        assert branch_count == 2

    def test_negative_conclusion_on_single_chain(self) -> None:
        """Conclusion with all refs on one linear chain: no multi-branch synthesis."""
        from llmharness.audit.hints import multi_branch_syntheses

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.HYPOTHESIS, "hyp", refs=[0]),
            _ev(2, EventKind.EVIDENCE, "evidence", refs=[1]),
            _ev(3, EventKind.CONCLUSION, "conclusion", refs=[2]),
        ]
        results = multi_branch_syntheses(graph)
        assert results == []

    def test_no_conclusion_returns_empty(self) -> None:
        """No conclusion events → no synthesis records."""
        from llmharness.audit.hints import multi_branch_syntheses

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.EVIDENCE, "evidence"),
        ]
        assert multi_branch_syntheses(graph) == []

    def test_single_direct_ref_conclusion_not_multi_branch(self) -> None:
        """Conclusion with only one direct ref: branch_count=1, not returned."""
        from llmharness.audit.hints import multi_branch_syntheses

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.EVIDENCE, "evidence", refs=[0]),
            _ev(2, EventKind.CONCLUSION, "conclusion", refs=[1]),
        ]
        results = multi_branch_syntheses(graph)
        assert results == []


# ---------------------------------------------------------------------------
# compute renderer
# ---------------------------------------------------------------------------


class TestCompute:
    """compute(graph) -> str

    Renders all anomaly signals into a compact textual block.
    Returns empty string when no anomaly is found.
    """

    def test_clean_linear_graph_returns_empty_string(self) -> None:
        """No anomalies in a linear graph → empty string."""
        from llmharness.audit.hints import compute

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.ACTION, "do thing"),
            _ev(2, EventKind.EVIDENCE, "result", refs=[1]),
            _ev(3, EventKind.CONCLUSION, "done", refs=[2]),
        ]
        result = compute(graph)
        assert result == ""

    def test_multi_anomaly_graph_contains_expected_tokens(self) -> None:
        """Graph with multiple anomalies: block mentions anomaly categories."""
        from llmharness.audit.hints import compute

        # repeated actions (ev 1, 2, 3) + open branch (ev 4) + orphan evidence (ev 5)
        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.ACTION, "read file readme"),
            _ev(2, EventKind.ACTION, "read file readme"),  # repeat
            _ev(3, EventKind.ACTION, "read file readme"),  # repeat
            _ev(4, EventKind.DECISION, "open decision"),  # never ref'd
            _ev(5, EventKind.EVIDENCE, "orphan evidence"),  # not reached by conclusion
            _ev(6, EventKind.CONCLUSION, "conclusion", refs=[]),  # refs nothing
        ]
        result = compute(graph)
        assert isinstance(result, str)
        assert len(result) > 0
        # Must mention the relevant categories.
        low = result.lower()
        assert "repeated" in low or "repeat" in low
        assert "open branch" in low or "open_branch" in low or "branch" in low

    def test_result_is_always_a_string(self) -> None:
        """compute never crashes; always returns str."""
        from llmharness.audit.hints import compute

        assert isinstance(compute([]), str)
        assert isinstance(compute([_ev(0, EventKind.TASK, "t")]), str)

    def test_nonempty_block_ends_with_newline(self) -> None:
        """Non-empty hints block ends with a newline for clean concatenation."""
        from llmharness.audit.hints import compute

        # Repeated action anomaly.
        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.ACTION, "same action"),
            _ev(2, EventKind.ACTION, "same action"),
        ]
        result = compute(graph)
        if result:  # only assert when non-empty
            assert result.endswith("\n")

    def test_phrasing_uses_consider_not_concern(self) -> None:
        """Hints are phrased as 'consider …', never 'concern: …' (design §7.5)."""
        from llmharness.audit.hints import compute

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.ACTION, "repeated action"),
            _ev(2, EventKind.ACTION, "repeated action"),
        ]
        result = compute(graph)
        if result:
            assert "consider" in result.lower()
            assert "concern:" not in result.lower()

    def test_result_contains_event_ids_when_anomaly_present(self) -> None:
        """Event ids appear in the rendered block for traceability."""
        from llmharness.audit.hints import compute

        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(10, EventKind.ACTION, "do the thing"),
            _ev(11, EventKind.ACTION, "do the thing"),  # repeat → ids 10, 11 should appear
        ]
        result = compute(graph)
        # Both ids should appear so the auditor can locate the events.
        assert "10" in result
        assert "11" in result

    # Snapshot test: pin the exact rendered output for one canonical fixture.
    def test_snapshot_multi_anomaly_canonical_output(self) -> None:
        """Snapshot: exact renderer output for a canonical multi-anomaly graph.

        Pinned so any accidental phrasing drift is caught immediately.
        Update this expected string when the renderer is intentionally changed.
        """
        from llmharness.audit.hints import compute

        # Canonical fixture: repeated actions (ids 1,2,3) + open decision (id 4).
        # No conclusion → reachability_gaps silent (in-progress graph).
        graph = [
            _ev(0, EventKind.TASK, "task"),
            _ev(1, EventKind.ACTION, "read file readme"),
            _ev(2, EventKind.ACTION, "read file readme"),
            _ev(3, EventKind.ACTION, "read file readme"),
            _ev(4, EventKind.DECISION, "chose approach A"),
        ]
        result = compute(graph)
        EXPECTED = (
            "Advisory hints (consider — not directives):\n"
            "- consider: repeated action detected (event ids: 1, 2, 3)\n"
            "- consider: open branch — decision event(s) with no closing evidence (ids: 4)\n"
            "\n"
        )
        assert result == EXPECTED, (
            f"snapshot mismatch:\n---got---\n{result}\n---want---\n{EXPECTED}"
        )


# ---------------------------------------------------------------------------
# Integration: Scenario F (hints in payload)
# ---------------------------------------------------------------------------


class TestScenarioFHintsInPayload:
    """Integration test wired into the adapter.

    Builds a session with k=3, stubs the extractor to produce an open_branch
    candidate (a decision event with no closing evidence), stubs the auditor
    to record its received JSON payload, and verifies that:
    - payload["hints"] is a non-empty string
    - the string contains "open branch" (or equivalent token)
    """

    pass  # implemented below as a pytest async test function


@pytest.mark.asyncio
async def test_scenario_f_hints_in_auditor_payload(tmp_path: Path) -> None:
    """Scenario F (PR 3 wire-up test).

    Adapter must compute hints from the graph and pass them in the auditor
    payload. Here the graph has an open decision event (id=0, kind=decision
    never ref'd), so open_branches fires and the hints block is non-empty.
    """
    import json
    import sys
    import types
    from collections.abc import AsyncIterator
    from typing import Any

    from agentm.core.abi import (
        AssistantMessage,
        AssistantStreamEvent,
        MessageEnd,
        Model,
        TextContent,
        ToolCallBlock,
    )
    from agentm.harness.extension import ProviderConfig
    from agentm.harness.session import AgentSession, AgentSessionConfig

    from llmharness.audit._enum_schema import EVENT_KIND_VALUES
    from llmharness.audit.auditor import SUBMIT_VERDICT_TOOL_NAME
    from llmharness.audit.extractor import SUBMIT_EVENTS_TOOL_NAME

    _EXTRACTOR_NEEDLE = "cognitive-audit **extractor**"
    _AUDITOR_NEEDLE = "cognitive-audit *auditor*"

    class _ScenarioFProvider:
        def __init__(self) -> None:
            self.extractor_calls = 0
            self.auditor_calls = 0
            self.auditor_payloads: list[Any] = []
            self.parent_calls = 0

        def __call__(
            self,
            *,
            messages: list[Any],
            model: Model,
            tools: list[Any],
            system: str | None = None,
            signal: Any = None,
            thinking: str = "off",
        ) -> AsyncIterator[AssistantStreamEvent]:
            del model, tools, signal, thinking
            sys_text = system or ""
            if _EXTRACTOR_NEEDLE in sys_text:
                self.extractor_calls += 1
                return self._extractor_iter(self.extractor_calls)
            if _AUDITOR_NEEDLE in sys_text:
                self.auditor_calls += 1
                if messages:
                    self.auditor_payloads.append(messages[-1])
                return self._auditor_iter()
            self.parent_calls += 1
            return self._parent_iter(self.parent_calls)

        async def _parent_iter(self, n: int) -> AsyncIterator[AssistantStreamEvent]:
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=f"turn-{n}")],
                timestamp=float(n),
                stop_reason="end_turn",
            )
            yield MessageEnd(message=msg)

        async def _extractor_iter(self, n: int) -> AsyncIterator[AssistantStreamEvent]:
            # Turn 1 (n=1): emit a task event (gets id=0 in the adapter).
            # Turn 2 (n=2): emit a decision event referencing the task (id=0,
            #               already in existing_events). This decision is never
            #               ref'd by any later event → open_branches fires.
            # Turn 3 (n=3): emit another task event for this window.
            #
            # This ensures the graph passes Phase 1 validation (task reachability
            # satisfied: decision refs task id=0) while the open-branch signal fires
            # (the decision event is never referenced by any subsequent event).
            if n == 1:
                # First window: just a task event.
                events: list[dict[str, Any]] = [
                    {
                        "kind": EVENT_KIND_VALUES[0],  # task
                        "summary": "initial task",
                        "source_turns": [],
                        "refs": [],
                    }
                ]
            elif n == 2:
                # Second window: a decision referencing the task (id=0 in existing).
                events = [
                    {
                        "kind": EVENT_KIND_VALUES[3],  # decision
                        "summary": "open decision — chose approach A",
                        "source_turns": [],
                        "refs": [0],  # refs task (existing id=0)
                    }
                ]
            else:
                # Third window: another task-level event (new task description).
                events = [
                    {
                        "kind": EVENT_KIND_VALUES[0],  # task
                        "summary": f"continued task window {n}",
                        "source_turns": [],
                        "refs": [],
                    }
                ]
            msg = AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id=f"submit-events-{n}",
                        name=SUBMIT_EVENTS_TOOL_NAME,
                        arguments={"events": events},
                    )
                ],
                timestamp=float(100 + n),
                stop_reason="tool_use",
            )
            yield MessageEnd(message=msg)

        async def _auditor_iter(self) -> AsyncIterator[AssistantStreamEvent]:
            msg = AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id=f"submit-verdict-{self.auditor_calls}",
                        name=SUBMIT_VERDICT_TOOL_NAME,
                        arguments={
                            "verdict": {
                                "surface_reminder": False,
                                "reminder_text": "",
                                "continuation_notes": [],
                                "matched_event_ids": [],
                                "cited_cards": [],
                            }
                        },
                    )
                ],
                timestamp=float(200 + self.auditor_calls),
                stop_reason="tool_use",
            )
            yield MessageEnd(message=msg)

    provider = _ScenarioFProvider()
    module_name = "tests._fake_scenario_f_provider"
    mod = types.ModuleType(module_name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-scenario-f",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-scenario-f",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-scenario-f",
            ),
        )

    mod.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = mod

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(module_name, {}),
            extensions=[
                (
                    "llmharness.adapters.agentm",
                    {
                        "mode": "sync",
                        "audit_interval_turns": 3,
                        "cards_tools_config": None,
                        "observability_config": None,
                    },
                ),
            ],
        )
    )

    # 3 turns at k=3 → auditor fires once at turn 3.
    for i in range(3):
        await session.prompt(f"user turn {i + 1}")

    await session.shutdown()

    # Auditor must have been called exactly once.
    assert provider.auditor_calls == 1, f"expected 1 auditor call, got {provider.auditor_calls}"

    # Extract the payload from the captured auditor message.
    assert len(provider.auditor_payloads) >= 1
    raw_msg = provider.auditor_payloads[0]
    content_blocks = getattr(raw_msg, "content", None) or []
    payload_text = ""
    for block in content_blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            payload_text = text
            break

    assert payload_text, "auditor payload must contain text"
    payload_dict = json.loads(payload_text)

    # hints must be a non-empty string.
    hints = payload_dict.get("hints")
    assert isinstance(hints, str), f"hints must be a string, got {type(hints)}"
    assert len(hints) > 0, "hints must be non-empty (open_branch signal should fire)"

    # Must mention "open branch".
    assert "open branch" in hints.lower(), f"hints block must mention 'open branch', got: {hints!r}"
