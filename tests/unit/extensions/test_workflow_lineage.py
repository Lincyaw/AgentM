"""Fail-stop tests for lineage derivation (reliability-substrate §5).

User-authorized invariant (2026-07-10): backward-edge derivation agrees with
the script's real dataflow under standard interpolation patterns.
"""

from __future__ import annotations

import pytest

from agentm.extensions.builtin._workflow.journal import (
    JournalEntry,
    load_journal_entries,
)
from agentm.extensions.builtin._workflow.lineage import ancestors, derive_lineage

from ._workflow_test_helpers import FakeArtifactStore, make_run


def _entry(key: str, result: str, prompt: str | None, ts: float) -> JournalEntry:
    return JournalEntry(key=key, result=result, prompt=prompt, timestamp=ts)


def test_verbatim_interpolation_yields_the_real_dataflow_edges() -> None:
    entries = [
        _entry("A", "alpha-result-123456", "fetch the raw data", 1.0),
        _entry("B", "beta-result-abcdef", "analyze this: alpha-result-123456", 2.0),
        _entry("C", "gamma-result-zzzzzz", "an unrelated task", 3.0),
    ]
    graph = derive_lineage(entries)
    assert {(e.src, e.dst) for e in graph.edges} == {("A", "B")}
    # C has a prompt but no verbatim parent → conservative order fallback.
    assert graph.order_candidates["C"] == ["A", "B"]
    # A is the first node: nothing earlier, so no fallback entry either.
    assert "A" not in graph.order_candidates
    # B has a verbatim parent → no fallback noise.
    assert "B" not in graph.order_candidates


def test_short_generic_results_do_not_create_spurious_edges() -> None:
    entries = [
        _entry("A", "ok", "first task", 1.0),
        _entry("B", "other-result-123", "the answer was ok, continue", 2.0),
    ]
    graph = derive_lineage(entries)
    assert graph.edges == []


def test_long_results_match_beyond_the_fingerprint_prefix() -> None:
    # Result longer than the 256-char fingerprint: the prefix pre-match must
    # still confirm with the full needle, in both directions.
    long_result = "x" * 300 + "-tail-marker"
    entries = [
        _entry("A", long_result, "start", 1.0),
        _entry("B", "downstream-result-1", f"use: {long_result}", 2.0),
        # Same 256-char prefix but different tail → NOT a real dataflow edge.
        _entry("C", "downstream-result-2", "use: " + "x" * 300 + "-other", 3.0),
    ]
    graph = derive_lineage(entries)
    assert {(e.src, e.dst) for e in graph.edges} == {("A", "B")}


def test_ancestors_walks_the_chain_transitively() -> None:
    entries = [
        _entry("A", "level-one-result-aaa", "start", 1.0),
        _entry("B", "level-two-result-bbb", "use level-one-result-aaa", 2.0),
        _entry("C", "level-three-result-ccc", "use level-two-result-bbb", 3.0),
    ]
    graph = derive_lineage(entries)
    assert ancestors(graph, "C") == ["B", "A"]
    assert ancestors(graph, "A") == []


@pytest.mark.asyncio
async def test_derivation_matches_script_dataflow_end_to_end() -> None:
    """Storage round-trip: run a real agent() flow against the journal, then
    derive lineage from what was stored and compare with the script's actual
    variable wiring."""
    store = FakeArtifactStore()
    run = make_run(store, ["upstream-result-123456", "downstream-final-answer"])
    upstream = await run.agent("collect the metrics")
    await run.agent(f"summarize these metrics: {upstream}")

    entries = await load_journal_entries(store)
    graph = derive_lineage(entries)
    key_up = next(e.key for e in entries if e.result == "upstream-result-123456")
    key_down = next(e.key for e in entries if e.result == "downstream-final-answer")
    assert {(e.src, e.dst) for e in graph.edges} == {(key_up, key_down)}
