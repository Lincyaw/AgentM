"""Fail-stop tests for workflow journal invalidation (reliability-substrate §5).

User-authorized invariants (2026-07-10):

1. A flagged journal entry is never served as a resume hit; a re-run result
   newer than the flag supersedes it (including the key-shift cascade that
   re-runs exactly the affected downstream subgraph).
2. An invalidation's feedback reaches the re-run child's prompt, and the
   result is recorded under the original key (addressability preserved).
"""

from __future__ import annotations

import pytest

from agentm.extensions.builtin._workflow.journal import (
    load_journal_entries,
    write_invalidation,
)

from ._workflow_test_helpers import FakeArtifactStore, StubRun, make_run


async def _only_key(store: FakeArtifactStore) -> str:
    entries = await load_journal_entries(store)
    assert len(entries) == 1
    return entries[0].key


@pytest.mark.asyncio
async def test_flagged_entry_forces_rerun_and_new_result_supersedes() -> None:
    store = FakeArtifactStore()

    out1 = await make_run(store, ["old result value"]).agent("analyze the data")
    assert out1 == "old result value"

    # Resume without invalidation: served from the journal, no spawn.
    run_hit = make_run(store, [])
    assert await run_hit.agent("analyze the data") == "old result value"
    assert run_hit.spawn_prompts == []

    key = await _only_key(store)
    await write_invalidation(store, key=key, reason="the number is wrong")

    # Flagged → forced miss: the node re-runs.
    run_rerun = make_run(store, ["new result value"])
    assert await run_rerun.agent("analyze the data") == "new result value"
    assert len(run_rerun.spawn_prompts) == 1

    # The fresh result supersedes the flag: next resume hits, no spawn.
    run_after = make_run(store, [])
    assert await run_after.agent("analyze the data") == "new result value"
    assert run_after.spawn_prompts == []


@pytest.mark.asyncio
async def test_feedback_reaches_rerun_prompt_under_original_key() -> None:
    store = FakeArtifactStore()
    await make_run(store, ["old result value"]).agent("analyze the data")
    key = await _only_key(store)

    await write_invalidation(
        store,
        key=key,
        reason="dropped the negative rows",
        feedback="keep negative values when cleaning",
        carry_previous=True,
    )

    run = make_run(store, ["corrected result"])
    assert await run.agent("analyze the data") == "corrected result"

    executed_prompt = run.spawn_prompts[0]
    assert executed_prompt.startswith("analyze the data")
    assert "keep negative values when cleaning" in executed_prompt
    assert "dropped the negative rows" in executed_prompt
    assert "old result value" in executed_prompt  # carry_previous

    entries = await load_journal_entries(store)
    assert [entry.key for entry in entries] == [key]  # original key, newest wins
    assert entries[0].result == "corrected result"
    assert entries[0].prompt == "analyze the data"  # identity, not the injection
    assert entries[0].invalidated is False  # flag superseded


@pytest.mark.asyncio
async def test_cascade_reruns_exactly_the_affected_subgraph() -> None:
    store = FakeArtifactStore()

    async def flow(run: StubRun) -> tuple[object, object, object]:
        a = await run.agent("produce the base number for the report")
        b = await run.agent(f"write one sentence using: {a}")
        c = await run.agent("translate 'hello' to French")
        return a, b, c

    run1 = make_run(
        store,
        ["base-number-result-1234", "sentence about 1234", "bonjour-result-xyz"],
    )
    await flow(run1)
    assert len(run1.spawn_prompts) == 3

    entries = await load_journal_entries(store)
    key_a = next(e.key for e in entries if e.result == "base-number-result-1234")
    await write_invalidation(store, key=key_a, reason="base number is wrong")

    # Resume: A re-runs (flagged), B re-runs (its prompt embeds A's new
    # result, so its key shifted), C keeps its cached hit.
    run2 = make_run(store, ["base-number-result-5678", "sentence about 5678"])
    a2, b2, c2 = await flow(run2)
    assert a2 == "base-number-result-5678"
    assert b2 == "sentence about 5678"
    assert c2 == "bonjour-result-xyz"
    assert len(run2.spawn_prompts) == 2
