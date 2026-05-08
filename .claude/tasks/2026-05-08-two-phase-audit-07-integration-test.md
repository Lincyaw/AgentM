# Task 07: V1 Stub-Provider Integration Test (Fail-Stop)

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-two-phase-audit.md)
**Design**: [design](../designs/llmharness-two-phase-audit.md) §11
**Assignee**: tdd
**Size**: M
**Dependencies**: 06

## Objective

Lock the V1 contract behind a stub-provider integration test that drives a
4-turn dialog at `k=3` and asserts the entry-tree shape. **This is the V1
fail-stop per CLAUDE.md "Core test positions" — the single test that
replaces V0's smoke pinning.**

## Why This Test Exists (Fail-Stop Justification)

Per CLAUDE.md testing philosophy: a test is justified only if a realistic
disaster ships when it's missing. This test protects against:

1. **Phase trigger regression** — Phase 2 firing every turn (token blowup) or never (silent watchdog).
2. **Cursor drift** — extractor re-extracting the same window or skipping a window.
3. **Failure-entry silence regression** — V0's silent-fallback bug returning under a different name.
4. **Schema contract drift** — `submit_events`/`submit_verdict` shape changing without coordination.

If broken, the audit pipeline can be silently dead while the rest of the
session looks fine — exactly the V0 failure mode V1 was designed to fix.

## Files

Create:
- `/home/ddq/AoyangSpace/AgentM/scenarios/llmharness/tests/test_two_phase_audit_integration.py`

Read:
- Adapter from task 04 (`adapters/agentm.py`)
- Extractor + auditor modules from tasks 02/03
- Existing AgentM stub-provider integration tests for setup pattern (search `tests/` in main repo for `StubProvider` or similar fixtures)

## Test Scenarios

### Scenario A: Happy-path 4-turn dialog at k=3

Setup:
- Spawn an `AgentSession` with the V1 audit adapter loaded and `audit_interval_turns=3`.
- Use a deterministic stub provider that:
  - For the main agent: emits 4 turns of plain assistant responses (no tool calls necessary).
  - For the extractor child: every spawn calls `submit_events` with one synthetic event referencing the turn window.
  - For the auditor child: calls `submit_verdict` with `drift=False`.

Drive 4 user turns → expect 4 `TurnEndEvent` firings.

Assert on the final entry tree of the parent session:
- [ ] Exactly 4 `llmharness.extractor_cursor` entries (one per Phase 1 firing).
- [ ] At least 4 `llmharness.audit_event` entries (one per turn from the stub).
- [ ] Exactly 1 `llmharness.verdict` entry (Phase 2 fired only at turn 3, since `4 % 3 != 0` and `1,2 % 3 != 0`; design §3 says `(turn_count % k) == 0`, so turn 3 is the only hit in a 4-turn run).
- [ ] Cursor `last_turn_index` is monotonically non-decreasing across the 4 cursor entries.

### Scenario B: Failure-entry probe

Setup variant:
- Stub provider configured so the *extractor* child on turn 2 exits without
  calling `submit_events` (e.g. emits a final assistant message and stops).
- All other turns happy-path.

Assert:
- [ ] Exactly 1 `llmharness.extractor_no_call` entry, payload includes
      `turn_window` covering turn 2.
- [ ] Other turns still produce `audit_event` entries normally.
- [ ] No `llmharness.verdict` with `drift=False` was synthesized as a
      silent fallback (V0 regression check).

### Scenario C: Schema contract probe

- [ ] Import shape: `from llmharness.audit.extractor import compose_extractor_extensions, SUBMIT_EVENTS_TOOL_NAME` works.
- [ ] Import shape: `from llmharness.audit.auditor import compose_auditor_extensions, SUBMIT_VERDICT_TOOL_NAME` works.
- [ ] `SUBMIT_EVENTS_PARAMETERS["properties"]["events"]["items"]["properties"]["kind"]["enum"] == EVENT_KIND_VALUES`.
- [ ] `SUBMIT_VERDICT_PARAMETERS` has an `if/then` clause keyed off `drift=True` requiring `type`.

## Acceptance

- [ ] Test file exists, runs in CI without API keys (stub provider only).
- [ ] All three scenarios pass deterministically (no flakes across 10 reruns).
- [ ] `mypy --strict`, `ruff check` clean for the test file.
- [ ] Test docstring restates "this is the V1 fail-stop" with pointers to
      design §11 and CLAUDE.md "Core test positions".

## Notes

- Stub provider must be reusable from a unit-style `pytest` test — do NOT
  spin up a real CLI. CLAUDE.md's E2E rule covers user-facing behavior;
  this test is a stub-provider integration test (CLAUDE.md says these are
  the right shape for regression locks once a real bug is found in
  trajectory inspection).
- Per design §11, this test replaces all V0 schema-pinning smoke tests.
- If `audit_interval_turns` knob plumbing is missing, that surfaces here
  rather than at runtime — surface it as a clear `KeyError`/config error,
  do not paper over.
- Keep assertions on entry types and counts (boundary-style) rather than
  on free-text content — content is LLM/stub-dependent.
