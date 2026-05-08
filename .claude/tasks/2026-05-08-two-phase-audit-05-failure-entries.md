# Task 05: Failure Entry Types + `_record_failure`

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-two-phase-audit.md)
**Design**: [design](../designs/llmharness-two-phase-audit.md) §4, §8
**Assignee**: implementer
**Size**: S
**Dependencies**: 04

## Objective

Replace V0's silent `Verdict(drift=False)` fallback with five typed
failure entries so audit-pipeline outage is visible in the entry stream.
Add a single `_record_failure(api, entry_type, payload)` helper used by
both phase paths in the adapter.

## Files

Modify:
- `scenarios/llmharness/src/llmharness/adapters/agentm.py` (add helper, wire failure paths from task 04 stubs)

## Entry Types

| Entry type | Producer phase | Trigger | Payload schema |
|---|---|---|---|
| `llmharness.extractor_no_call` | Phase 1 | child exits without calling `submit_events` | `{"reason": str, "turn_window": [a, b]}` |
| `llmharness.extractor_error` | Phase 1 | spawn or prompt raised | `{"reason": str, "turn_window": [a, b]}` |
| `llmharness.extractor_empty` | Phase 1 | submitted but `events == []` AND `turn_window` non-trivial | `{"turn_window": [a, b]}` |
| `llmharness.audit_no_call` | Phase 2 | child exits without calling `submit_verdict` | `{"reason": str}` |
| `llmharness.audit_error` | Phase 2 | spawn or prompt raised | `{"reason": str}` |

"Non-trivial window" definition (extractor_empty): `turn_window` covers at
least one assistant message OR one tool call/result. A pure
user-only window does not warrant `extractor_empty`.

## `_record_failure` Contract

```python
def _record_failure(
    api: ExtensionAPI,
    entry_type: str,           # one of the five above
    payload: dict[str, Any],
) -> None:
    """Append a typed failure entry to the parent session's branch."""
```

- Single chokepoint; phase paths call this and continue (do NOT raise).
- Payload validated against the table above (assert in dev; permissive in prod).

## Wiring

In `_run_extractor` (task 04):
- `try: spawn ... await ...` — on exception → `extractor_error`.
- After child returns: if `submit_events` was not called (no terminate
  payload) → `extractor_no_call`.
- If parsed `events == []` and the window is non-trivial → `extractor_empty`.

In `_run_auditor`:
- Mirror the first two cases for `audit_no_call` / `audit_error`.
- No `audit_empty`: `submit_verdict` always carries a verdict; provider-side
  if/then (task 03) prevents the `drift=true,type=null` malformed case.

## Acceptance

- [ ] `_record_failure` exists in `adapters/agentm.py` and is the single
      writer of failure entries.
- [ ] All five entry types are reachable from the adapter (covered by
      task 07 integration test plus a small unit-style probe — see Notes).
- [ ] Phase paths never raise out of the adapter; failures are recorded
      and the handler returns normally.
- [ ] `extractor_empty` does NOT fire for pure user-only windows.
- [ ] `mypy --strict`, `ruff check` clean.

## Notes

- Auditor reading the graph (task 04) MUST filter out failure entries —
  only `llmharness.audit_event` is the graph; failure entries are
  diagnostic-only. Confirm filter is in place when reviewing this task.
- A small probe in `tests/test_two_phase_audit_integration.py` (task 07)
  injects a deliberate stub-provider failure to assert one of the five
  entries lands; this is the regression lock for §8.
