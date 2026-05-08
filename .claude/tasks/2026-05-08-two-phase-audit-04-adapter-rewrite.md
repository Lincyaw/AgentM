# Task 04: Adapter Rewrite (Two-Phase Orchestrator)

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-two-phase-audit.md)
**Design**: [design](../designs/llmharness-two-phase-audit.md) §3, §4, §7.3, §7.4
**Assignee**: implementer
**Size**: M
**Dependencies**: 02, 03

## Objective

Rewrite `adapters/agentm.py` as the two-phase orchestrator. Per
`TurnEndEvent`: always run Phase 1 (extractor), and run Phase 2 (auditor)
when `turn_count % k == 0`. Maintain the `extractor_cursor` entry. Slice
new turns *with* thinking blocks for the extractor. Pass graph + recent
verdicts to the auditor.

## Files

Modify (full rewrite):
- `scenarios/llmharness/src/llmharness/adapters/agentm.py`

Read:
- Current `adapters/agentm.py` (V0 single-pass orchestrator)
- `audit/extractor/` (task 02) and `audit/auditor/` (task 03)
- `schema.py` (`Event`, `Verdict`, `Reminder`)

## Behavior

### State (closure-locals on the handler)
- `turn_count: int` — count of `TurnEndEvent` firings observed.
- `pending: Reminder | None` — reminder to inject on next `BeforeAgentStartEvent`.
- `k: int` — read from atom config, default 3, knob name `audit_interval_turns`.

### `BeforeAgentStartEvent` handler
- If `pending` is set, mutate `event.system` to append the reminder body
  in the `[harness] {free text}` half-structured form (V0 behavior; design §9).
- Clear `pending` after injection.
- This handler stays unchanged from V0 in shape — only the source of
  `pending` shifts to Phase 2.

### `TurnEndEvent` handler

```
turn_count += 1
branch = api.session.get_branch()
cursor = _read_extractor_cursor(branch)         # last_turn_index + run_id
new_turn_window = _slice_new_turns(branch, cursor)   # incl. thinking blocks
new_events = await _run_extractor(api, branch, new_turn_window)
_write_extractor_cursor(api, last_turn_index=..., run_id=fresh_uuid)

if (turn_count % k) == 0:
    graph_events = _collect_audit_events(branch) + new_events
    recent_verdicts = _collect_recent_verdicts(branch)
    verdict = await _run_auditor(api, graph_events, recent_verdicts)
    if verdict and verdict.drift and verdict.reminder and verdict.type:
        pending = verdict.reminder
```

### `_run_extractor`
- Builds `compose_extractor_extensions()` list.
- `api.spawn_child_session(extensions=..., provider=None)` (auto-wires via `inherit_provider` builtin — unchanged from V0).
- Hands the new-turn window + recent-graph slice to the child as the user message (the SDK already serializes thinking blocks; pass them through).
- Awaits child to call `submit_events` (`ToolTerminate`); coerces via `RawExtractorOutput`.
- Failure paths delegate to task 05 (`_record_failure` helper).

### `_run_auditor`
- Builds `compose_auditor_extensions()` list, spawns child same way.
- User-message content is the structured graph: `[Event.to_dict() for e in graph_events]` plus the last N verdicts (N = small const, e.g. 5).
- Auditor receives NO trajectory serialization (design §7.4).
- Awaits `submit_verdict`; coerces via `RawVerdictOutput`.

### Cursor entry
- Type: `llmharness.extractor_cursor`
- Payload: `{"last_turn_index": int, "extraction_run_id": str}` (fresh UUID per firing).
- Read by walking `branch` for the most recent entry of this type; if absent (first firing) treat `last_turn_index = -1`.

### Trajectory slice (`_slice_new_turns`)
- Slices `messages[cursor.last_turn_index + 1 :]` from the live branch.
- Keeps thinking blocks; only drops blocks the SDK does not surface to extensions.
- Tool-result content kept structured (not flattened).
- Updates `last_turn_index` to the index of the last message included.

## Acceptance

- [ ] V1 adapter present at `adapters/agentm.py`; V0 single-pass shape gone.
- [ ] `k` knob read from atom config under name `audit_interval_turns`, default 3.
- [ ] `BeforeAgentStartEvent` reminder injection shape preserved (V0 behavior).
- [ ] On the first `TurnEndEvent`, `extractor_cursor` is absent and treated as `last_turn_index=-1`; subsequent firings read and update it.
- [ ] Phase 2 only fires when `turn_count % k == 0` (test in task 07).
- [ ] Auditor input contains zero raw-trajectory fields — graph (`Event.to_dict`) + recent verdicts only.
- [ ] `mypy --strict`, `ruff check` clean for the new file.

## Notes

- Failure-mode wiring is task 05's responsibility; for task 04 it is enough to
  add a stub `_record_failure` call site marked `# TODO(task-05)`.
- `inherit_provider` and `spawn_child_session(provider=None)` paths
  unchanged — do not introduce per-phase provider plumbing.
- `fetch_turn` drill-down tool stays unregistered (design §2, §10).
- Do NOT introduce a typed graph query API on `ReadonlySession` (design §5.3
  — V1 stays branch-walk; promotion deferred until a second consumer exists).
