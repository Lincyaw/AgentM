# Task: Layer B Wave-Crossover Completion Report (B-9, B-4)

**Date**: 2026-05-09
**Status**: COMPLETE
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)

## Tasks completed

- B-9 — Structural `stop_after_no_improvement` counter — DONE
- B-4 — System Aware Merge (crossover) — DONE

Order: **B-9 first** (smaller, plumbing-only), **B-4 second** (more
invasive, builds on the counter). The B-9 commit is conservative —
gate fires only on `decision="activate"`. The B-4 commit broadens the
gate to `{"activate", "merge"}` so a successful merge resets the
counter and a rejected merge increments it.

## Lines changed per file

| File | Insertions | Deletions |
|---|---:|---:|
| `src/agentm/extensions/builtin/tool_propose_change.py` | 180 | 4 |
| `src/agentm/extensions/builtin/tool_query_candidates.py` | 19 | 3 |
| `tests/integration/test_per_task_evolution.py` | 7 | 3 |
| `tests/integration/test_stop_after_no_improvement.py` (new) | 349 | 0 |
| `tests/integration/test_merge_gate.py` (new) | 429 | 0 |
| **Total** | **984** | **10** |

Substantive Python source change: ~196 LoC across the two builtin
atoms. Tests: 785 LoC (heavy because each scenario stands up its own
session + on-disk fixture). Total Python diff is 974 / 10 — within the
250-400 LoC source-code estimate; the test footprint is the bulk.

## Commits

```
e59968e feat(propose_change): B-4 System Aware Merge crossover channel
2f0b0a7 feat(propose_change): B-9 structural stop_after_no_improvement counter
```

Both pushed to local branch only (no remote push performed, per
instructions).

## New tests (count + fail-stop class)

5 new tests total. Fail-stop class: **structural enforcement of
deployment-gate semantics across decision channels**.

### B-9 (`tests/integration/test_stop_after_no_improvement.py`)

- `test_stop_counter_persists_across_session_restart` — seeds 3
  `kind=rejected` records on disk, then opens a fresh session and
  asserts the next `decision="activate"` is refused with
  `stop_after_no_improvement: 3 consecutive rejections`. Pass means
  the counter was reconstructed from `activations.jsonl`, not from
  in-memory state. Without persistence the constraint is vacuous.
- `test_stop_counter_resets_after_successful_activate` — seeds
  `[reject, reject, activate, reject]`. With threshold=3, only 1
  trailing reject sits at the tail; the next activate is NOT
  blocked by the stop gate. Without reset the counter is monotonic
  and every tuner is eventually permanently blocked.

### B-4 (`tests/integration/test_merge_gate.py`)

- `test_merge_within_noise_floor_rejected` — the headline fail-stop.
  Constructs a merge proposal with parents `c_parent_a` + `c_parent_b`,
  baseline `0.80 ± 0.20`, proposed `0.90 ± 0.20`. Threshold passes
  (delta/baseline = 0.125 > 0.05) but 2σ ≈ 0.566 >> 0.10 = delta —
  so the noise floor MUST reject. Asserts `is_error`, `deployment
  gate failed`, atom-on-disk unchanged, and the activations.jsonl
  entry has `kind=rejected, decision="merge", parent_ids=[a,b]`.
- `test_merge_passing_all_floors_activates` — complement. Baseline
  `0.10 ± 0.05`, proposed `0.95 ± 0.05`. Asserts the merge child
  activates, atom-on-disk swaps to v2, the activation record has
  `kind="merge"` with `parent_ids` preserved, and the candidate
  companion JSON carries the multi-parent schema.
- `test_merge_requires_two_parents` — sanity. `parents=[]`/single
  parent / unknown parent ID all reject before any gate work.

## Schema migration: `parent_id` → `parent_ids`

Candidate records (`.agentm/decisions/<scenario>/candidates/<id>.json`)
now carry `parent_ids: list[str]` in place of the legacy
`parent_id: str | None` scalar. Migration semantics:

- **Writer** (`tool_propose_change`): always emits `parent_ids`.
  - First activation in the log → `[]` (was `null`).
  - Subsequent `activate` / `exploratory` / `rollback` → `[<prior>]`.
  - `merge` → the validated `>=2` parents from args.
- **Reader** (`tool_query_candidates`): accepts BOTH shapes via
  `_coerce_parent_ids(rec)`. New `parent_ids` field wins; legacy
  `parent_id` is promoted to a 1-element list. Pre-B-4 fixtures
  (e.g. the existing `test_pareto_inclusion.py`) keep passing.
- **`tree.jsonl`**: now records one edge per parent so the lineage
  graph is complete for both single-parent (activate) and
  multi-parent (merge) cases.
- **Activations.jsonl** records: `kind=rejected` and the success
  record both gain `parent_ids: list[str]`. The success record's
  `kind` is `"merge"` for merge entries (vs `"activate"` for
  single-parent), so consumers can distinguish without inspecting
  `parent_ids` length.

The existing test `test_per_task_evolution.py::test_end_to_end_loop_…`
was updated to assert the new key set (`parent_ids` instead of
`parent_id`) and the empty-list value for first activations. No
production fixtures lived under git — the only on-disk consumer was
test fixtures and the `_coerce_parent_ids` reader covers them.

## Counter semantics summary (B-9)

The counter is the count of contiguous "non-progress" entries
preceding the most-recent log line. Walk `activations.jsonl` from end
backward:

| Record kind | Counter effect | Reason |
|---|---|---|
| `rejected` | +1 | Gate said no; counts as no-progress |
| `reload_failed` | +1 | Attempted swap that didn't land |
| `stop_blocked` | transparent (skip) | Avoid self-perpetuating block |
| `activate` | reset (stop walk) | Forward progress |
| `exploratory` | reset (stop walk) | Intentional gate-bypass = progress |
| `rollback` | reset (stop walk) | Deliberate decision |
| `merge` | reset (stop walk) | Forward progress (B-4) |
| `pending_human_approval` | reset (stop walk) | Decision was made |

Threshold reached → return `is_error=True` with text
`"stop_after_no_improvement: <n> consecutive rejections; change
strategy or escalate"`. A `kind=stop_blocked` audit record is
appended (carries `decision`, `evidence.consecutive_rejections`,
`evidence.threshold`) but does not itself feed the counter.

Default threshold: 3. Configure via
`tool_propose_change.config.promotion.stop_after_no_improvement`
(int, or `None` / non-positive to disable).

## Test status

`uv run pytest --tb=short` → **109 passed, 1 skipped, 14 deselected**
(was 104 passed; +5 new tests across the two new files). Distribution:

- 2 in `test_stop_after_no_improvement.py` (B-9).
- 3 in `test_merge_gate.py` (B-4).

`uv run ruff check src/agentm/extensions/builtin/tool_propose_change.py
src/agentm/extensions/builtin/tool_query_candidates.py
tests/integration/test_merge_gate.py
tests/integration/test_stop_after_no_improvement.py
tests/integration/test_per_task_evolution.py` — clean.

`uv run mypy src/agentm/extensions/builtin/tool_propose_change.py
src/agentm/extensions/builtin/tool_query_candidates.py` — clean
(0 issues in 2 source files).

## §11 contract

`tests/unit/extensions/test_extension_contract.py` — 5 passed
(unchanged). `tool_propose_change` and `tool_query_candidates` remain
single-file, no atom-to-atom imports, no `harness.session` import,
no `core._internal` import.

## Existing test impact (back-compat)

- `test_pareto_inclusion.py` — uses pre-B-4 `parent_id` fixture
  shape; passes via `_coerce_parent_ids` reader fallback. **No edit
  required.**
- `test_per_task_evolution.py::test_end_to_end_loop_…` — asserted on
  the candidate record's exact key set. Updated `parent_id` →
  `parent_ids`, scalar `None` → `[]`. Single 7-line edit.
- `test_reflect_changespec_roundtrip.py`, all unit tests — unaffected.

The format_fix smoke (the end-to-end test that exercises the trio of
atoms — query_traces -> eval_run -> propose_change) still passes:
`test_per_task_evolution.py::test_end_to_end_loop_activates_known_good_replacement`
is green.

## Blockers

None. No `agentm.core.*` modification was required. Both task
acceptance conditions met.
