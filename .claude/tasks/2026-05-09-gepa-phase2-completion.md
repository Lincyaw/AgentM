# Task: GEPA Phase 2 — Final Completion Report

**Date**: 2026-05-09
**Status**: COMPLETE — Layer A + Layer B (all 4 waves) shipped
**Plan**: [Layer A](../plans/2026-05-09-gepa-layerA-forward-compat.md) ·
         [Layer B](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop](../designs/per-task-evolution-loop.md) §11

## Layer B status table (10 of 10 shipped)

| ID | Wave | Title | Status | Commit | Source LoC |
|---|---|---|---|---|---:|
| B-1 | Foundation | Pareto candidate pool | ✓ | `2b2b93c` | ~250 |
| B-3 | Foundation | Generalized ChangeSpec kinds | ✓ | `f4da295` | ~510 (4 validators) |
| B-6 | Foundation | Rollout budget tracking | ✓ | `692c3e5` | ~80 |
| B-2 | Reflection | `tool_reflect` atom | ✓ | `d9d4e79` | ~398 |
| B-5 | Reflection | Module credit assignment | ✓ | `1d5a6c0` | ~192 + prompt edits |
| B-4 | Crossover | System Aware Merge | ✓ | `e59968e` | ~180 |
| B-9 | Crossover | `stop_after_no_improvement` | ✓ | `2f0b0a7` | ~50 |
| B-10 | Hardening | `baseline_fingerprint` validation | ✓ | `2222b0d` | ~80 src + 290 test |
| B-8 | Hardening | Cross-task transfer | ✓ | `14f5a1a` | ~50 src + 280 test |
| B-7 | Hardening | Production guard watch + rollback | ✓ | `d72c4a4` | ~280 src + 250 test |

## Total Python diff (Layer A + Layer B vs `9984cd2` plan commit)

| Area | Files | Insertions |
|---|---:|---:|
| `src/agentm/extensions/builtin/` | 6 (4 new) | ~2,300 |
| `contrib/extensions/changespec_validators/` | 5 (5 new) | ~510 |
| `tests/integration/` | 7 new files | ~2,500 |
| **Total** | **18** | **~5,300** |

Source-code-only (atoms + validators, excluding tests): ~2,810 LoC across
the eleven Layer-B-touched source files. Wave B-Hardening alone added
~410 source LoC + ~820 test LoC across 5 files — comfortably under the
700-LoC escalation threshold.

## Fail-stop test class roster (Phase 2 — 5 new positions)

| Position | Test file | Why load-bearing |
|---|---|---|
| Pareto inclusion correctness (B-1) | `test_pareto_inclusion.py` | Wrong → search collapses to greedy, loses GEPA's 35×-rollout property |
| Stop-counter persistence across restarts (B-9) | `test_stop_after_no_improvement.py` | Wrong → tuner escapes anti-thrash by restarting; constraint becomes honor-system |
| Merge gate noise floor (B-4) | `test_merge_gate.py` | Wrong → merge becomes a back-door past the 4-floor noise gate |
| Stale baseline rejection (B-10) | `test_baseline_fingerprint.py` | Wrong → two tuners both win the gate against each other's pre-image, silently overwriting |
| Auto-rollback evidence floor (B-7) | `test_guard_watch.py` | Wrong → outlier trace reverses every activation; loop never converges |

These are additive to the pre-Phase-2 fail-stop suite (constitution
boundary, atom-hash determinism, active-set fingerprint pairing,
catalog freeze idempotence, indexer rebuild idempotence, transactional
reload atomicity, §11 extension contract validator).

## New atoms registered (4)

All under `src/agentm/extensions/builtin/`:

- `tool_query_candidates` — Pareto frontier read API (B-1).
- `tool_query_module_feedback` — recent module-feedback distribution (B-5).
- `tool_reflect` — diagnosis + ChangeSpec proposal (B-2).
- `tool_guard_watch` — passive observer auto-rollback (B-7).

Plus four ChangeSpec validators under `contrib/extensions/changespec_validators/`:
`atom_source`, `system_prompt`, `manifest_extensions`, `manifest_field`.
(Validators are not atoms — no MANIFEST — and live under contrib
because they encode scenario-shape policy, not core mechanism.)

## Schema additions to `.agentm/decisions/<scenario>/`

| Path | Introduced | Schema |
|---|---|---|
| `activations.jsonl` (renamed from `decisions.jsonl`) | A-1 | append-only; per-record `kind` enum extended below |
| `candidates/<id>.json` | A-4 + B-1 | `{candidate_id, parent_ids, change_spec, per_task_scores, holdout_scores, eval_run_id, created_at}`; B-8 adds optional `transferred_from`, `source_eval_run_id`, `source_candidate_id` |
| `candidates/<id>.json.pruned` | B-1 | sidecar marker; presence = "strictly dominated, retained for audit" |
| `tree.jsonl` | B-1 | one record per parent→child edge; B-4 multi-parent merges write one edge per parent |
| `budget.json` | A-5 + B-6 | `{scenario, rollouts_used, usd_used, updated_at}` |

`activations.jsonl` `kind` values introduced or extended this phase:

| `kind` | Phase | Counter effect (B-9) | Notes |
|---|---|---|---|
| `activate` | pre-existing | reset | Successful single-parent deploy |
| `merge` | B-4 | reset | Multi-parent merge with `parent_ids: list[str]` |
| `exploratory` | pre-existing | reset | Gate bypass; intentional |
| `rollback` | pre-existing | reset | Manual or B-7 auto (`auto: true`) |
| `rejected` | pre-existing | +1 | Gate said no |
| `reload_failed` | pre-existing | +1 | Reload didn't land |
| `pending_human_approval` | pre-existing | reset | Tier-2 deferral |
| `stop_blocked` | B-9 | transparent (skip) | Anti-thrash audit; doesn't itself feed counter |
| `stale_baseline` | B-10 | not counted | Operator-error class; concurrent tuners can't lock each other out |
| `guard_watch_warning` | B-7 | not counted | When `auto_rollback=false`, diagnostic-only signal |

`parent_ids: list[str]` schema migration (B-4): readers
(`tool_query_candidates`) accept both legacy `parent_id: str | None`
and new `parent_ids: list[str]`. Writers always emit the new key.

## §11 atom contract status

`tests/unit/extensions/test_extension_contract.py` — all 5 tests pass.
The catalog validator agrees: every new atom (`tool_query_candidates`,
`tool_query_module_feedback`, `tool_reflect`, `tool_guard_watch`) is
single-file, no atom-to-atom imports, no `harness.session` import, no
`core._internal` import. The four validator modules under `contrib/`
are not atoms (no MANIFEST) and are exempt from §11 by construction.

## Test status (final)

```
uv run pytest --tb=short
  → 116 passed, 1 skipped, 14 deselected
```

Phase-2 contribution: 109 → 116 (+7 from Wave Hardening alone; total
Phase 2 added ~16 tests across all waves — see prior wave reports).

`uv run ruff check src/ tests/` — clean on touched files.
`uv run mypy src/agentm/extensions/builtin/tool_propose_change.py
src/agentm/extensions/builtin/tool_guard_watch.py` — clean.

## Hard constraints met

- **No `agentm.core.*` modifications** across all 10 tasks. Verified
  by `git diff 9984cd2..HEAD -- src/agentm/core/` returning empty.
- **§11 atom contract** clean (validator passes).
- **Each task's Acceptance Conditions** met. Per-task notes in the
  individual commits and prior wave reports
  (`layerB-foundation-completion.md`, `layerB-reflection-completion.md`,
  `layerB-crossover-completion.md`).
- **No remote push** performed.

## Deferred / known limitations

1. **`scenario_compose` ChangeSpec kind** — out of scope for Phase 2
   per design §11. The validator dispatch returns
   `not_yet_implemented` until the harness compose-graph reload exists.
2. **B-10 TOCTOU window** — between `git rev-parse` and the actual
   write there is a small race. The design doc flags this; full fix
   needs cross-process locking, out of scope.
3. **B-7 heuristic guard extractor** — the MVP counts only
   `tool_error_count` per turn from observability `turn.summary`
   records. Production-grade signals (refusals, content-policy hits,
   user-reported failures) are post-MVP. The atom keeps the extractor
   layout open so future work can plug in richer metrics without
   touching the algorithm.
4. **B-7 cooldown is best-effort** — if `activations.jsonl` is rotated
   or cleared, cooldown evaporates with it. Acceptable: the watcher's
   contract is "auto-rollback when evidence is overwhelming" and the
   cooldown exists to prevent oscillation, not to be cryptographically
   binding.
5. **Cross-task transfer skips score carry** (B-8 by design). A
   destination tuner must re-eval to claim a frontier slot. This is
   intentional — per-task scores aren't comparable across task classes.

## Blockers

None. All ten Phase 2 tasks landed without architectural escalation.
The design doc's Phase 2 §11 roadmap is fully realized in code.

## Commits (Wave Hardening)

```
2222b0d feat(propose_change): B-10 baseline_fingerprint validation
14f5a1a feat(propose_change): B-8 cross-task atom transfer
d72c4a4 feat(guard_watch): B-7 production-traffic auto-rollback atom
```

(See prior wave reports for B-1/B-3/B-6 Foundation, B-2/B-5
Reflection, B-4/B-9 Crossover commits.)
