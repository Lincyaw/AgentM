# Plan: GEPA Layer B — Phase 2 (Pareto-pool reorganization)

**Date**: 2026-05-09
**Status**: DRAFT
**Target design**: [per-task-evolution-loop](../designs/per-task-evolution-loop.md) §11 (10-item Phase 2 roadmap)
**Methodological reference**: [GEPA summary §6 / §7](../../knowledge/summary_gepa_reflective_evolution.md)
**Builds on**: [Layer A plan](2026-05-09-gepa-layerA-forward-compat.md)

## Requirements Restatement

Phase 2 reshapes the loop from **single-incumbent hill climbing** into
**MAP-Elites illumination over a Pareto candidate pool** (per GEPA evidence:
35× fewer rollouts than GRPO, escapes local optima the four-floor gate
gets stuck on). Each item is independently shippable. Layer A is the
forward-compat scaffold; Layer B is the algorithmic content that sits on
top of it.

## Prerequisites

- Layer A merged. In particular A-1 (`activations.jsonl`), A-2
  (`ChangeSpec` signature), A-3 (`μ_f` contract), A-4 (`candidates/` dir),
  A-5 (`budget.json`) are upstream of specific Layer B items (see graph).
- No `agentm.core.*` modifications (hard constraint). Items that appear to
  require it are flagged "blocked: requires architect review".

## Implementation Phases

Each item is one task file. The graph below dictates wave structure.

### Wave B-Foundation (after Layer A — pool primitives)

- **B-1: Pareto candidate pool**
  Implements the search frontier. Adds `tree.jsonl`, switches inclusion
  criterion from "beats incumbent" to "wins on ≥1 task", deployment gate
  unchanged (the four floors still apply but to *which member is live*,
  not *whether to keep the candidate*). Adds new tier-1 atom
  `tool_query_candidates` (returns Pareto frontier).
  Files: new `src/agentm/extensions/builtin/tool_query_candidates.py`;
  edits to `tool_propose_change.py` (split inclusion vs deployment).
  Task: [task](../tasks/2026-05-09-B1-pareto-pool.md). Size: M.
  Depends on: A-1, A-4. Parallel with: B-3, B-6.
  TDD pass: yes — fail-stop "Pareto inclusion correctness" (a candidate
  best on ≥1 task must not be pruned even if globally dominated).
  Test position: new fail-stop class — *evolution-substrate inclusion correctness*.
  Justify in the task: "wrong → search collapses to greedy and loses GEPA's
  35×-rollout property".

- **B-3: Generalized `ChangeSpec` kinds**
  Lifts the MVP `kind="atom_source"` restriction. Adds validators for
  `system_prompt`, `manifest_extensions`, `manifest_field`. (Defers
  `scenario_compose` — out of scope, see §11 / §12 below.)
  Files: `tool_propose_change.py` (per-kind validator dispatch);
  helper module `contrib/extensions/changespec_validators/` (kept under
  contrib because validators are scenario-shape policy, not core).
  Task: [task](../tasks/2026-05-09-B3-changespec-kinds.md). Size: M.
  Depends on: A-2. Parallel with: B-1, B-6.
  TDD pass: yes — each validator's reject path is the fail-stop.
  Test position: §11 atom contract validator (extension of the existing
  fail-stop class).

- **B-6: Rollout budget tracking**
  Increments `rollouts_used`, `usd_used` in `budget.json` from both
  `tool_eval_run` and `tool_propose_change`. Per-tuning-session caps;
  refuses further rollouts past `rollouts_budget` / `usd_budget`.
  Files: `tool_eval_run.py`, `tool_propose_change.py`.
  Task: [task](../tasks/2026-05-09-B6-rollout-budget.md). Size: S.
  Depends on: A-5. Parallel with: B-1, B-3.
  TDD pass: pure plumbing — no fail-stop; covered by integration smoke.

### Wave B-Reflection (after B-Foundation pool exists)

- **B-2: Reflection atom (`tool_reflect`)**
  Separates diagnosis from mutation. Reads `μ_f.feedback_text` (from A-3)
  verbatim. Mutation prompt template at
  `contrib/scenarios/<name>/eval/reflection_template.md` (per-scenario,
  itself mutable). Returns `{diagnosis: str, proposed_mutation: ChangeSpec}`.
  Files: new `src/agentm/extensions/builtin/tool_reflect.py`.
  Task: [task](../tasks/2026-05-09-B2-tool-reflect.md). Size: M.
  Depends on: A-3 (μ_f shape), B-1 (so diagnosis can target a Pareto member).
  Parallel with: B-5.
  TDD pass: light — fail-stop is "ChangeSpec output validates against B-3
  validators". Otherwise pure prompt scaffolding.

- **B-5: Per-module credit assignment**
  Round-robin module-selection policy in the tuner that biases toward
  modules `μ_f.module_feedback` fingered. Lives in the *tuner prompt*
  (`contrib/scenarios/<name>/tuner/prompt.md`), not in code. Plus a
  helper atom `tool_query_module_feedback(scenario, n)` that surfaces
  the recent feedback distribution.
  Files: new `src/agentm/extensions/builtin/tool_query_module_feedback.py`;
  prompt edits to `format_fix/tuner/prompt.md` and `rca/tuner/prompt.md`.
  Task: [task](../tasks/2026-05-09-B5-module-credit-assignment.md). Size: S.
  Depends on: A-3. Parallel with: B-2.
  TDD pass: no — prompt + read-only atom.

### Wave B-Crossover (after Reflection)

- **B-4: System Aware Merge (crossover)**
  New decision channel `decision="merge"` taking `parents: list[candidate_id]`.
  Tuner reads two non-dominated candidates, calls `tool_reflect` (B-2) with
  combined per-task scores, generates a unified ChangeSpec. Same gate
  semantics as `activate`.
  Files: `tool_propose_change.py` (decision enum + merge handler).
  Task: [task](../tasks/2026-05-09-B4-system-aware-merge.md). Size: M.
  Depends on: B-1 (Pareto pool), B-2 (reflection). Parallel with: B-9.
  TDD pass: yes — fail-stop "merge child must be evaluated under full
  4-floor deployment gate" (otherwise merge becomes a back-door past the
  noise floor).

- **B-9: Structural `stop_after_no_improvement`**
  Wrap `tool_propose_change` with a counter that refuses further `activate`
  after N consecutive rejections, forcing strategy change or escalation.
  Files: `tool_propose_change.py`.
  Task: [task](../tasks/2026-05-09-B9-stop-after-no-improvement.md). Size: S.
  Depends on: A-1 (`activations.jsonl` is where the counter reads from).
  Parallel with: B-4.
  TDD pass: yes — counter must persist across tuner sessions (read from
  `activations.jsonl`, not in-memory). Fail-stop: "tuner cannot reset
  counter by restarting".

### Wave B-Hardening (last; production / safety)

- **B-7: Production-traffic guard regression watch → auto-rollback**
  After an activation, watch subsequent production traces; if a guard
  metric regresses beyond tolerance, auto-emit `decision="rollback"`.
  Files: new `src/agentm/extensions/builtin/tool_guard_watch.py`
  (a periodic-check atom, OR a passive observer that reacts on each
  production session start). Decision: passive — runs at production
  scenario startup, reads recent traces.
  Task: [task](../tasks/2026-05-09-B7-guard-watch-rollback.md). Size: M.
  Depends on: B-1 (rollback target = previous Pareto member, not just
  previous file SHA). Parallel with: B-8, B-10.
  TDD pass: yes — fail-stop "auto-rollback requires evidence (≥k samples
  beyond regression threshold)" — without this an outlier triggers spurious
  rollback. Test position: new fail-stop "auto-rollback evidence floor".

- **B-8: Cross-task transfer**
  When `tool_read` evolves under RCA, surface as a candidate in `plan_mode`'s
  pool with parent pointer. Eval there decides if it deploys.
  Files: `tool_propose_change.py` (cross-scenario candidate write); new
  optional config knob `tuner/manifest.yaml::tool_propose_change.config.transfer_to: list[str]`.
  Task: [task](../tasks/2026-05-09-B8-cross-task-transfer.md). Size: S.
  Depends on: B-1. Parallel with: B-7, B-10.
  TDD pass: no — opt-in feature, integration test only.

- **B-10: `baseline_fingerprint` validation (concurrent-tuner race fix)**
  Closes §10 P6. `tool_propose_change` accepts `baseline_fingerprint`,
  validates `git rev-parse HEAD -- <atom_file>` is unchanged at activate
  time; rejects with `stale_baseline` otherwise.
  Files: `tool_propose_change.py`.
  Task: [task](../tasks/2026-05-09-B10-baseline-fingerprint-validation.md). Size: S.
  Depends on: A-2 (uses `ChangeSpec.path`). Parallel with: B-7, B-8.
  TDD pass: yes — fail-stop "stale baseline cannot activate" (otherwise
  two tuners can both win the gate against each other's pre-image).

## Dependency Graph

