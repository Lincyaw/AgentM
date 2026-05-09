# Task: B-6 — Rollout budget tracking (harden A-5)

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §11.6](../designs/per-task-evolution-loop.md)
**Assignee**: implementer

## Objective

Harden A-5's `budget.json` with proper per-tuning-session caps. Both
`tool_eval_run` (rollouts incremented per task×sample) and
`tool_propose_change` (counts each call) update the file under file-lock
to remove A-5's "last writer wins" footnote.

Add `rollouts_budget` config knob on tuner manifests; refuse further
operations when exceeded.

## Inputs

- A-5 budget.json writer.

## Outputs

- `tool_eval_run.py` and `tool_propose_change.py`: shared helper module
  `contrib/extensions/budget_lock.py` providing
  `with_budget_lock(scenario, fn)` using `fcntl` (POSIX) on a sidecar
  `budget.json.lock` file.
- New config field `rollouts_budget: int | null` accepted on both atoms'
  install configs.
- Aborts return `{"ok": false, "reason": "budget_exhausted", ...}`.

## Acceptance Conditions

- [ ] Two concurrent `tool_eval_run` calls produce monotonically-increasing
  `rollouts_used` with no lost updates (test by spawning two processes).
- [ ] `rollouts_budget=0` causes the next call to abort.

## Notes

- File-lock is good enough for single-host MVP; cross-host concurrency is
  out of scope.
- Estimated diff: ~80 lines.
