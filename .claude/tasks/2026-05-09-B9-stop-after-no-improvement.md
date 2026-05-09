# Task: B-9 — Structural stop_after_no_improvement

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §8 post-MVP, §11.9](../designs/per-task-evolution-loop.md)
**Assignee**: tdd → implementer

## Objective

After N consecutive `activate` rejections (read from `activations.jsonl`),
`tool_propose_change` refuses further `activate` calls until the tuner
either (a) waits for at least one accepted activation in another scenario
or (b) operator manually clears the counter.

Forces the tuner to change strategy (jump to a different Pareto member,
mutate a different module) rather than thrash.

## Inputs

- A-1 `activations.jsonl` (counter source — read recent entries).
- `tool_propose_change.config.promotion.stop_after_no_improvement: int`
  (already in the config_schema).

## Outputs

- `tool_propose_change.py`: counter loop reading the last K activation
  records; if last N are all `kind in {rejected, reload_failed}`, refuse
  with `cooldown_after_no_improvement`.

## Acceptance Conditions

- [ ] **Fail-stop test**: counter reads from disk (not in-memory) — restarting
  the tuner does **not** reset the counter. Otherwise the constraint is
  vacuous.
- [ ] After N rejections, the next `activate` is refused with the cooldown
  reason.
- [ ] An accepted activation resets the counter.

## Notes

- Justification for fail-stop: without persistence the tuner restarts past
  the constraint — the load-bearing property is "structural enforcement",
  not honor-system.
- Estimated diff: ~50 lines.
