# Task: B-8 — Cross-task atom transfer

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §11.8, §13](../designs/per-task-evolution-loop.md)
**Assignee**: implementer

## Objective

When `tool_read` evolves under RCA's tuner, surface the change as a
candidate in `plan_mode`'s candidate pool with a `parent_id` pointer to
RCA's pool. Eval there decides if it deploys (deployment gate independent).

## Inputs

- B-1 Pareto pool.
- New config knob on `tool_propose_change` install:
  `transfer_to: list[str]` (list of sibling scenario names).

## Outputs

- `tool_propose_change.py`: when `transfer_to` is set, after writing the
  primary candidate, write a sibling candidate under each target scenario's
  `candidates/` with `parent_id` pointing across scenarios.
- Sibling candidate is **inactive** by default — the sibling's tuner / next
  eval run decides.

## Acceptance Conditions

- [ ] Format_fix tuner with `transfer_to: ["other_scenario"]` writes a
  candidate to `.agentm/decisions/other_scenario/candidates/`.
- [ ] Cross-scenario `parent_id` is preserved in the sibling's tree.jsonl.
- [ ] No deployment occurs in the sibling without its own gate passing.

## Notes

- Opt-in feature, integration test only.
- Estimated diff: ~80 lines.
