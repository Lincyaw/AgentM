# Task: B-4 — System Aware Merge (decision="merge")

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §11.4](../designs/per-task-evolution-loop.md)
**Reference**: [GEPA summary §6.2.6](../../knowledge/summary_gepa_reflective_evolution.md)
**Assignee**: tdd → implementer

## Objective

Add `decision="merge"` channel to `tool_propose_change`. Takes
`parents: list[candidate_id]` (≥2). The atom:

1. Loads the parents from the candidate pool.
2. Calls `tool_reflect` (B-2) with combined per-task scores and feedback.
3. Receives a unified `ChangeSpec` from reflect.
4. Subjects the merge child to the **full four-floor deployment gate**
   (same as `activate`). Inclusion in the pool follows B-1's "wins on ≥1
   task" criterion as usual.

## Inputs

- B-1 Pareto pool.
- B-2 `tool_reflect`.
- `tool_propose_change.py` decision dispatch.

## Outputs

- `tool_propose_change.py`:
  - `decision` enum gains `"merge"`.
  - `parents: list[str]` accepted in args.
  - New handler `_handle_merge(...)` orchestrating parents → reflect →
    eval → gate.
- Activation record gains `parents: [...]` field for merge entries.

## Acceptance Conditions

- [ ] **Fail-stop test**: a merge child that fails the noise floor is **not**
  activated even though its parents both passed independently. (Otherwise
  merge becomes a back-door past the gate.)
- [ ] Merge child is added to the candidate pool regardless of deployment
  outcome (search frontier includes it).
- [ ] Activation record cleanly distinguishes merge from activate via
  `kind: "merge"` and the `parents` field.

## Notes

- Estimated diff: ~120 lines.
