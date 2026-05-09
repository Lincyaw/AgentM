# Task: B-5 — Per-module credit assignment (round-robin)

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §11.5](../designs/per-task-evolution-loop.md)
**Reference**: [GEPA summary §6.2.3](../../knowledge/summary_gepa_reflective_evolution.md)
**Assignee**: implementer

## Objective

Surface `μ_f.module_feedback` (from A-3) to the tuner via a small read-only
atom, and bias the tuner's mutation-target selection toward fingered modules.

## Inputs

- A-3 `module_feedback` populated on eval-run summaries.
- Tuner prompt files for `format_fix` and `rca`.

## Outputs

- New atom `src/agentm/extensions/builtin/tool_query_module_feedback.py`:
  takes `scenario, n` → returns recent module → feedback distribution
  (count of mentions, sample feedback texts).
- Prompt edits in `contrib/scenarios/format_fix/tuner/prompt.md` (and rca
  if present): instruct round-robin selection biased toward modules with
  highest mention count.

## Acceptance Conditions

- [ ] `tool_query_module_feedback("format_fix", n=20)` returns a non-empty
  distribution after a tuner cycle.
- [ ] §11 atom contract validator clean.

## Notes

- No fail-stop; pure read-only + prompt scaffolding.
- Estimated diff: ~100 lines.
