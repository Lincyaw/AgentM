# Task: B-2 — tool_reflect atom (separate diagnosis from mutation)

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §11.2](../designs/per-task-evolution-loop.md)
**Reference**: [GEPA summary §3.2, §6.2.4](../../knowledge/summary_gepa_reflective_evolution.md)
**Assignee**: implementer

## Objective

New tier-1 atom `tool_reflect(failures: list[TraceSummary], target_module: str,
scenario: str) → {diagnosis: str, proposed_mutation: ChangeSpec}`.

The atom reads `μ_f.feedback_text` (from A-3) verbatim, calls a reflection
LM (uses the session's StreamFn), and emits a structured diagnosis +
ChangeSpec. The mutation prompt template lives at
`contrib/scenarios/<name>/eval/reflection_template.md` (per-scenario,
itself mutable by future meta-tuning).

## Inputs

- A-3 μ_f shape on eval-run summaries.
- B-3 ChangeSpec validators (the proposed_mutation must pass them).
- `ExtensionAPI` access to a StreamFn (existing — `api.llm` or similar).

## Outputs

- New file `src/agentm/extensions/builtin/tool_reflect.py`.
- New file `contrib/scenarios/format_fix/eval/reflection_template.md`
  (template; mutable).
- `tool_reflect` install requires `target_scenario` config to locate the
  template.

## Acceptance Conditions

- [ ] Returned `proposed_mutation` validates against B-3's ChangeSpec
  validators (or returns `is_error=True` if it can't construct one).
- [ ] Template-not-found returns a clear error rather than crashing.
- [ ] No `core._internal` import; no atom-to-atom import; §11 contract
  validator clean.

## Notes

- LLM call inside an atom — first such atom. Uses the existing stream
  surface; no new core hook.
- Estimated diff: ~180 lines (atom + template).
