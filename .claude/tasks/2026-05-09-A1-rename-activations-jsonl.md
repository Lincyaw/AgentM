# Task: A-1 — rename decisions.jsonl → activations.jsonl

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerA-forward-compat.md)
**Design**: [per-task-evolution-loop §6, §7](../designs/per-task-evolution-loop.md)
**Assignee**: implementer

## Objective

Rename the activation log written by `tool_propose_change` from
`.agentm/decisions/<scenario>/decisions.jsonl` to
`.agentm/decisions/<scenario>/activations.jsonl`. Semantic split: the file
is the *deployment log*, distinct from the Phase 2 *candidate pool*.

## Inputs

- `src/agentm/extensions/builtin/tool_propose_change.py` (lines 434–447 hold
  `_decisions_path`).
- `src/agentm/core/_internal/manifest.py` or wherever `is_constitution_path`
  globs `.agentm/decisions/**` — confirm the glob still matches the new
  filename. **Do not modify core**; if the glob is `**/*.jsonl` it covers
  the new name without changes; if it's literal `decisions.jsonl`, escalate.

## Outputs

- `tool_propose_change.py`: rename `decisions.jsonl` literal in
  `_decisions_path` and the `MANIFEST.description` string.
- `contrib/scenarios/format_fix/tuner/prompt.md` and any rca tuner prompt
  that references the old filename — grep first.

## Acceptance Conditions

- [ ] `grep -r "decisions.jsonl" src/ contrib/ .claude/designs/` shows zero
  hits in code (matches in design docs are fine — they're already updated).
- [ ] After running format_fix tuner end-to-end (per validation report),
  `.agentm/decisions/format_fix/activations.jsonl` exists and
  `decisions.jsonl` is **not** created anew.
- [ ] Constitution write-protect still rejects `tool_edit` against the new
  filename. If the glob in `core-manifest.yaml` is literal, **flag and
  escalate** rather than edit core.

## Notes

- This is a hard rename (no back-compat read of the old name). User-side
  pre-existing `decisions.jsonl` files stay readable as plain JSONL but the
  tooling stops touching them.
- Estimated diff: ~5 lines.
