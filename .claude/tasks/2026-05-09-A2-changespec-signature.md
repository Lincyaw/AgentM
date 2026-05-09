# Task: A-2 — `target_atom: str` → `target: ChangeSpec`

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerA-forward-compat.md)
**Design**: [per-task-evolution-loop §6](../designs/per-task-evolution-loop.md)
**Assignee**: implementer

## Objective

Generalize `tool_propose_change` parameters from flat `target_atom: str` +
`new_source: str` to nested `target: ChangeSpec` per design §6. MVP accepts
**only** `kind="atom_source"`; other kinds are rejected with
`not_yet_implemented`. Hard switch — no positional adapter.

## Inputs

- `src/agentm/extensions/builtin/tool_propose_change.py` `_PARAMETERS`
  (lines 66–88) and `_execute` body.
- Design §6 `ChangeSpec` TypedDict.
- `contrib/scenarios/format_fix/tuner/prompt.md` (only existing caller).

## Outputs

- `tool_propose_change.py`:
  - New JSON schema for `target: object` with `kind`, `path`,
    `new_content`, `target_atom?` (string-or-null) properties.
  - `_execute` extracts `target.kind`; rejects with explicit
    `not_yet_implemented: <kind>` for any kind ≠ `atom_source`.
  - `target.target_atom` and `target.new_content` flow into the rest of
    the existing path (atom resolution, gate, reload).
  - `target.path` is recorded in the activation record (forward-compat for
    Layer B B-3 / B-10).
- `contrib/scenarios/format_fix/tuner/prompt.md`: update the call example
  to use the nested shape.
- (No public API removal beyond the rename — `target_atom`/`new_source` at
  top level are gone.)

## Acceptance Conditions

- [ ] `target: {kind: "atom_source", target_atom: "x", new_content: "...",
  path: "x.py"}` round-trips through the format_fix tuner.
- [ ] Calling with `target.kind = "system_prompt"` returns `is_error=True`
  with text containing `"not_yet_implemented"`.
- [ ] `activations.jsonl` records the full `change_spec` dict, not just the
  atom name (forward-compat).
- [ ] Real-LLM smoke per validation report still passes.

## Notes

- A-1 must merge first (this task writes to the new filename).
- Ship paired with the format_fix tuner prompt migration in the same commit.
- Estimated diff: ~60 lines.
