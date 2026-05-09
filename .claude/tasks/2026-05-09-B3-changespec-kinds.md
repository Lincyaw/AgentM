# Task: B-3 — Generalized ChangeSpec kinds (system_prompt, manifest_*)

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §6, §11.3](../designs/per-task-evolution-loop.md)
**Assignee**: tdd → implementer

## Objective

Lift the MVP `kind="atom_source"` restriction. Add validators for:

- `system_prompt`: replace `system_prompt` text in scenario manifest.
  Validator: target file is `contrib/scenarios/<name>/manifest.yaml` or a
  `!include`'d `.md`; new content is non-empty UTF-8 ≤ N chars.
- `manifest_field`: edit a scalar field in the manifest. Validator: path
  is a YAML dotted path; field exists; new value is the right scalar type.
- `manifest_extensions`: add/remove an atom from the extensions list.
  Validator: round-trips through the existing scenario loader's validation;
  reject if loader rejects.

`scenario_compose` is **explicitly out of scope** (flag as
`not_yet_implemented` with a comment pointing at this task's "Notes").

## Inputs

- `tool_propose_change.py` post-A-2.
- Existing scenario loader at `src/agentm/extensions/loader.py` (read-only
  reuse).

## Outputs

- New directory `contrib/extensions/changespec_validators/`:
  - `atom_source.py`, `system_prompt.py`, `manifest_field.py`,
    `manifest_extensions.py`. Each exports `validate(change_spec, cwd) ->
    {ok: bool, error: str|None}`.
- `tool_propose_change.py`: dispatch table `kind → validator`.
- `tool_propose_change.py`: per-kind `_apply_change(...)` (atom_source
  reuses existing path; manifest kinds write through the same write-via-
  ResourceWriter / git-commit mechanism).

## Acceptance Conditions

- [ ] **Fail-stop tests** (per kind, reject path):
  - `system_prompt` with empty `new_content` rejected.
  - `manifest_field` targeting non-existent field rejected.
  - `manifest_extensions` adding an atom that loader rejects → propose
    rejected with loader's error text surfaced.
- [ ] `system_prompt` happy-path: write goes to manifest, next session
  loads the new prompt (E2E with format_fix).
- [ ] §11 atom contract validator unaffected (validators live under
  contrib/extensions/, not under builtin/).

## Notes

- `manifest_extensions` is the riskiest. If the loader-reuse strategy
  doesn't cleanly fit, descope this kind and ship `system_prompt` +
  `manifest_field` only; B-3 then gets a follow-up task.
- `scenario_compose` deferred; needs harness compose-graph reload — out of
  scope per Layer B plan.
- Estimated diff: ~300 lines across 4 validator files + dispatch.
