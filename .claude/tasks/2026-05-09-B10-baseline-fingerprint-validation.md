# Task: B-10 — baseline_fingerprint validation (concurrent-tuner race)

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §10 P6, §11.10, §13](../designs/per-task-evolution-loop.md)
**Assignee**: tdd → implementer

## Objective

`tool_propose_change` accepts `baseline_fingerprint: dict | None`. At
activate time, validate that `git rev-parse HEAD -- <change_spec.path>`
equals the recorded fingerprint. If different, reject with
`stale_baseline` and require the tuner to re-eval.

Closes the §10 P6 race where two tuners on a shared atom both win a gate
against each other's pre-image.

## Inputs

- A-2 `ChangeSpec.path` (the file under contention).
- Existing git plumbing in `_write_cross_session`.

## Outputs

- `tool_propose_change.py`: new optional arg `baseline_fingerprint` in
  `_PARAMETERS` (a `{path: sha}` dict).
- Pre-activate check: `git rev-parse HEAD -- <path>` matches.
- Activation record gains `baseline_fingerprint` for audit.

## Acceptance Conditions

- [ ] **Fail-stop test**: simulate two activations on the same atom.
  First succeeds; second carries a stale baseline_fingerprint and is
  rejected with `stale_baseline`. Without this, second activation
  overwrites first silently.
- [ ] Backward-compat: omitted `baseline_fingerprint` skips the check
  (preserves single-tuner ergonomics).

## Notes

- TOCTOU window remains between rev-parse and write — documented; full
  fix needs cross-process locking, out of scope.
- Estimated diff: ~50 lines.
