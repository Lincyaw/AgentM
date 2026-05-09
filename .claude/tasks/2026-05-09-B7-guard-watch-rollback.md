# Task: B-7 — Production guard regression watch → auto-rollback

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerB-phase2.md)
**Design**: [per-task-evolution-loop §11.7, §10 P5](../designs/per-task-evolution-loop.md)
**Assignee**: tdd → implementer

## Objective

After an activation, watch subsequent production traces for guard regression.
If the proposed atom's guard metric (e.g. `tool_error_rate`) crosses the
configured tolerance for ≥k consecutive samples, auto-emit `decision="rollback"`
to the previous Pareto-pool member.

Implementation: passive observer atom that runs at production scenario
startup, reads recent `.agentm/observability/*.jsonl` for the active
fingerprint, computes guard delta vs the activation's recorded baseline.

## Inputs

- B-1 Pareto pool (rollback target = previous member, not just previous file SHA).
- Existing `activations.jsonl` records with `evidence.gate.baseline_score`.

## Outputs

- New atom `src/agentm/extensions/builtin/tool_guard_watch.py`.
- Activation records gain a `guard_watch` field (cooldown_until_at, last
  triggered, etc.).
- Distinct decision kind `auto_rollback` in `activations.jsonl`.

## Acceptance Conditions

- [ ] **Fail-stop test**: a single outlier trace does **not** trigger
  rollback (evidence floor: ≥k samples beyond threshold). k = 3 default,
  configurable.
- [ ] 24h cooldown between consecutive auto-rollbacks of the same atom.
- [ ] Operator override flag in `tuner/manifest.yaml::tool_guard_watch.config.disabled: true`.

## Notes

- Justification for new fail-stop: without the evidence floor, transient
  noise reverses every activation — the loop never converges.
- Estimated diff: ~200 lines.
