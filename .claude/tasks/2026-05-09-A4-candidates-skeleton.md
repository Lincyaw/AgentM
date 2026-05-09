# Task: A-4 — Reserve candidates/ directory and write degenerate entries

**Date**: 2026-05-09
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-09-gepa-layerA-forward-compat.md)
**Design**: [per-task-evolution-loop §7, §11.1](../designs/per-task-evolution-loop.md)
**Assignee**: implementer

## Objective

On every successful `tool_propose_change` activation, also write a
`candidates/<candidate_id>.json` file alongside the `activations.jsonl`
entry. MVP writes a single degenerate pool member per activation (parent_id
= previous activation's candidate_id, or null for the first). This locks
the schema in so B-1 (Pareto pool) is a non-breaking extension.

## Inputs

- `tool_propose_change.py` after the activation succeeds (line ~280).
- Eval run summary JSONL — read `per_task` records to populate
  `per_task_scores`.

## Outputs

- New helper `_write_candidate_record(decisions_dir, record)` in
  `tool_propose_change.py`.
- Schema (per design §11.1):
  ```json
  {
    "candidate_id": "c_<uuid12>",
    "parent_id": "c_<...>" or null,
    "change_spec": {...},                  // the ChangeSpec verbatim
    "per_task_scores": {"task_id": float, ...},
    "holdout_scores": {"task_id": float, ...},
    "eval_run_id": "er_...",
    "created_at": <epoch>
  }
  ```
- `activations.jsonl` entry gains a `candidate_id` field pointing at the
  written file.

## Acceptance Conditions

- [ ] After format_fix tuner run, `.agentm/decisions/format_fix/candidates/`
  contains one `<candidate_id>.json` per activation.
- [ ] `parent_id` chains correctly across consecutive activations.
- [ ] Schema fields exactly match design §11.1 — no extra fields, no missing
  ones.
- [ ] `.agentm/decisions/<scenario>/candidates/**` is constitution-protected
  (same `**` glob covers it; verify by attempting a `tool_write` to the
  path and confirming rejection).

## Notes

- Forward-compat only — B-1 generalizes the inclusion logic but reads the
  same files.
- Estimated diff: ~50 lines.
