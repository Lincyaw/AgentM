
`tool_eval_run` increments `rollouts_used` (one per task×sample) and
`usd_used` (sum of LLM cost from child sessions, read from each child's
trace `llm.request.end` cost_usd). `max_cost_usd` config field on
`tool_eval_run` install acts as a per-tuning-session cap; when exceeded,
the eval run aborts **between tasks** (not mid-task) and records
`aborted_due_to_budget: true` in the summary.

## Inputs

- `tool_eval_run.py` install (line 111) — accept `max_cost_usd` config.
- Child session trace path discovery — already exists via `api.cwd /
  ".agentm" / "observability"`.

## Outputs

- New helper `_load_budget(cwd, scenario)` / `_save_budget(...)` (atomic
  write via tmpfile + os.replace).
- `tool_eval_run` per-task loop reads child trace's `llm.request.end`
  records to sum cost; updates `budget.json` after each task.
- Eval-run summary gains `usd_used_in_run`, `aborted_due_to_budget` fields.

## Acceptance Conditions

- [ ] Running format_fix tuner with `max_cost_usd: 0.001` aborts after 1
  task and `budget.json::usd_used > 0`.
- [ ] Without the cap, normal run succeeds and `budget.json::usd_used`
  monotonically grows across calls.
- [ ] No race when two `tool_eval_run` calls run concurrently — last-writer-
  wins is acceptable for MVP, document in code comment. (B-6 hardens this.)

## Notes

- The `target_scenario` config from `tool_eval_run` install determines
  which scenario's budget.json to write (matches `tool_propose_change`).
- Estimated diff: ~70 lines.
