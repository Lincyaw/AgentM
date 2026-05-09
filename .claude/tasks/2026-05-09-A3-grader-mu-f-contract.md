
`tool_eval_run` aggregates `feedback_text` per task and exposes it on the
eval-run summary so Layer B `tool_reflect` can read it directly.

Keep a back-compat shim: when `grade()` returns a bare `float`, wrap as
`{score: float, dimensions: {}, feedback_text: "", module_feedback: {}}`.

## Inputs

- `src/agentm/extensions/builtin/tool_eval_run.py` `_run_single_sample`
  (line 300+) and per-task aggregation (line 148+).
- `contrib/scenarios/format_fix/eval/grader.py` (current returns float).

## Outputs

- `tool_eval_run.py`:
  - Adapter: `_normalize_grade(value) -> GradeResult` (handles bare float).
  - Per-task record adds `feedback_texts: list[str]`,
    `module_feedback_union: dict[str, str]` (keys union, values
    space-joined or last-write-wins — pick last-write, document).
  - `eval_run.summary` JSONL line gains an aggregated
    `feedback_corpus: list[{task_id, sample_idx, feedback_text}]` capped at
    e.g. 200 entries to prevent runaway growth.
- `format_fix/eval/grader.py`: opt-in upgrade — return the full dict; this
  unblocks B-5's module_feedback later.

## Acceptance Conditions

- [ ] Existing format_fix tuner run produces an `eval_run` summary with
  non-empty `feedback_corpus`.
- [ ] Bare-float graders (any third-party scenario) still work via shim.
- [ ] mypy clean on `tool_eval_run.py` with the new TypedDict.
- [ ] No core file modified.

## Notes

- This is the load-bearing forward-compat: B-2 (tool_reflect) and B-5
  (module credit) both consume this shape.
- Estimated diff: ~80 lines.
