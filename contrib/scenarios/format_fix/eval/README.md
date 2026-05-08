# format_fix eval suite

Deterministic eval set for the `format_fix` scenario. Each task in
`tasks/` is a YAML record with:

- `id`: stable identifier (filename stem by default)
- `task_class: format_fix`
- `input.user_message`: the malformed JSON the agent receives
- `expected.value`: the canonical Python-dict / list result
- `holdout: true` (optional): excluded from the MVP eval; only run by
  `tool_eval_run(holdout_only=true)`

`grader.py` parses the agent's final assistant text as JSON (tolerating
markdown fences and surrounding commentary) and compares deep-equal to
`expected.value`. Score is 1.0 on match, 0.0 otherwise.

The atom under evolution (`tool_normalize_json`) v1 is deliberately weak
— the tuner's job is to evolve it into something that handles tasks
02-05 (nested objects, unicode, numeric coercion, arrays).
