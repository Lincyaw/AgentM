# rca_single tuner — practitioner reference

The runtime tuner system prompt is inlined into `tuner/manifest.yaml`
(under the `system_prompt` extension). This file is the human-readable
sibling — keep them in sync.

## Quality signal for this task class

- **primary**: `grade_mean` from the programmatic grader at
  `eval/grader.py` (`0.7 * service_hit + 0.3 * fault_kind_hit`).
- **guards**: `tool_error_rate`, `turns_mean`.

A guard worth watching but not gated on:
- **fraction of traces with `Binder Error` in `query_sql` results** —
  proxy for the SQL-quoting gap. If the prompt mutation works, this
  drops; that's the canonical first-iteration win for rca_single.

## Round-robin module selection (B-5)

Before designing a mutation, call
`query_module_feedback(target_scenario="rca_single", n=20)` and bias
target choice toward the module the grader most-frequently fingered.
Today the only mutable module surfaced by the grader is `query_sql`
(via `module_feedback`) — see `eval/grader.py:_detect_sql_quoting_issue`.
The round-robin discipline matters once a second feedback channel
appears (e.g. evidence-completeness, propagation-edge correctness).

## Loop

1. `query_traces(task_class="rca_single", n=20)` — recent production
   traces. Inspect `stop_reason` and `total_turns` to spot
   no-progress loops vs honest failures.
2. `query_module_feedback(target_scenario="rca_single", n=20)` —
   identify the candidate target module.
3. `query_candidates(target_scenario="rca_single")` — review the
   Pareto pool. On iteration 1 this returns empty, which is fine.
4. `read prompts/investigator.md` to internalize the contract.
5. `reflect(failures=[...], target_module="prompts/investigator.md",
   target_scenario="rca_single")` — receive the scaffold with the
   rca_single-specific `<MUTATION_INSTRUCTIONS>` block.
6. `eval_run({})` — baseline; capture `eval_run_id`.
7. `eval_run({"atom_source_overrides":
   {"prompts/investigator.md": <new_prompt_text>}})` — proposed;
   capture `eval_run_id`.
8. `propose_change(target={"kind": "system_prompt",
   "path": "prompts/investigator.md", "new_content": <new_prompt_text>},
   rationale=..., eval_run_baseline=..., eval_run_proposed=...,
   decision="activate")`.

## Stop condition

If 3 consecutive iterations cross **no improvement** (proposed score
fails the threshold, or the noise floor rejects), conclude with a
brief summary and exit. The `tool_propose_change` atom enforces this
structurally via `stop_after_no_improvement`.

## Constraints

- The mutation MUST preserve the agent contract block at the bottom of
  `investigator.md` (spliced in by `setup.py` from
  `rcabench_platform.v3.sdk.evaluation.v2.get_agent_contract_prompt`).
  Delete it and the runtime rejects the final report.
- The mutation MUST preserve the five-tool inventory section.
- Single-concern mutations only — never multi-concern. The first
  evidence-driven win on rca_single is almost certainly adding
  schema-quoting guidance for `attr.*` columns; do not bundle other
  changes with it.
