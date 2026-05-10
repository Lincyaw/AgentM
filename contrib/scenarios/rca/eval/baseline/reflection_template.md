# Reflection scaffold — `<TARGET_MODULE>` (scenario: `<TARGET_SCENARIO>`)

You are evolving the **rca:baseline investigator system prompt**
(`prompts/investigator.md`). The slots below are filled by `tool_reflect`;
treat the assembled text as your in-context design brief.

## Recent failure traces

<TRACES>

## Current source of `<TARGET_MODULE>`

```markdown
<CURRENT_SOURCE>
```

## Recent grader feedback fingering this module

<RECENT_FEEDBACK>

## <MUTATION_INSTRUCTIONS>

1. **Diagnose, then design.** State in 1–3 sentences the single most
   evidence-backed gap in the current prompt. Examples we have already
   observed: agents writing SQL with unquoted `attr.*` column names and
   hitting a DuckDB Binder Error before self-correcting; agents naming
   non-existent services in `submit_final_report`; agents skipping
   `list_tables` and guessing the schema.

2. **One concern per mutation.** Pick the single defect with the
   broadest evidence base. If two are tied, pick the one most directly
   reflected in `module_feedback` from the grader — that's the credit
   the per-module attribution gives you.

3. **Preserve the agent contract block.** The bottom of
   `prompts/investigator.md` ends with the `<agent_contract>` block
   spliced in by `setup.py` from `rcabench_platform.v3.sdk.evaluation.v2`.
   Your mutation MUST keep:
     - the goal statement,
     - the available data tables list,
     - the tool inventory (`list_tables`, `query_sql`,
       `add_hypothesis` family, `read`, `submit_final_report`),
     - the termination clause that mandates `submit_final_report`.
   Add concrete guidance — short, actionable, evidence-driven. Do not
   delete sections.

4. **Emit a `ChangeSpec` for `propose_change`.** The exact shape:

   ```json
   {
     "kind": "system_prompt",
     "path": "prompts/investigator.md",
     "new_content": "<full new prompt as a string>",
     "asi": {
       "hypothesis": "<one sentence: I think changing X will improve Y because Z>",
       "next_focus": "<what to look at if this fails>",
       "learned": "<what prior attempts taught — '' on first try>"
     }
   }
   ```

   `new_content` must be the **complete** new file contents — diffs are
   not accepted. `asi.hypothesis` is required in spirit (the gate
   doesn't reject missing keys, but reflection goes blind across
   episodes without it). When the prior attempt was `discard` or
   `crash`, populate `asi.learned` with what was tried and why it
   failed.

5. **Cite evidence in the rationale.** When you call `propose_change`,
   the `rationale` argument must reference at least one trace_id from
   the failure block above and one feedback line if any was supplied.
   This keeps the activations log auditable.

6. **Do not call `propose_change` yet.** First run `eval_run` to produce
   a baseline + proposed score pair under the new prompt via
   `atom_source_overrides`. The deployment gate (4-floor) will reject
   without that evidence.
