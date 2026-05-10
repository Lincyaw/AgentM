# Reflection scaffold — `<TARGET_MODULE>` (scenario: `<TARGET_SCENARIO>`)

You are diagnosing a single atomic module before mutating it. The slots
below are filled by `tool_reflect`; treat the assembled text as your
in-context design brief.

## Recent failure traces

<TRACES>

## Current source of `<TARGET_MODULE>`

```python
<CURRENT_SOURCE>
```

## Recent grader feedback fingering this module

<RECENT_FEEDBACK>

## <MUTATION_INSTRUCTIONS>

1. **Diagnose, then design.** State in 1-3 sentences the root cause the
   evidence above points at. Avoid speculating about modules not in
   scope — the round-robin policy already chose this target.

2. **One concern per mutation.** Pick the single most evidence-backed
   defect. If two are tied, pick the one with broader test coverage
   (more failing tasks, not the most cosmetic).

3. **Preserve the contract.** Keep `MANIFEST.name = "<TARGET_MODULE>"`,
   keep imports stdlib-only unless the existing source already pulls
   from agentm.* internals. Do not import `agentm.harness.session` or
   `agentm.core._internal.*`.

4. **Emit a `ChangeSpec` for `propose_change`.** The exact shape:

   ```json
   {
     "kind": "atom_source",
     "path": "<TARGET_MODULE>.py",
     "new_content": "<full Python source as a string>",
     "target_atom": "<TARGET_MODULE>",
     "asi": {
       "hypothesis": "<one sentence: I think changing X will improve Y because Z>",
       "next_focus": "<what to look at if this fails>",
       "learned": "<what prior attempts taught — '' on first try>"
     }
   }
   ```

   `asi.hypothesis` is **required** in spirit (the gate doesn't reject
   missing keys, but reflection across episodes goes blind without it).
   When the prior attempt was a `discard` or `crash`, populate
   `asi.learned` with what was tried and why it failed — that's the
   load-bearing signal for failure-driven search.

   `new_content` must be the **complete** new file contents — diffs are
   not accepted. The validator under
   `contrib/extensions/changespec_validators/atom_source.py` will reject
   empty strings and missing fields.

5. **Cite evidence in the rationale.** When you eventually call
   `propose_change`, the `rationale` argument must reference at least
   one trace_id from the failure block above and one feedback line if
   any was supplied. This keeps the activations log auditable.

6. **Do not call `propose_change` yet.** First run `eval_run` to
   produce a baseline + proposed score pair under the new source via
   `atom_source_overrides`. The deployment gate (4-floor) will reject
   without that evidence regardless of how plausible the diagnosis
   sounds.
