# FINALIZE state

A confirmed hypothesis covers every recorded symptom. Submit the final
report via `submit_final_report`.

Required fields:

- `root_cause` — free-form statement of the root cause. Name the
  confirmed hypothesis claim and the evidence chain that supports it.
- `supporting_observations` — list of observation ids from L1 that anchor
  the claim. Reviewers will follow these back to citable facts.
- `refuted_alternatives` — list of hypothesis ids the investigation
  refuted. This is the elimination trail; without it the report has no
  process-of-elimination context.

The coverage check runs again at submission time. If any symptom is
unexplained, the tool will reject the report with the list of missing
symptom ids — record observations linking them to the confirmed
hypothesis's predictions, or open a new hypothesis to cover them.

`submit_final_report` is the only path out of this state. Calling any
other tool will not advance the trace. On acceptance, the agent loop
terminates.
