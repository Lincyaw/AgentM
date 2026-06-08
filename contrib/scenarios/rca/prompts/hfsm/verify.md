# VERIFY state

A hypothesis is open. Your task is to **try to refute its predictions**,
not to confirm them. The information-gain scheduler has already chosen
the next prediction to attack (its id is suggested above).

For each prediction:

1. Dispatch a worker session (or run the appropriate query yourself if
   the prediction is cheap) with a **falsification framing**: "find a
   piece of evidence that contradicts this prediction." The brief builder
   produces these for you on dispatch — do not rewrite it to ask for
   verification, the wording is structural.
2. The worker returns `observations` (facts, ingested into L1) and
   `interpretation` (advisory, recorded in the trace but ignored by the
   gate). The orchestrator re-derives the verdict from observations
   alone. Do not paraphrase the interpretation into your own verdict.
3. Attach the verification result with `attach_check`. State the
   `verdict_proposal` in plain English. The gate looks for the words
   `triggered` / `support(s|ed)` / `steelman` to classify the outcome —
   say what actually happened, do not coin new vocabulary.

Independence rule: a `confirm` later in JUDGE requires two distinct
`worker_session_id`s on supporting positive checks. Re-using the same
worker session for both positive checks will be downgraded.

Once a check is attached, the trace advances to JUDGE. Available tools:
`attach_check`, `record_observation`, `dispatch_agent`, `query_sql`,
`list_tables` (cheap predictions you can probe yourself — but two
positive checks for `confirm` later still need distinct
`worker_session_id`s, so `dispatch_agent` is required for at least one
of them).
