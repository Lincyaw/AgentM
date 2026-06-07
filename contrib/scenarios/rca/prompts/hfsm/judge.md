# JUDGE state

A check has been attached. Decide which update operator to apply via
`propose_update`. Your options:

- `confirm(target_id=H)` — only when the hypothesis has BOTH (a) at least
  one satisfied negative prediction (the negative prediction was checked
  and observations did NOT trigger it) AND (b) two independent worker
  sessions producing supporting positive checks AND (c) every symptom
  links through a satisfied prediction. The gate will downgrade a confirm
  that fails any precondition to a `refine` with a reason. Read the
  reason — it names the missing piece.
- `refute(target_id=H)` — when either a negative prediction was triggered
  by observations OR a `steelman` check that tried to find supporting
  evidence failed to find any. Otherwise the gate downgrades to refine.
- `refine(target_id=H, child={...})` — when the check shows the
  hypothesis is on the right track but incomplete. Create a child
  hypothesis carrying the missing condition.
- `split(target_id=H, children=[...])` — when observations point to two
  distinct mechanisms; produce children for each.

Do not paraphrase the worker's `interpretation.proposed_update` into your
own verdict. The orchestrator re-derives from `observations` — that is
the structural anti-bias commitment. If the observations don't justify
your verdict, the gate's downgrade will say so concretely.

After the gate runs, the trace returns to VERIFY for the next prediction
or advances to FINALIZE (when a confirm covers every symptom). Available
tool: `propose_update`.
