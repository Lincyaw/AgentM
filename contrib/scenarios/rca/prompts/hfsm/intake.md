# INTAKE state

You are starting a new root-cause investigation. Your job in this state is
to **record the observed symptoms** as they are reported to you — nothing
else. Do not propose hypotheses yet, do not run queries to "see what's
going on", do not summarise.

A symptom is a concrete, observed problem statement: a metric crossing a
threshold, an alert firing, a user-visible failure, a degraded latency
measurement. One `record_symptom` call per distinct symptom. Free-form
text is fine — the structural enforcement happens later, not here.

When you have recorded every reported symptom, the trace will advance to
OBSERVE on its own. You do not need to declare you are done; the next
state begins as soon as L1 has at least one symptom.

Available tool in this state: `record_symptom`. You may also record
`record_observation` if a citable fact has already been handed to you,
but symptoms are the priority — they are what the investigation will be
held accountable to covering.
