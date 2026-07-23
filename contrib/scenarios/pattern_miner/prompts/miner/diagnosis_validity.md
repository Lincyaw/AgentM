# Diagnosis validity (claimed cause <-> collected evidence)

Comparand: the agent's causal explanation of the problem against the evidence
its own trajectory collected.

Reconstruct the diagnosis the agent committed to — what it believed the
defect or mechanism was — and the observations it had gathered by that point.
Ask whether the evidence entails the diagnosis: would the same observations
be produced under a rival explanation the agent never excluded? A diagnosis
adopted on consistent-but-not-discriminating evidence is the first form of
this dimension.

The second form sits at the exit: does the change actually address the
diagnosed cause, or only the symptom through which the cause was observed? A
correct diagnosis followed by a symptom-level patch is a finding here, not in
implementation fidelity — the code does what the agent decided, but the
decision abandoned its own diagnosis.

Online signature guidance: both forms are computable from the trajectory
alone — the evidence-to-diagnosis entailment is a reading of turns the
monitor already has, and the diagnosis-to-change relation is a reading of the
diff against the agent's stated cause.
