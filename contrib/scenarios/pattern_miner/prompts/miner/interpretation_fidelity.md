# Interpretation fidelity (task <-> understanding)

Comparand: the task text against the reading of it that the agent's actions
imply.

Re-derive the requirements from the task description alone, before looking at
what the agent did: constraints, directions, orderings, literal patterns,
scope boundaries, and negative requirements. Then reconstruct, from the
trajectory and the final patch, the reading the agent actually worked under.
The finding is a divergence between the two that the agent held consistently:
a faithful implementation of a wrong reading.

Do not report weak validation here; that belongs to evidence adequacy. A
misreading typically propagates: tests authored in-session encode the same
wrong reading, so downstream checks look healthy. That downstream health is
part of the evidence that the root is here.

Online signature guidance: the live monitor has the task text but no oracle,
so signatures for this dimension are states in which a blind re-derivation of
the task is cheap and worth triggering — direction-sensitive or
order-sensitive task language, and gating oracles that are all self-authored.
