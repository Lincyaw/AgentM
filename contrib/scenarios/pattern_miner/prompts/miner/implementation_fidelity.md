# Implementation fidelity (claimed behavior <-> actual code semantics)

Comparand: what the agent says its change does against what the code can
actually do.

Collect the agent's behavioral claims from its summaries, commit-style
messages, and test names. Then reason about the patch as code: are the new
branches reachable in the scenario they target, are the guards satisfiable
there, do the produced effects match the claimed effects? Dead code behind an
unsatisfiable guard, a handler registered on a path never taken, an effect
claimed but not produced — these are the findings.

This is pure code reasoning against the agent's own stated intent. No oracle
is involved; the contradiction is internal to the case. Note when the agent's
own tests green-light the dead path — that observation also feeds evidence
adequacy, but the root lives here if the code cannot do what is claimed
regardless of how it was tested.

Online signature guidance: the claim inventory and the reachability question
are both computable live from the trajectory and the diff. The discriminating
probe — driving the targeted scenario and observing whether the claimed
behavior appears — is available to an online critic with tools.
