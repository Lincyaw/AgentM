# Root attribution (across dimensions)

Input: the per-dimension reports for one case. Output: which failing
dimension is the root, and which firings are its shadows.

The chain ordering supplies the rule: attribute the failure to the most
upstream mismatch in causal and temporal order — the earliest point where the
chain went wrong such that correcting it would plausibly have changed the
outcome. Chain position alone is not enough: an upstream-looking dimension
that fails late (a belief formed in one iteration poisoning the next
iteration's diagnosis) is downstream in time, and time wins.

The counterfactual is the test. For each candidate root, ask whether fixing
that dimension, with everything else left as the agent had it, changes the
result. A dimension whose correction changes nothing is a shadow, not a root.
Shadows remain valuable: they are often the observable trigger that should
route a live session to the root-dimension critique, so name the
shadow-to-root pairing explicitly.

If the case fails without any chain mismatch — the decisive requirement never
appeared anywhere the agent could see — say so rather than forcing a root;
that honesty marks the boundary of what runtime critique can reach. If the
label-validity report invalidated the measurement, the only sound conclusion
is that the case carries no attribution at all.
