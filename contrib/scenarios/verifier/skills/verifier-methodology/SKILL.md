---
confidence: fact
description: 'Verifier methodology: confirm a known fault injection materialised and verify/supplement a candidate service-level fault-propagation graph against the data, reasoning from the fault mechanism.'
name: verifier-methodology
tags:
- verifier
- methodology
- rca
- propagation
trigger_patterns:
- get_injection_spec
- submit_propagation_report
- propagation
type: skill
---

# Verifier Methodology

You are not deriving a graph from scratch and you are not doing
root-cause analysis. The root cause is given (`get_injection_spec`) and a
candidate propagation graph (from the dataset's existing labels) is given
in your first message. Your job: confirm the injection materialised, then
**verify and supplement** the candidate graph against the data — keep the
edges the data supports (查准), drop the ones it does not, and add the
hops it is missing (查漏).

## Reason from the fault mechanism, not from a signal checklist

Before judging any edge, build a causal model of the injection from
`get_injection_spec` + `get_fault_kind_doc`:

1. What does this fault physically do to its target? (pod-failure = the
   process is gone; cpu/mem stress = the target is starved/slow; network
   delay/loss = the link degrades; jvm fault = a specific call throws or
   stalls.)
2. What would that look like on the target itself in traces / metrics /
   logs?
3. What does it then do to whoever DEPENDS on the target, and onward to
   the user? Derive the expected downstream symptom from the mechanism —
   then go check the data for THAT symptom.

Do not memorise a list of "valid signals". A fault kind the reference
doesn't cover is reasoned out the same way: what does it break, what
would that cause. The per-kind doc (`get_fault_kind_doc`) is your
authority for how each fault propagates and what symptom to expect.

## What an edge means

`A → B` is a FAULT-IMPACT claim: A's fault causes B to degrade. This is
about CAUSE, not about who calls whom. The request call usually runs the
other way (B calls A) — expected, because a broken dependency A drags
down its caller B. So impact `A → B` normally rides on a call `B → A`;
the reversed call CONFIRMS the edge, it never refutes it.

## Judging a candidate edge A → B

Tell the causal story and check it with SQL: does A's fault,
mechanism-propagated through the A–B connection, produce B's observed
abnormal-window state? You need

- a direct call relationship between A and B (look in the NORMAL window,
  EITHER direction — they just have to be wired together);
- A is a fault source (an injection target, or already shown impacted, so
  the chain runs through it);
- B shows the abnormal-window symptom the mechanism PREDICTS (whatever it
  is — throughput collapse, latency, errors, retries).

If the predicted consequence is there → keep the edge (write the SQL). If
the data contradicts the mechanism → drop it. Co-occurrence alone (two
services both look off but never call each other) is never an edge.

## Propagation is transitive

Impact spreads as a GROWING affected frontier, not a star around the
injection target. Begin with the target as the only affected node;
confirm each direct neighbour; then treat THAT neighbour as itself
affected and look at ITS neighbours next. The `from` of an edge is
whichever already-affected service sits on the dependency side of the
hop — **not always the injection target**. A second- or third-hop `from`
is normal; rejecting an edge merely because `from` is not the injection
target is the most common mistake. Keep extending toward the user-facing
entry tier until the mechanism's symptom no longer appears.

## Effectiveness, per injection

When several faults were injected, judge each on its own signals. A weak
or non-materialised injection (e.g. a CPU stress with no observed CPU
rise and no downstream effect) did not engage and must NOT be a
propagation source — leave its edges out even if the labels include them.

## Evidence

`sql` is DuckDB that runs against the case parquets and returns rows;
`claim` is one sentence the rows justify. Cite both windows so the delta
is visible. An edge you cannot back with re-executable SQL is one you
must not emit.
