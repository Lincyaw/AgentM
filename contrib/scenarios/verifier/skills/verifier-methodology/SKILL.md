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
would that cause.

The per-kind doc (`get_fault_kind_doc`) is a REFERENCE to inform that
reasoning — the TYPICAL way the fault manifests — not a rule to satisfy.
The DATA is the authority. A fault can manifest differently than the doc
describes (e.g. a pod-kill that disrupts via a latency spike rather than
the textbook zero-span collapse); when the data and the doc disagree,
trust the data and your own causal reasoning, not the doc's canonical
signature. Use the doc to form expectations, then judge what actually
happened.

## What an edge means

`A → B` is a FAULT-IMPACT claim: A's fault causes B to degrade. This is
about CAUSE, not about who calls whom. The request call usually runs the
other way (B calls A) — expected, because a broken dependency A drags
down its caller B. So impact `A → B` normally rides on a call `B → A`;
the reversed call CONFIRMS the edge, it never refutes it.

## Judging a candidate edge A → B — reason, don't pattern-match

Tell the causal story from the mechanism, then ask whether the data
actually bears it out. Two things must hold:

- A and B are DIRECTLY connected — a call (NORMAL window, either
  direction) or a shared k8s deployment/node. With no physical path there
  is no edge.
- A is a fault source (the injection target, or already shown dragged
  down so the chain runs through it), AND B's abnormal-window behaviour,
  read AS A WHOLE against its own baseline, is what "A's fault dragging B
  down" would actually look like.

That second judgement is not a checklist — do NOT scan for any single
metric that moved and call it a hit. Look at B's whole picture together
(latency, errors, throughput, where B sits relative to A on the call
path, the timing) and decide: does it cohere into "B was dragged down by
A", or is B just behaving normally — or even better — under a different
overall load? One dipping number inside an otherwise unchanged-or-improved
picture is not degradation; it is noise or the environment. If the story
and the data genuinely agree, keep the edge and write the SQL; if the
data does not show B actually suffering, drop it, even if the label
asserts it.

Two things to reason past (not rules to memorise):
- the abnormal window can carry different total load than the normal one,
  so a change B shares with the whole system is the environment, not your
  fault. Separate fault-induced degradation from system-wide drift by
  asking how B moved relative to its baseline AND relative to how the
  rest of the system moved.
- co-occurrence — two services both looking off but never touching — is
  never an edge.

## Propagation spreads only as far as the damage actually reaches

Impact is a growing frontier, not a star: the target is dragged down,
then whoever depends on it, then their dependents. The `from` of a hop is
whichever already-dragged-down service sits on the dependency side —
**not always the injection target** (rejecting an edge merely because
`from` is not the target is a classic mistake).

Follow the damage outward ONLY as far as it genuinely reaches. At each
hop ask whether the next service is actually suffering because this one
is; when the answer becomes no, the frontier ends there — that is the
real blast radius. Reaching the user-facing entry tier is an OUTCOME, not
a goal: some faults ride all the way to the entry, others attenuate
before it (an outage often leaves the entry merely serving fewer or
faster requests — which is not degradation). If your graph stops short of
the entry, decide honestly which case you are in — you have not finished
tracing the real failures, or the impact truly dies out here — and never
invent a node with no real degradation just to reach further.

## Effectiveness — one verdict per injection

A case may carry SEVERAL injections; they can combine and cascade. There
is no single "effective" verdict — submit one entry per injection in
`injections`, judging each on its own merits: from the mechanism, what
should THIS injection have done to its own target, and does the target's
data bear that out? All, some, or none may have engaged. An injection
whose target shows no genuine sign of the fault did not engage and must
NOT be used as a propagation source — leave its edges out even if the
labels include them. So the degraded-service count, and the evidence
behind it, ranges from zero (nothing engaged) to many.

## Everything is a query — points and edges

The graph you submit is built only from queryable facts. Two SQL-backed
pieces, both re-executed after you submit (each must run and return rows):

- a **node** (`propagation_nodes`) is a service you have JUDGED to be
  genuinely dragged down by the fault (per "Judging a candidate edge").
  It carries `symptom_evidence` — one or more SQLs, each returning the
  NORMAL and ABNORMAL window side by side (e.g. `... WHERE <normal> UNION
  ALL ... WHERE <abnormal>`) so the delta is visible. Prefer DIVERSE
  signals (trace latency / throughput / errors, the metrics tables, logs)
  — one fault often shows in several, and corroboration across signals is
  stronger than one number. The rows must bear out the judgement — the
  whole picture, the way the mechanism predicts, not one cherry-picked
  metric. A service whose overall behaviour is unchanged or improved is
  not a node, however much one number you could point at dipped.
- an **edge** (`propagation_edges`) is a directed hop `from → to` between
  two nodes. Its `relationship_sql` proves the two services are DIRECTLY
  connected — a trace parent/child call (either direction; look in the
  normal window) or a shared k8s deployment/node. This proves the edge
  can physically carry impact.

Both endpoints of every edge MUST also appear in `propagation_nodes`
(both must be independently proven symptomatic) — submission is rejected
otherwise. So an edge means: from is symptomatic (its node SQL), to is
symptomatic (its node SQL), and the two are connected (the edge SQL). The
fault-impact direction (`from`'s failure drags down `to`) rides on the
reverse call, exactly as in "What an edge means" above.

If you cannot produce passing `symptom_evidence` for a service, it is not
a node and cannot be in any edge. If you cannot produce a passing
`relationship_sql`, there is no edge. Co-occurrence, or a uniform
throughput drop shared by unrelated services, proves neither.
