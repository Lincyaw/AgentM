---
confidence: fact
description: 'Verifier methodology: confirm a known fault injection materialised and emit the service-level fault-propagation graph with SQL evidence per hop.'
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

You are not doing root-cause analysis. The root cause is given —
`get_injection_spec` returns it. Verification asks two questions:

1. **Did the injection actually materialise?** That is: do the
   abnormal-window signals match what this fault kind should produce
   on this target?
2. **What is the full fault-propagation graph?** That is: every
   service whose anomaly in the abnormal window can be traced back
   to the injection target, plus every directed edge connecting
   them. The output is a graph, not a list of examples — the impact
   typically fans out through several hops (direct callers of the
   target, then their callers, and so on) and you keep extending
   until probes find no more candidates the evidence supports.

A propagation edge `from → to` is a *causal* claim: `from`'s
anomaly causes `to`'s anomaly. Two parquet rows that both look
abnormal in the same window are not, by themselves, a causal edge —
co-occurrence is symmetric, causation is not. To justify direction
you need either a timing argument (`from` shifts at-or-before `to`),
a call-graph argument (a span shows `to` calling `from` and
suffering), or a mechanistic argument grounded in what
`get_injection_spec` says the fault does.

An edge you cannot back with re-executable SQL is an edge you must
not emit. Evidence is what separates verification from speculation.

## What "downstream" means concretely

After `get_injection_spec` returns the `fault_kind`, call
`get_fault_kind_doc(fault_kind)` to read the per-kind reference:
how the injection physically works, what signals it produces, and
how it tends to propagate. Re-derive who is downstream *from that
mechanism*, not from generic heuristics. The same observable
("service X reports errors when calling service Y") admits
different causal readings depending on whether the injection is on
X, Y, the network between them, or a shared dependency — the
mechanism doc disambiguates.

## Edges around link-type faults

Some faults — the `network_*` family and the `http_*` family —
are not inside a single service's process but on a *link* between
two services. The chaos rule (tc netem, iptables, or an HTTP
proxy) is installed at exactly one of the two endpoints; that
endpoint is the rule-bearing side, and the peer at the other end
of the link is observed to be unreachable / slow / malformed from
its perspective. The injection spec records the rule-bearing side
under `injection_point.source_service` (or `app_name` for the
http_* variants); the peer is `target_service` (or
`server_address`).

For these faults, write the link-spanning edge as `rule-bearing
side → peer`. The rule-bearing side is the originator of the
fault, the peer is the part of the world that side fails to
communicate with. The cascade then continues from the
rule-bearing side outward through its callers — same as for
in-pod faults, with the link edge sitting at the head of the
chain.

## What good evidence looks like

`sql` is DuckDB SQL that runs against the case parquets and returns
rows. `claim` is one sentence the rows justify. Specific is good
("error rate 0% → 30% within the first minute of the abnormal
window"); vague is not ("service degraded"). When citing magnitudes,
cite both windows so the delta is visible.

## When to stop

You stop when you can no longer find a service whose
abnormal-window anomaly is both real and traceable to a service
already in the graph. Until then, every service you add becomes a
new pivot — its callers and the services it calls are the next
candidates to probe. Stopping at the first hop almost always
under-reports; a typical attributed fault reaches several services
through a chain.

Don't stretch the graph past what evidence supports — an
under-claimed report you can defend beats a wide one you can't —
but don't quit early either: missing the multi-hop fan-out is the
more common failure mode.
