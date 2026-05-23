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

You are NOT doing root-cause analysis. `get_injection_spec` returns
the root cause. Your job is:

1. **Did the injection materialise?** Compare the target's behaviour
   in the abnormal window vs the normal window.
2. **What is the service-level propagation graph downstream?**

Output is the directed graph of services affected by the injection,
plus an effectiveness verdict. **An edge you cannot back with SQL is
an edge you must not emit.**

## Stage 1 — confirm the injection

Pull the **fault-signatures** skill for the `fault_kind` returned by
`get_injection_spec`. Run queries that compare abnormal vs normal on
the kinds of signals that fault-kind would change.

- If the target is a **microservice that emits traces** (`ts-*`
  services in train-ticket), expect span volume / error rate /
  latency changes on `service_name = <target>`.
- If the target is **infrastructure that doesn't emit traces of its
  own** (`mysql`, `redis`, `mongo`, network proxies), the target
  doesn't appear in `abnormal_traces.service_name` directly. Look at
  - downstream services' DB / network spans (`db.system`,
    `net.peer.name`, span name patterns) — these are how the
    infrastructure shows up in traces;
  - logs of services that depend on it;
  - k8s-level metrics on the target's pod / container if exported
    (`k8s.container.ready`, `kube_pod_status_phase`, restart count).

If the target shows no anomaly, set `injection_effective="false"`
with rationale citing the comparison and emit empty
`propagation_edges`.

If the signature is partial / ambiguous (one signal moved, others
didn't), set `injection_effective="ambiguous"` and document.

## Stage 2 — find direct-downstream services

A service is "directly affected" when its abnormal-window behaviour
differs from normal AND the change is consistent with the injection
target being the cause. **Don't guess service names — probe the
data**. Pick the strategy that matches the target type:

- **Target emits traces** → walk the trace topology. Use
  `parent_span_id` to find which services CALLED the target during
  the abnormal window. Those services may be affected if their spans
  show errors / latency rise / volume drop while the target is dead.
- **Target is infrastructure** → query for the downstream services
  that use this kind of infrastructure. For a DB target like mysql,
  look at spans with DB-related attributes (`db.system='mysql'`,
  `net.peer.name LIKE '%mysql%'`, span names like `SELECT`/`INSERT`).
  For a network target, look at spans whose remote-peer matches the
  isolated endpoint. Cross-reference with logs that mention the
  target or generic connectivity errors.
- **Fallback** — services with a sharp abnormal-window cliff
  (throughput / error-rate / latency delta vs normal) are
  candidates regardless of target type. Filter out services with
  zero or unstable baselines.

Each first-hop service → one `propagation_edges` entry with
`from_service = <target>`, `to_service = <candidate>`, evidence =
the SQL+claim showing the abnormal delta and a timing argument.

## Stage 3 — extend the graph

For each first-hop service, repeat the same probes scoped to that
service as the new "target". A service that degraded because of a
first-hop service (not because of the original target directly) is a
second-hop. Add edges from the first-hop service to its downstream.

Stop when probes return no new candidates or when adding more would
require speculation. **Don't extend the graph speculatively.**

## Evidence discipline

Each edge MUST carry at least one `{sql, claim}` pair:
- `sql` is re-executable DuckDB SQL against the case parquets.
- `claim` is one short sentence the rows back. Specific is good
  ("error rate 0% → 30% on T+12s"); vague is not ("service was
  affected").

Edge direction is **fault-impact** (failing service →
caller-that-degraded), not the request-call direction. If A calls
B and B is the broken one, the edge is `B → A`.

## Rule out coincidence

Before emitting an edge, ask:
- Did the candidate's anomaly start AT or AFTER the injection?
- Does the candidate have the same anomaly in the normal window
  too (baseline noise)?
- Could a different upstream service explain it?

If you can't rule out coincidence, run one more query. If still
unclear, leave the edge out.
