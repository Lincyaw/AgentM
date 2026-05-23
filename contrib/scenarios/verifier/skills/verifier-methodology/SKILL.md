---
confidence: fact
description: 'Verifier methodology: confirm a known fault injection materialised in the rcabench parquet data and emit the service-level propagation graph with SQL evidence per hop.'
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

You are NOT doing root-cause analysis. `get_injection_spec` returns the
root cause already. Your job is to answer two questions with evidence
drawn from the parquet data:

1. **Did the injection materialise?**
2. **What is the service-level propagation graph downstream?**

Output is the directed graph of **services** affected by the injection
plus an effectiveness verdict. The downstream consumer trusts the
graph as a propagation label, so **an edge you cannot back with SQL
is an edge you must not emit.**

## The four stages

### Stage 1 — confirm materialisation

Call `get_injection_spec` first. It returns:
- `injection_targets`: target services / pods / containers (use only
  the service name for output purposes — lower granularity is for
  diagnosis, not for the report).
- `fault_kind`: `pod_failure`, `network_delay`, `cpu_stress`, etc.
- `windows.normal_*` / `windows.abnormal_*`: unix seconds. Use these
  verbatim in every SQL `WHERE` clause.

Pull the **fault-signatures** skill section matching the `fault_kind`.
Run the diagnostic queries the signature prescribes, comparing
abnormal vs normal windows.

If the target shows **no anomaly**, set `injection_effective="false"`
with a rationale citing the comparison; emit empty
`propagation_edges`. Do not invent propagation when the injection
didn't fire.

If the signature is **partial / ambiguous** (metric moved but spans
didn't, or anomaly smaller than normal-window variance), set
`injection_effective="ambiguous"` and document what's inconsistent.

### Stage 2 — find directly-affected downstream services

A service is "directly affected" when its traces / logs / metrics in
the abnormal window show a delta vs normal AND the timing aligns with
the injection start.

Run queries that surface candidate services:
- spans whose `service_name` shows abnormal-window error rate /
  latency / throughput delta;
- log messages naming the injection target (e.g. mysql connection
  errors) — the services emitting those logs are first-hop callers.

Each first-hop downstream service → one `propagation_edges` entry
with `from_service = <injection target service>`, `to_service =
<this downstream service>`, evidence = the SQL proving the delta.

### Stage 3 — extend the graph

For each downstream service you found, ask: is there a further
downstream service whose anomaly is caused by THIS service rather
than by the injection target directly? Look for:
- traces where `to_service` is a callee of THIS service AND shows
  abnormal-window degradation;
- timing: state change later than THIS service's state change;
- a "fan-out" pattern where one degraded service drags down
  several callers.

Each such hop is another `propagation_edges` entry.

**Coincidence is not propagation.** Rule out:
- a flap that existed in the normal window too;
- a delta whose timestamp predates the injection;
- a service that has 20% errors baseline regardless of incidents.

If you can't rule out coincidence after one query, run another. If
you still can't, **leave the edge out** — false positives cost more
than false negatives.

### Stage 4 — submit

Call `submit_propagation_report` with:

- `injection_effective`: "true" | "false" | "ambiguous"
- `effectiveness_rationale`: one sentence citing the abnormal vs
  normal comparison.
- `propagation_edges`: each `{from_service, to_service, evidence[]}`
  where evidence is at least one `{sql, claim}` pair.

Service names must match the `service_name` column exactly. No
`service|` prefix, no canonicalisation. Just the bare name as the
data uses it (`mysql`, `ts-train-service`, `ts-ui-dashboard`).

## Evidence discipline

Every edge MUST have at least one `{sql, claim}` pair where:
- `sql` is re-executable against the case parquets (DuckDB);
- `claim` is one short sentence the rows back. "Service degraded" is
  not a claim. "ts-price error rate 0% → 30%, span count drops 80%
  at T+12s" is.

Edge direction is **fault-impact** (failing service → affected
service), NOT request-call direction. If `A` calls `B` and `B` is
the broken one, the edge is `B → A` (B's failure caused A's
degradation), not `A → B`.
