---
confidence: fact
description: 'High-level signatures of each rcabench fault_kind: what changes on the target, how it usually propagates, and what would disqualify it. No verbatim SQL — pick the right query yourself based on the target type.'
name: fault-signatures
tags:
- verifier
- fault-kind
- diagnostics
- propagation
trigger_patterns:
- fault_kind
- pod_failure
- network_delay
- cpu_stress
- mem_stress
- http_aborted
- jvm_method_exception
type: skill
---

# Fault Signatures

Quick reference per `fault_kind`. For each kind:
- **Target signature** — what should change on the injected component
- **Propagation pattern** — typical first-hop downstream effect
- **Disqualifying signals** — what would prove this isn't the fault

Pick the right query based on whether the target is a service that
emits traces or infrastructure (DB, network, etc.). See
**verifier-methodology** for the probe strategy.

Conventions in rcabench parquets:
- Pair tables: `abnormal_*` vs `normal_*`.
- Span duration: `duration` in nanoseconds.
- Status code: `status_code` or `attr.status_code` (quote the dotted
  attribute names).
- Status HTTP: `attr.http.response.status_code`.

---

## `pod_failure` / `container_kill`

**Target signature**: span volume from the target drops sharply or
to zero; container restart counter increments; k8s pod-ready gauges
toggle. For a DB target (mysql, redis, mongo), the DB itself has no
traces — confirm via metrics (`k8s.container.ready`) and via
downstream services' DB-related spans / connection errors.

**Propagation pattern**: services that directly call the target
accumulate connection / timeout errors during the abnormal window.
The fault then fans out — callers of those callers see latency
spikes or 500s.

**Disqualifying signals**:
- No restart-count bump AND no span-volume drop on the target → set
  ``injection_effective="false"``.
- Anomaly timestamp predates `abnormal_start` → pre-existing failure.

---

## `network_delay` / `network_loss` / `network_partition` / `network_corrupt`

**Target signature**: latency on inbound spans rises (`delay`),
error rate rises (`loss` / `corrupt`), or span count drops to zero
(`partition`). All effects are scoped to the target's network
endpoint.

**Propagation pattern**: callers experience the symptom one hop
removed — delay propagates as caller-side latency, loss/corrupt as
caller-side errors, partition as caller-side timeouts.

**Disqualifying signals**:
- Target latency / error rate unchanged → injection didn't fire.
- Errors spread evenly across unrelated services → coincidence.

---

## `cpu_stress` / `mem_stress` / `jvm_heap_stress` / `jvm_thread_cpu_stress`

**Target signature**: utilization metrics on the target spike
(CPU / memory / GC pause). Target's inbound span latency rises
because a saturated process serves slowly.

**Propagation pattern**: callers see latency proportional to the
target's degradation. If stress is severe, target may OOM and
collapse into `pod_failure` pattern.

**Disqualifying signals**:
- Target utilization flat → fault didn't materialize.
- Latency rises but target utilization is normal → another
  bottleneck (lock contention, DB, downstream).

---

## `http_aborted` / `http_slow` / `http_payload_modified` / `http_response_status_modified`

**Target signature**: inbound spans on the target show modified
status codes / latency rises / response-shape anomalies depending on
the variant. Modification is **endpoint-scoped** — only spans
matching the modified URL pattern change; sibling endpoints don't.

**Propagation pattern**: callers calling the modified endpoint
surface the modification (deserialisation errors, retries, fallback
paths). Service-level propagation matches the request graph for the
affected endpoint.

**Disqualifying signals**:
- Target's spans look identical to normal → injection didn't fire.

---

## `jvm_method_exception` / `jvm_jdbc_exception` / `jvm_method_latency` / `jvm_jdbc_latency`

**Target signature**: only the spans whose `span_name` matches the
targeted method show the symptom. Sibling methods on the same
service are unchanged.

**Propagation pattern**: only callers exercising the targeted
method propagate. JDBC variants surface as failed DB-call spans.

**Disqualifying signals**:
- All methods on the target service degrade → too broad to be a
  method-level injection; the real fault may be service-wide.

---

## `dns_resolution_failed` / `dns_resolution_wrong`

**Target signature**: outbound calls from the target fail with DNS
errors / timeouts. Inbound traffic to the target is unaffected
unless the target itself needs DNS to serve.

**Propagation pattern**: outbound only.

---

## `clock_skew`

**Target signature**: often subtle — token expiry, time-window
cache misses, trace reconstruction issues. May be silent in this
dataset. Be ready to mark `injection_effective="ambiguous"`.

---

## Unknown / unmapped `fault_kind`

Fall back to fault-kind-agnostic checks:
1. Span / log volume on the target.
2. Error rate on the target.
3. Latency p95 on the target.

If none moved, set `injection_effective="false"`. Document the
agnostic approach in the rationale.
