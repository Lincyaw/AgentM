---
confidence: fact
description: 'Per fault_kind reference: what the injection physically does on the target and what signals it typically produces. Reference material, not procedure — the verifier reasons about propagation from these facts plus the parquet evidence.'
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

A reference card per `fault_kind`. Each entry states the physical
effect of the injection on the target and the trace / log / metric
signals it tends to produce. Use it to recognise a materialised
injection in the data; reason about who else is affected from the
mechanism plus the observed evidence.

## Conventions in rcabench parquets

- Pair tables: `abnormal_*` vs `normal_*` share the schema and let
  you compare the two windows side by side.
- Span duration is `duration`, in nanoseconds.
- Status codes appear as `status_code` or the dotted attributes
  `attr.status_code` and `attr.http.response.status_code`
  (quote dotted names).

---

## `pod_failure` / `container_kill`

Target's process is killed for the duration. Span volume from the
target drops sharply, often to zero; container restart counter
increments; pod-ready gauges toggle. For infrastructure targets
that emit no traces of their own (databases, network proxies), the
target's outage appears in callers' connection errors and in k8s
pod-level metrics rather than in the target's own span rows.

---

## `network_delay` / `network_loss` / `network_partition` / `network_corrupt`

A traffic-control rule on the target pod's interface affects packets
to and from the target. `delay` raises latency; `loss` and
`corrupt` raise error rates and retransmits; `partition` cuts the
target off entirely. The target's process is intact — its network
fabric is the bottleneck. Both the target's inbound and outbound
spans can carry the signal.

---

## `cpu_stress` / `mem_stress` / `jvm_heap_stress` / `jvm_thread_cpu_stress`

Resource pressure inside the target pod or JVM. Utilisation metrics
on the target spike (CPU, memory, GC time); inbound span latency
rises because the saturated process serves requests slowly. Severe
memory stress can OOM-kill the container and degenerate into a
pod-failure pattern.

---

## `http_aborted` / `http_slow` / `http_payload_modified` / `http_response_status_modified`

A chaos proxy at the target intercepts HTTP traffic on a matching
path/method. `aborted` returns 5xx or resets connections; `slow`
adds latency; the two `modified` variants alter response body or
status. Only spans matching the configured path/method are
affected; sibling endpoints look normal. The proxy emits these
responses from the target itself.

---

## `jvm_method_exception` / `jvm_jdbc_exception` / `jvm_method_latency` / `jvm_jdbc_latency`

A JVM agent rewrites a specific method or the JDBC path on the
target to throw or to sleep. Only spans whose name matches the
targeted method show the symptom; sibling methods on the same
service are unchanged. The JDBC variants surface as failures or
latency on the target's DB-call sub-spans, even when the database
itself is healthy.

---

## `dns_resolution_failed` / `dns_resolution_wrong`

DNS lookups inside the target pod fail or return wrong addresses.
Target's outbound calls cannot reach (or reach the wrong) peers;
inbound traffic to the target's IP is unaffected unless the target
itself needs DNS to serve.

---

## `clock_skew`

Target's system clock is shifted. Effects are app-specific —
token expiry, time-window cache misses, trace timestamp anomalies.
Often subtle or silent in observability data.

---

## Unmapped `fault_kind`

Fall back to fault-kind-agnostic checks on the target: span / log
volume, error rate, latency percentiles. Document the agnostic
reasoning in the rationale.
