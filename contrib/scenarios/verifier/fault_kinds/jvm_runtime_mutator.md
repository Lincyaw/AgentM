# jvm_runtime_mutator

## How the injection works
A JVM agent intercepts a method on the target at runtime and mutates
a value it produces or consumes. The mutation types include:

- **constant substitution** (most common): a string constant inside
  the method is replaced — e.g. an outgoing URL path gets a
  `mutated_` prefix, a config key is changed, or a lookup value is
  altered. The method itself succeeds, but the value it returns or
  sends is wrong.
- **return-value mutation**: the method's return value is replaced
  with null, an empty collection, or a wrong value.
- **field mutation**: a class field read by the method is changed.

In all cases the target service keeps running with no crash, no
exception, and often no error status — the corruption is semantic,
not structural.

## What the data should show
The target may look healthy in aggregate (no errors, similar or
lower latency). The real signal is on the **call path**:

- **constant (URL) mutation**: outgoing calls from the target to a
  specific downstream disappear or return fast errors (the mutated
  URL hits a 404 instead of the real endpoint). The downstream's
  throughput for the affected endpoint drops because correct
  requests stop arriving.
- **return-value / field mutation**: callers of the target receive
  wrong or empty data. They may fail downstream validation, skip
  processing, or return degraded results.

Because the mutated call path may return quickly (404 is fast, null
skips processing), the target's aggregate latency can actually
**drop** — this is NOT a sign of health, it is the fault's
signature.

## When it did NOT materialise
If the mutated method is never exercised during the abnormal
window AND the service's traffic is otherwise normal (similar span
count, no flow disappearance), the injection has no visible effect.

### Selective traffic vanishing can be the mutation's effect
A common pattern: the affected target path drops from substantial
normal volume to zero or near-zero in the abnormal window, yet the
service is still running (resource metrics present, CPU near-idle).
The seed agent must NOT conclude "method never invoked, mutation had
no effect" solely from that zero-traffic pattern. Instead:

1. Check whether the target service is alive (resource metrics
   present in abnormal window).
2. If alive but the affected path has zero or near-zero spans: the
   mutation may have taken effect on earlier calls, broken the
   outbound path (mutated URL → 404), and caused upstream callers to
   stop sending that specific business operation.
3. Corroborate: does the entire call chain downstream of the
   mutated method also show zero spans? (e.g. if the mutation is
   on a food-service endpoint, do food-related services system-wide
   go to zero?) If yes, the mutation caused a flow-level collapse
   — confirm with predicate `data_corrupted` or `flow_interrupted`.

This evidence must be selective to the mutated method/path. Do not
confirm from a small-count decline, a proportional whole-system
throughput drop, or healthy successful requests on the affected
endpoint. If the only signal is that the target moved with the global
traffic baseline while latency, trace status, HTTP status, logs,
metrics, callers, and downstream paths stayed healthy, reject or mark
inconclusive instead of confirming. The zero-traffic pattern is a
mutation signature only when it is path-specific and corroborated, not
when it is ordinary reduced demand.

## How to observe on a neighbour
The mutation affects a specific call path, so the signal on a
caller is concentrated on the endpoint(s) that route through the
mutated method. The caller's service-wide aggregate may look
healthy if unaffected endpoints dominate. Always break down by
`span_name` on the caller to isolate the affected path — check
both latency changes (fast 404 on the mutated URL) and error rate
changes (HTTP 4xx/5xx from the corrupted request).

## How it propagates
The cascade is driven by **data / functionality loss**, not
latency. Distinguish by call direction:

### Callers of the target
A caller's endpoint that routes through the mutated method may
vanish entirely: the mutated URL returns 404 or the corrupted
data causes fast failure, so the caller's flow stops completing.
The caller's aggregate may look healthy (other endpoints dilute
it), but the specific endpoint that depends on the target lost
its spans.

If the caller shows explicit errors (4xx/5xx, exceptions from
validating the corrupted response), confirm with
`error_rate_elevated`.

If the caller is an entrypoint or user-visible business service and
the endpoint that normally routes through the mutated method
selectively vanishes, confirm with `flow_interrupted` even when
HTTP errors are absent. For runtime mutations, a disappeared
business operation is often the corruption signal: the request path
stopped completing because the mutated value broke the downstream
operation. Corroborate by showing the normal parent-span path, the
abnormal disappearance of that specific endpoint, and sibling or
aggregate context proving this is not just a proportional
system-wide traffic drop.

Use **inconclusive** only when the vanished path is internal,
low-volume, not clearly user-visible, or cannot be tied to the
mutated method from this single edge.

### Callees of the target
Services the target calls may receive fewer or zero requests
because the mutated path stopped sending them. If the callee's
own latency and error rate are unchanged and its only signal is
"fewer calls", that is **not** the callee's degradation — the
target simply stopped calling it. Reject.

Confirm a callee only when its OWN error rate rises or its OWN
behaviour breaks (e.g. it receives a corrupted request and
rejects it).
