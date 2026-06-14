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

### Traffic vanishing IS the mutation's effect
A common pattern: the target's span count drops from hundreds to
zero in the abnormal window, yet the service is still running
(resource metrics present, CPU near-idle). The seed agent must NOT
conclude "method never invoked, mutation had no effect." Instead:

1. Check whether the target service is alive (resource metrics
   present in abnormal window).
2. If alive but zero spans: the mutation likely took effect on
   earlier calls, broke the outbound path (mutated URL → 404),
   and upstream callers stopped sending requests after discovering
   the flow is broken.
3. Corroborate: does the entire call chain downstream of the
   mutated method also show zero spans? (e.g. if the mutation is
   on a food-service endpoint, do food-related services system-wide
   go to zero?) If yes, the mutation caused a flow-level collapse
   — confirm with predicate `data_corrupted` or `flow_interrupted`.

The zero-traffic pattern is the mutation's SIGNATURE, not evidence
of a non-effective injection.

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
latency. Distinguish upstream (callers of the target) from
downstream (services the target calls):

### Upstream (callers)
A caller's endpoint that routes through the mutated method may
vanish entirely: the mutated URL returns 404 or the corrupted
data causes fast failure, so the caller's flow stops completing.
The caller's aggregate may look healthy (other endpoints dilute
it), but the specific endpoint that depends on the target lost
its spans. This IS the caller's degradation — its user-facing
flow is broken. Confirm with `flow_interrupted`.

A caller may also show explicit errors (4xx/5xx, exceptions)
if it validates the corrupted response. Confirm with
`error_rate_elevated`.

### Downstream (callees)
Services the target calls may receive fewer or zero requests
because the mutated path stopped sending them. If the
downstream's own latency and error rate are unchanged and its
only signal is "fewer calls", that is **not** the downstream's
degradation — the target simply stopped calling it. Reject.

Confirm a downstream only when its OWN error rate rises or its
OWN behaviour breaks (e.g. it receives a corrupted request and
rejects it).
