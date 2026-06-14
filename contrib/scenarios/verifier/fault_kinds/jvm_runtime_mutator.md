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
latency. The propagated signal on a neighbour must be an
**observable change in that neighbour's own behaviour**:

- **error rate up** on a caller that validates the response (e.g.
  HTTP 500 from a null-pointer caused by empty data, or a 4xx/5xx
  on the receiver of a mutated URL); or
- **errors / functional breakage** where the mutated value is
  consumed (a downstream that receives a corrupted request and
  rejects it, retries, or logs failures).

### Key judgment rule
A **throughput drop is not, by itself, propagation** — even for
this fault, and even when concentrated on specific endpoints.
Fewer correct requests reaching a neighbour means the *target*
stopped sending them; the neighbour itself is healthy and
idle-but-fine. That is the target's degradation, already counted,
not the neighbour's. Confirm a neighbour only when its OWN error
rate rises or its OWN behaviour breaks. If the only change you can
find on the neighbour is "fewer calls, same latency, same error
rate", reject it.

The target itself is the seed and is not re-judged here; its
aggregate latency may even drop (404/null returns fast) — that is
the fault's signature on the *target*, not evidence about a
neighbour.
