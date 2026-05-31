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
window, or the mutated value is never consumed by a code path that
matters, the injection has no visible effect.

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
