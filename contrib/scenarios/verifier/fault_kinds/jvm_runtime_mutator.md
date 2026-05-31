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
latency. Unlike PodFailure or CPUStress where the signal is
latency explosion or throughput collapse, here the signal is
subtler:

- throughput drop concentrated on **specific endpoints** that
  depend on the mutated call path (not a uniform system-wide
  drop — that would be load drift);
- error rate increase on callers that validate the response
  (e.g. HTTP 500 from a null-pointer caused by empty data);
- OR, no errors but functional degradation: fewer completed
  business transactions, missing data in responses.

### Key judgment rule
A throughput drop on a service that directly calls the target (or
depends on data the target provides), concentrated on the specific
endpoints that exercise the mutated method, IS propagation — even
if latency and error rate are unchanged. This is the one fault type
where throughput-only drop on a related call path counts, because
the mechanism is data corruption / path breakage, not overload.

Distinguish from PodFailure's system-wide throughput reduction:
check whether the drop is specific to endpoints that route through
the target vs uniform across all services.
