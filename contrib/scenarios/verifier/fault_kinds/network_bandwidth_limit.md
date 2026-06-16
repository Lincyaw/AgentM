# network_bandwidth_limit

## How the injection works
A `tc` rule caps available bandwidth on the target's interface.
Small requests pass near-normal; large payloads queue and slow.

## What the data should show
Latency rises only on spans whose payloads are large enough to
saturate the cap. Small spans look normal. Throughput metrics on
the rule-bearing side flatten.

### When the target looks healthy but traffic dropped

A tight bandwidth cap can cause callers to time out waiting for
responses. Requests that complete within the cap look normal at the
target — latency may even IMPROVE because only "fast" small
requests survive. The signal is on the caller side: callers show
timeout-level latency on endpoints that depend on the target.

When the target shows reduced traffic but no degradation:

1. JOIN `parent_span_id` in the **normal** window to find callers.
2. Check those callers in the **abnormal** window: did their p99
   spike to the timeout ceiling? Did their call volume to the
   target drop?
3. If caller-side latency exploded → the bandwidth cap is effective
   but the signal is on the caller side. Mark **confirmed** with
   `flow_interrupted`.

## How the failure tends to propagate
Link-type fault. Edge: `rule-bearing side → peer`. Cascade reaches
callers whose own latency budget depends on big payload transfers
through the rule-bearing side; lightweight callers are usually
unaffected.
