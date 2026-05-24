# network_bandwidth_limit

## How the injection works
A `tc` rule caps available bandwidth on the target's interface.
Small requests pass near-normal; large payloads queue and slow.

## What the data should show
Latency rises only on spans whose payloads are large enough to
saturate the cap. Small spans look normal. Throughput metrics on
the rule-bearing side flatten.

## How the failure tends to propagate
Link-type fault. Edge: `rule-bearing side → peer`. Cascade reaches
callers whose own latency budget depends on big payload transfers
through the rule-bearing side; lightweight callers are usually
unaffected.
