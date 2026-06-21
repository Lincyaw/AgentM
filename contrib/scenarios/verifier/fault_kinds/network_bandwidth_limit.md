# network_bandwidth_limit

## How the injection works
A `tc` rule caps available bandwidth on the target's interface.
Small requests pass near-normal; large payloads queue and slow.

## What the data should show
Latency rises only on spans whose payloads are large enough to
saturate the cap. Small spans look normal. Throughput metrics on
the rule-bearing side flatten.

For links to backing datastores such as MySQL, the peer may not be
instrumented as a separate trace service. In that case, look on the
rule-bearing service for datastore/client spans: SQL operation span
names such as `SELECT ...`, `INSERT ...`, `UPDATE ...`, `DELETE ...`,
`ALTER ...`, repository/DAO spans, client span-kind values, and metric
attributes that name the source or destination workload. A bandwidth
cap is visible only if those datastore spans slow, time out, disappear
selectively, or cause caller-side timeout/error symptoms.

A bandwidth cap requires traffic through the capped path. If the
rule-bearing service's datastore/client spans simply vanish in the
abnormal window and there is no caller-side timeout/error, no p99/max
latency increase on the endpoint that would use the link, and no
network/throughput metric showing saturation, do not confirm the link
fault from zero traffic alone. Treat it as not visibly exercised or
inconclusive, especially when another concurrent fault can explain why
the path stopped sending requests.

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
   but the signal is on the caller side. Mark the affected link/caller
   path **confirmed** with `flow_interrupted`. Do not generalize this
   to unrelated downstream callees that merely receive proportionally
   fewer requests from a blocked caller.

## How the failure tends to propagate
Link-type fault. Edge: `rule-bearing side → peer`. Cascade reaches
callers whose own latency budget depends on big payload transfers
through the rule-bearing side; lightweight callers are usually
unaffected.

Downstream dependencies that merely receive fewer calls from a
blocked caller should usually stay out of the final graph. Their
span-count drop is evidence that the caller path is interrupted, not
proof that the callee service is anomalous. Promote a downstream
`flow_interrupted` node only when the interrupted endpoint is itself
an alarm/user-visible path, disappears selectively, or has timeout /
error / fail-fast evidence beyond ordinary reduced demand.
