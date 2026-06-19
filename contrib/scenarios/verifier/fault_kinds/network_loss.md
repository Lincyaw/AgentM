# network_loss

## How the injection works
A `tc netem` rule on the target pod's interface drops a fraction of
packets. Small loss is masked by TCP retransmission (visible as
tail-latency); larger loss leads to connection timeouts and failed
RPCs.

## What the data should show
Inbound and outbound spans on the rule-bearing side show elevated
error rate and tail latency. Both directions of the link surface
errors — the caller logs failures against the target, and the
target logs failures against its outbound dependencies that route
through the same interface.

### When the target looks healthy but traffic dropped

Heavy packet loss can cause callers to time out before the request
reaches the target. Requests that DO arrive (surviving the loss)
complete normally — so the target's own latency and error rate stay
flat while span volume drops sharply. The signal is on the caller
side: callers show timeout-level latency (p99 jumping far beyond
the normal range, toward the system's timeout ceiling)
and the timeout blocks synchronous callers up the chain, collapsing
global throughput.

When the target itself shows reduced traffic but no degradation:

1. JOIN `parent_span_id` in the **normal** window to find which
   services call the target.
2. Check those callers in the **abnormal** window: did their p99
   latency spike to the timeout ceiling? Did call volume to the
   target drop disproportionately?
3. If caller-side latency exploded while the target's surviving
   requests are healthy → the loss is effective but the signal is
   on the caller side. Mark the affected link/caller path
   **confirmed** with `flow_interrupted`. Do not generalize this
   to unrelated downstream callees that merely receive
   proportionally fewer requests from a blocked caller.

## How to observe on a neighbour
Packet loss causes TCP retransmissions, which produce **tail-latency
spikes** (p99/max), not average-latency shifts. Most requests
succeed normally; a minority hit retransmit delays and show up only
in the tail. On a neighbour, always check p99 and max latency — an
average that looks flat can hide a p99 that spiked by 10x or more.
Break down by `span_name` if the neighbour has mixed endpoints:
only call paths that route through the lossy link are affected.

## How the failure tends to propagate
Link-type fault. Write the link edge as `rule-bearing side → peer`.
Cascade from the rule-bearing side outward through its callers: a
failed call into the rule-bearing side translates into the caller's
own anomaly only if the caller cannot mask the error.

Downstream dependencies that merely receive fewer calls from a
blocked caller should usually stay out of the final graph. Their
span-count drop is evidence that the caller path is interrupted, not
proof that the callee service is anomalous. Promote a downstream
`flow_interrupted` node only when the interrupted endpoint is itself
an alarm/user-visible path, disappears selectively, or has timeout /
error / fail-fast evidence beyond ordinary reduced demand.
