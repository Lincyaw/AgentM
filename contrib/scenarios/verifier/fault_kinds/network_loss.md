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
side: callers show timeout-level latency (e.g. p99 jumping to 20 s)
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
   on the caller side. Mark **confirmed** with `flow_interrupted`.

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
