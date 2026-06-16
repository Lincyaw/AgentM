# network_partition

## How the injection works
An `iptables` rule drops all traffic to and from the target pod (or
between the rule-bearing side and a named peer). The target's
process is alive but cannot reach the network in either direction
across the partitioned link.

## What the data should show
Outbound spans from the rule-bearing side fail with connection
errors against the peer; the peer never receives requests on that
path. Both sides log connection-refused / no-route errors. The
target's own healthchecks may still pass (its loopback is fine).

### When the target looks healthy but traffic dropped

A partition severs a specific link, not the target process. If the
target has other callers whose traffic does NOT cross the partitioned
link, those requests arrive and complete normally — so the target's
own latency and error rate stay flat while its span volume drops.

**This does not mean the injection failed.** The signal is on the
caller side of the severed link: callers that used to reach the
target through the partitioned path now time out (typically 20 s).
The timeout blocks synchronous callers, which blocks THEIR callers,
cascading up to the load generator and collapsing overall throughput.

When the target itself shows reduced traffic but no degradation:

1. JOIN `parent_span_id` in the **normal** window to find which
   services call the target and on which endpoints.
2. Check those callers in the **abnormal** window: did their calls
   to the target vanish? Did their latency on those endpoints spike
   toward the timeout ceiling (e.g. 20 s)?
3. If caller-side calls vanished or hit timeout while the target's
   surviving requests are healthy → the partition is effective but
   the signal is on the caller side. Mark **confirmed** with
   `flow_interrupted`.

## How the failure tends to propagate
Link-type fault. The edge `rule-bearing side → peer` is the
link-spanning hop. From the rule-bearing side, any further callers
that depended on it completing work that required the peer also
become anomalous — cascade up the call-graph from the rule-bearing
side.
