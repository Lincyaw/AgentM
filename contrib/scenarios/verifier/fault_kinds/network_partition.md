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

## How the failure tends to propagate
Link-type fault. The edge `rule-bearing side → peer` is the
link-spanning hop. From the rule-bearing side, any further callers
that depended on it completing work that required the peer also
become anomalous — cascade up the call-graph from the rule-bearing
side.
