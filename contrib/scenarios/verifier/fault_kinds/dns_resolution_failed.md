# dns_resolution_failed

## How the injection works
DNS lookups inside the target pod (typically via a chaos rule on
the pod's DNS resolver) are made to fail. The target's outbound
calls cannot resolve peer hostnames; inbound traffic to the
target's existing IP is unaffected unless the target itself needs
DNS to serve requests.

## What the data should show
Target logs name-resolution errors. Target's outbound spans error
out at the resolve stage, before reaching any peer. The peers
themselves show no anomalies — they simply receive no traffic.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade reaches callers whose
requests required the target to make outbound DNS-dependent calls.
The peers the target failed to resolve are NOT downstream of the
target — they're unreachable destinations, not victims.
