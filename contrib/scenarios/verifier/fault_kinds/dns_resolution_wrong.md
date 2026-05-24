# dns_resolution_wrong

## How the injection works
DNS lookups inside the target return wrong IPs. The target's
outbound calls connect to unintended hosts (often nonexistent or
the wrong service), or hang waiting for replies that never come.

## What the data should show
Target's outbound spans show unexpected errors or timeouts. Logs
may cite mismatched TLS hostnames or refused connections from the
wrong-IP peers.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade reaches callers whose
requests required the target to reach a now-misdirected dependency.
Like `dns_resolution_failed`, the legitimate peers are NOT victims.
