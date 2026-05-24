# pod_failure

## How the injection works
Chaos-mesh kills the target pod (sends SIGKILL to the container's
process) and prevents kubelet from restarting it for the configured
duration. The pod object stays in the cluster but has no running
process.

## What the data should show
Span volume produced by the target drops sharply — often to zero —
through the abnormal window. The target's container restart-counter
typically increments at the boundaries. Callers of the target log
connection-refused / dial-timeout errors on outbound spans whose
peer is the target.

## How the failure tends to propagate
The target is unreachable, so its direct callers see errors on every
attempt to reach it; whether those callers themselves become anomalous
depends on whether they have a fallback. From the target outward, the
graph follows the call-graph in reverse: direct callers → their
callers → ... — extending as long as each next hop's anomaly is
attributable to its dependency's failure.
