# pod_unavailable

## How the injection works
Chaos-mesh kills the container's process inside the target pod; the
pod stays scheduled and the container is restarted in place by the
kubelet. The effect is a brief outage rather than the indefinite gap
of `pod_failure`.

## What the data should show
A short gap in the target's span volume around the injection start,
followed by recovery once the container restarts. The container's
restart counter increments. Callers log transient connection errors
during the gap.

## How the failure tends to propagate
Same shape as `pod_failure` but shorter. Direct callers see a burst
of errors / timeouts during the gap, then recover. Cascade depth
depends on how many caller layers retry or fall back; with quick
restarts the impact often dies one or two hops out.
