# pod_unavailable

## How the injection works
Chaos-mesh kills the container's process inside the target pod; the
pod stays scheduled and the container is restarted in place by the
kubelet. The effect is a brief outage rather than the indefinite gap
of `pod_failure`.

## What the data should show
The container is gone for a brief window and then restarts in place, so
requests arriving in that window cannot complete normally. How it
surfaces depends on traffic and timing — read the WHOLE picture, do not
expect one fixed signature:

- requests that would have started during the outage don't, so the
  target's completed-span volume drops — toward **zero** under steady
  traffic, only **partially** under low / bursty traffic;
- requests in flight when the container dies, or that block trying to
  reach it, hang until they time out or it restarts — so the target's
  (and its callers') **latency TAIL can explode** (ms → tens of seconds)
  and transient connection / timeout **errors** may appear.

Throughput collapse, a latency-tail explosion, and connection errors are
all faces of the SAME brief outage; which dominates depends on the case.
A modest span-count dip COMBINED with a latency tail exploding from ms to
tens of seconds is the outage biting just as much as a drop to zero — do
NOT call it ineffective merely because spans did not reach near-zero. The
container's restart counter increments. Judge from what the data actually
shows, across traces, metrics and logs.

## How the failure tends to propagate
Same shape as `pod_failure`: the impact runs UP the call graph (opposite
to request-call direction) and is **transitive**. A direct caller's calls
to the unavailable target hang or fail, so the caller shows the same mix —
fewer completed requests AND/OR a latency tail / connection errors — and
each affected caller drags ITS callers the same way, attenuating outward.

Trace it as a **growing affected frontier, not a star around the
target**: confirm each direct caller is genuinely affected (throughput
drop and/or latency-tail / error spike on its calls to the dependency),
then treat that caller as itself affected and examine ITS callers next.
The `from`
of a propagation edge is whichever already-affected service sits on the
dependency side of the hop — **not always the injection target**; a
second- or third-hop `from` is normal. Rejecting an edge only because
`from` is not the injection target is the most common mistake here.
Cascade depth depends on how quickly the container restarts and on
caller retries / fallbacks; a brief outage may die one or two hops out,
so verify each hop rather than assuming the chain reaches the entry tier.

Because the outage reduces traffic system-wide, a throughput drop is
evidence of a hop ONLY when the two services actually call each other —
confirm the call relationship (either direction, normal window) first.

### Throughput drop — distinguish by call direction
Same rule as `pod_failure`: a callee that simply receives fewer
requests (target stopped calling) is healthy — reject. But for a
caller whose specific endpoint to the target vanished, mark as
**inconclusive** — use the normal window to identify which caller
endpoints interact with the target, then check if those vanished
in the abnormal window. The judge can determine causality with the
full graph context.
