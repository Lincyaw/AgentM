# pod_failure

## How the injection works
Chaos-mesh kills the target pod (sends SIGKILL to the container's
process) and prevents kubelet from restarting it for the configured
duration. The pod object stays in the cluster but has no running
process.

## What the data should show
For the configured window the target has no running process, so requests
that arrive then cannot complete normally. How that surfaces depends on
traffic volume and timing — read the WHOLE picture; do not expect one
fixed signature:

- requests that would have started during the dead window never do, so
  the target's completed-span volume drops — toward **zero** under
  steady traffic, or only **partially** under low / bursty traffic;
- requests in flight when the pod dies, or that arrive and block trying
  to reach it, hang until they time out or the pod returns — so the
  target's (and its callers') **latency TAIL can explode** (e.g. p90/p99
  jumping from milliseconds to tens of seconds), and connection-refused /
  timeout **error spans** may appear.

A near-total throughput collapse, a latency-tail explosion, and
connection errors are all faces of the SAME kill; which dominates is a
function of the case, not a fixed rule. In particular, a modest
span-count dip COMBINED with a latency tail blowing up from ms to tens of
seconds is the kill biting just as much as a clean drop to zero is — do
NOT dismiss an injection as ineffective merely because spans did not fall
to near-zero. The container's restart counter typically increments at the
boundaries. Judge from what the data actually shows, across traces,
metrics and logs.

## How the failure tends to propagate
The impact runs UP the call graph — the OPPOSITE of the request-call
direction (callers depend on the target, so the target's failure drags
its callers down) — and it is **transitive**. A direct caller's calls to
the dead dependency hang or fail, so the caller shows the same mix:
fewer completed requests AND/OR a latency tail / connection errors on the
calls that blocked. Each affected caller then drags ITS callers the same
way, attenuating outward until it reaches the entry tier or fades into
noise.

Trace it as a **growing affected frontier, not a star around the
target**: begin with the target as the only affected node; confirm each
direct caller is genuinely affected (a throughput drop and/or a
latency-tail / error spike on its calls to the dependency); then treat
THAT caller as itself affected and examine ITS callers next. The `from`
of a propagation edge
is whichever already-affected service sits on the dependency side of the
hop — it is **not always the injection target**. An edge whose `from` is
a second- or third-hop service is normal and expected; rejecting an edge
merely because `from` is not the injection target is the single most
common mistake on this fault.

Because killing a dependency reduces traffic system-wide, a throughput
drop alone is evidence of a hop ONLY when the two services actually call
each other — confirm the call relationship (either direction, from the
normal window) before counting a drop as propagation.

### Throughput drop without latency / error change
When a service's span count drops but its latency and error rate are
unchanged, the service itself is healthy — its callers simply stopped
sending it requests (because THEIR upstream died). This is NOT
degradation of the checked service. Reject it.

A genuine propagation hop shows the service's OWN health worsening:
latency tail exploding, error rate rising, or connection errors
appearing — not just "fewer requests arrived."

### Aggregate latency drops after a kill (overloaded baseline)
Sometimes the normal window already shows high latency (resource
pressure before the fault). When the fault kills a dependency and
traffic drops, contention eases and the surviving requests complete
faster — so aggregate latency DROPS in the abnormal window.

A latency drop does not automatically mean "fine." If the aggregate
is ambiguous, check per-span_name: the call path that depends on
the killed service may show its spans disappearing or fast-failing
while unrelated paths stay normal.

### Uninstrumented backing components (DB / cache)
These have no spans. Judge them via the CLIENT-side spans inside the
caller (filter the caller's spans to its client/outbound kind — check
how this dataset marks span direction first).
If the client-span latency or error rate for DB operations worsened,
the backing component is affected. If the caller simply stopped
sending DB calls (because the caller itself died), the backing
component is NOT degraded — its CPU/memory dropping merely reflects
reduced load.
