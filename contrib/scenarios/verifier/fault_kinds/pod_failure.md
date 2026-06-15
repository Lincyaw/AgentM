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

- alternatively, the caller's client library may detect the dead
  connection instantly (connection refused) and return in microseconds
  instead of milliseconds — **latency DROPS sharply** rather than
  exploding. The client may not mark this as an error (gRPC/HTTP
  frameworks often treat fast connection failures as non-error returns).
  A latency drop from ms to μs on calls to a dead dependency is the
  fast-fail signature, not evidence of health.

A near-total throughput collapse, a latency-tail explosion, connection
errors, AND fast-fail latency drops are all faces of the SAME kill;
which dominates is a function of the client library and case, not a
fixed rule. In particular, a modest
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

### Throughput drop — distinguish by call direction

**Callees of the target** (services the killed target used to call):
they receive fewer requests because the target stopped calling them.
Their own latency and error rate are unchanged — they are healthy,
just idle. This is NOT degradation. Reject.

**Callers of the target** (services that call the killed target):
a throughput-only drop does NOT automatically mean "healthy." The
caller's aggregate may look fine because unaffected endpoints dilute
it, but the specific endpoint(s) that depend on the dead target may
have vanished. Use the NORMAL window to find which of the caller's
endpoints interact with the target (JOIN on parent_span_id), then
check whether those endpoints disappeared in the abnormal window.

If the fault-related endpoints vanished while other endpoints are
healthy, this is ambiguous from a single-edge view — mark as
**inconclusive** so the judge can determine, with the full graph,
whether the disappearance traces back to the confirmed fault.

A caller whose aggregate latency drops after the kill is NOT
automatically "fine." The drop often means the slow dependency is
gone and surviving requests complete faster. Always check
per-`span_name` before concluding.

### Uninstrumented backing components (DB / cache)
These have no spans. Judge them via the CLIENT-side spans inside the
caller (filter the caller's spans to its client/outbound kind — check
how this dataset marks span direction first).
If the client-span latency or error rate for DB operations worsened,
the backing component is affected. If the caller simply stopped
sending DB calls (because the caller itself died), the backing
component is NOT degraded — its CPU/memory dropping merely reflects
reduced load.
