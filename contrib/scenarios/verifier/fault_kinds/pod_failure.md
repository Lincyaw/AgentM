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

### When the target looks healthy (quick restart)

If kubelet restarts the pod quickly (< 60 s), the target's aggregate
metrics over the full abnormal window may look nearly normal — the
brief outage is diluted by the healthy period after restart. The
signal then lives on the **caller side** during the kill window:

- Callers return **5xx** or connection errors on requests that
  hit the dead pod — sometimes with short latency (fast connection
  refused), sometimes with longer latency (waiting for timeout).
- The caller's HTTP status breakdown or Error-status spans are
  concentrated in a narrow time band matching the kill window.
- k8s deployment-available metrics may stay at their normal value
  if the sampling interval is coarser than the outage.
- explicit restart counters are often missing. In that case, look for
  **process-restart fingerprints** in metrics:
  - monotonic counters such as `container.cpu.time`, `jvm.cpu.time`,
    `process.runtime.*`, exported span/log counters, or similar values
    reset to a much lower value at the start of the abnormal window;
  - JVM/application startup counters jump or reload, e.g.
    `jvm.class.loaded` spikes after being near zero;
  - container/JVM memory usage briefly drops to a fresh-process
    baseline before recovering.

These counter resets are strong seed evidence for `process_killed` even
when traces look normal over the full abnormal window and
`k8s.deployment.available` remains 1.

Important: the restart fingerprint is often in cumulative/sum metric
tables such as `normal_metrics_sum` / `abnormal_metrics_sum`, not in the
gauge table. Discover metric names across all metric-like tables before
claiming restart evidence is absent; exact metric names vary, so prefer
pattern searches such as `%cpu%time%`, `%class%loaded%`, `%memory%usage%`,
`%rss%`, and `%restart%`. For monotonic counters, compare the last
normal sample with the first abnormal sample. A lower abnormal value is
the reset signal; it does not need to be exactly zero. Do not summarize
only min/max and conclude "no reset" when the abnormal counter range is
below the normal window's end value.

Seed confirmation is separate from propagation confirmation. A restart
fingerprint can confirm the injected pod kill on the target even when no
caller or entrypoint is visibly degraded. Propagation to callers still
requires the caller-side evidence described below.

When the target's own span count and latency look nearly unchanged,
check the caller side:

1. JOIN `parent_span_id` in the normal window to find callers.
2. In the abnormal window, check those callers for error responses,
   Error-status spans, or latency spikes on calls to the target.
3. Also check container restart metrics for the target — an
   increment confirms the kill even when other metrics missed it.

If caller-side errors appear on the target's call path, the kill
worked. Mark **confirmed** with `process_killed`.

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
drop alone is only investigation evidence, not a confirmed hop. First
confirm the call relationship (either direction, from the normal window),
then require a caller-side failure signal before marking propagation:
HTTP/gRPC errors, timeout-level p99/max, near-total/selective
disappearance of the endpoint that normally calls the killed dependency,
or a dramatic fail-fast latency drop on that same endpoint. A small
proportional count drop with zero errors and slightly lower latency is
not flow interruption; reject it if metrics/logs are otherwise healthy,
or mark inconclusive only when the path vanished but context is missing.

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
per-`span_name` before concluding. But do not overread tiny latency
drops: fast-fail propagation means the dependency call path changed
materially. If the caller endpoint has the same shape, no errors, no
timeout, no selective disappearance, and only a modest lower p99 during
a proportional traffic dip, the caller is not confirmed degraded.

### Uninstrumented backing components (DB / cache)
These have no spans. Judge them via the CLIENT-side spans inside the
caller (filter the caller's spans to its client/outbound kind — check
how this dataset marks span direction first).
If the client-span latency or error rate for DB operations worsened,
the backing component is affected. If the caller simply stopped
sending DB calls (because the caller itself died), the backing
component is NOT degraded — its CPU/memory dropping merely reflects
reduced load.
