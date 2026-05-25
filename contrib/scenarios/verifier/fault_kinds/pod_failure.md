# pod_failure

## How the injection works
Chaos-mesh kills the target pod (sends SIGKILL to the container's
process) and prevents kubelet from restarting it for the configured
duration. The pod object stays in the cluster but has no running
process.

## What the data should show
Span volume produced by the target drops sharply — usually to **zero** —
across the abnormal window: the process is gone, so it emits no spans.
That zero IS the injection materialising, not missing data or "no
impact". The target's container restart-counter typically increments at
the boundaries.

The dominant downstream signal is a **throughput / completed-span
collapse**, NOT errors or latency. A caller's outbound calls to the dead
target fail fast or never return, so the caller completes fewer of its
own requests — its span volume drops too. Connection-refused /
dial-timeout error spans MAY appear on callers, but frequently do NOT
(the request is abandoned upstream before any error span is recorded),
and latency usually does not rise (failing paths return fast). So:
absence of 5xx / errors and absence of a latency increase are EXPECTED
for this fault — do not read them as absence of impact. Look for the
throughput drop.

## How the failure tends to propagate
The impact runs UP the call graph — the OPPOSITE of the request-call
direction (callers depend on the target, so the target's failure drags
its callers down) — and it is **transitive**. The target's dependency
vanishing throttles its direct callers' throughput; each throttled
caller then completes fewer of the requests ITS callers depend on, so
the collapse propagates another hop, attenuating outward until it
reaches the user-facing entry tier or fades into noise.

Trace it as a **growing affected frontier, not a star around the
target**: begin with the target as the only affected node; confirm each
direct caller's throughput collapse; then treat THAT caller as itself
affected and examine ITS callers next. The `from` of a propagation edge
is whichever already-affected service sits on the dependency side of the
hop — it is **not always the injection target**. An edge whose `from` is
a second- or third-hop service is normal and expected; rejecting an edge
merely because `from` is not the injection target is the single most
common mistake on this fault.

Because killing a dependency reduces traffic system-wide, a throughput
drop alone is evidence of a hop ONLY when the two services actually call
each other — confirm the call relationship (either direction, from the
normal window) before counting a drop as propagation.
