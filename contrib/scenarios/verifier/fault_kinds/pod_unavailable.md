# pod_unavailable

## How the injection works
Chaos-mesh kills the container's process inside the target pod; the
pod stays scheduled and the container is restarted in place by the
kubelet. The effect is a brief outage rather than the indefinite gap
of `pod_failure`.

## What the data should show
A sharp drop — often to zero — in the target's span volume across the
abnormal window (recovery may begin once the container restarts). That
near-zero IS the injection materialising, not missing data or "no
impact". The container's restart counter increments.

As with `pod_failure`, the dominant downstream signal is a **throughput
/ completed-span collapse**, NOT errors or latency. Callers' calls to
the unavailable target fail fast or never return, so the callers
complete fewer requests and their own span volume drops. Transient
connection errors MAY appear on callers but frequently do not, and
latency usually does not rise — so absence of 5xx / errors and absence
of a latency increase are EXPECTED, not evidence of no impact. Look for
the throughput drop.

## How the failure tends to propagate
Same shape as `pod_failure`: the impact runs UP the call graph (opposite
to request-call direction) and is **transitive**. The unavailable target
throttles its direct callers' throughput; each throttled caller then
completes fewer of the requests ITS callers depend on, so the collapse
propagates another hop, attenuating outward toward the user-facing entry
tier.

Trace it as a **growing affected frontier, not a star around the
target**: confirm each direct caller's throughput collapse, then treat
that caller as itself affected and examine ITS callers next. The `from`
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
