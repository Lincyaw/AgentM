# mem_stress

## What the fault physically does
Stress-ng allocates memory inside the target pod, driving it toward OOM. The
effect is GC thrashing (JVM apps), swap thrashing, or — if an OOM-kill fires —
a container restart. During the pressure phase it behaves as a LATENCY fault;
an OOM-kill turns that interval into a brief outage. Steady 4xx/5xx are not
its signature.

## Data profile — what "effective" looks like
On the TARGET's own spans:

- inbound `duration` RISES in the abnormal window (GC / swap pauses), p95/p99
  especially — same latency signature as cpu_stress. Compare `AVG(duration)`
  and p95 `duration` on the target, normal vs abnormal.
- if OOM-killed: a gap / throughput collapse on the target for the restart
  interval (see pod_unavailable), and callers may see connection errors only
  during that gap.

A target-specific latency rise (or a restart gap) = it materialised.

## When it did NOT materialise (effectiveness gate)
Same gate as cpu_stress. If the target's latency is unchanged and there is no
restart gap, the injection did not engage — report `injection_effective:
false`/`ambiguous` and emit NO propagation edges from it. Confounds that are
NEVER edge evidence:

- a roughly UNIFORM drop in span COUNT across many/all services is load /
  throughput variation in the capture window, not this fault.
- co-occurrence (two services both looking "off" but with no target latency
  rise and no direct call carrying it) is not an edge.

A weak injection that produced no measurable latency change or restart gap is
exactly the case that must NOT be turned into a propagation graph.

## How it propagates (only if it materialised)
In-pod fault — the target is the `from`. During the pressure phase it
propagates like cpu_stress: a direct caller waits on the slow dependency, its
OWN latency rises, and it becomes the `from` of the next hop — each hop must
show its own latency rise or the cascade stops there. If OOM-kill fires, that
interval degenerates into a pod_unavailable-style throughput-collapse cascade.

### Co-deployed (shared-node) neighbours
Same principle as cpu_stress: memory stress on pod A does NOT
automatically degrade pod B on the same node. A small latency wobble on
a co-deployed neighbour that falls within its normal day-to-day variance
is jitter, not propagation. To confirm, look for a disproportionate
change that clearly stands out, AND corroborate with node-level memory
metrics showing actual pressure (e.g. page faults spiking, working_set
approaching node capacity).
