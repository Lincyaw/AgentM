# cpu_stress

## What the fault physically does
Stress-ng (or equivalent) spawns busy threads inside the target pod and
saturates its CPU. The process keeps answering — it is **not** down — but
every request handler is starved of cycles, so it answers SLOWLY. This is a
LATENCY fault. It is not an outage and not an error fault.

## Data profile — what "effective" looks like
The single load-bearing signature is on the TARGET's own spans:

- inbound span `duration` on the target RISES in the abnormal window vs the
  normal window, across endpoints (the bottleneck is the CPU, not one code
  path). Check the tail, not just the mean — `PERCENTILE_CONT(0.95)` /
  `0.99 WITHIN GROUP (ORDER BY duration)`; the p95/p99 moves most.
- the rise is SPECIFIC to the target (and, transitively, callers that wait on
  it) — not something every service shares.
- errors (4xx/5xx) are NOT the primary signal. They appear only as a
  secondary effect if a caller's timeout trips. Do not look for errors to
  confirm CPU stress — look for latency.

Concretely: compare `AVG(duration)` and p95 `duration` on the target in the
normal vs abnormal window. A clear, target-specific increase = it materialised.

## When it did NOT materialise (effectiveness gate)
If the target's own latency is essentially UNCHANGED between the two windows,
the injection did not engage. Report `injection_effective: false` (or
`ambiguous`) and emit NO propagation edges from it — even if other services
look different. Two confounds that are NEVER edge evidence:

- a roughly UNIFORM drop in span COUNT across many/all services (everything
  ~30-40% fewer spans) is load / throughput variation in the capture window,
  not this fault. A real CPU-stress signature is a latency rise on the target,
  not a throughput dip shared by unrelated services.
- two services merely co-occurring in the same traces, or both looking "off",
  is not propagation. Without (1) a target latency rise AND (2) a direct call
  carrying that latency to the neighbour, there is no edge.

A weak injection that produced no measurable latency change is exactly the
case that must NOT be turned into a propagation graph.

## How it propagates (only if it materialised)
In-pod fault — the target is the `from`. A direct caller of the target waits
on the slow dependency, so the caller's OWN latency rises; that caller then
becomes the `from` of the next hop. The cascade extends through callers only
as long as each layer shows its own latency rise attributable to the slow
dependency. A caller whose latency is flat stops the cascade — do not extend
past it.
