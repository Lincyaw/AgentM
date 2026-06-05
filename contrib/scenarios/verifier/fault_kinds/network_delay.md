# network_delay

## How the injection works
A `tc netem` rule on the target pod's network interface adds latency
to packets traversing that interface — both inbound and outbound.
The target's CPU, memory and application code are untouched; only
its network path is slowed.

## What the data should show
Inbound spans on the target show duration increases roughly equal to
the configured delay. Outbound spans from the target also slow.
Error rates usually stay flat unless the added delay pushes callers
past their timeout. The rule physically sits on ONE side of the
link, recorded as `injection_point.source_service`.

## How to observe on a neighbour
Network delay adds a fixed latency to every packet, so the signal
on a caller is a flat latency shift (not just a tail spike). Check
both average AND p99 — but the shift should be roughly commensurate
with the configured delay. Break down by `span_name` if the caller
has mixed endpoints: only call paths that route through the delayed
service are affected; unrelated endpoints stay flat.

## How the failure tends to propagate
This is a link-type fault. Latency cascades **upward**: a slow callee
makes its synchronous callers block, so they slow too, and so on up
the call graph. It does **not** flow downward — a service the target
merely *calls* is not slowed by the target's added inbound delay.

**Magnitude must be commensurate.** The propagated slowdown is a
blocking effect: a caller that waits on the slowed path gains roughly
the configured delay (seconds, typically), attenuating only mildly up
the chain. If the target slowed by seconds but a candidate service
moved by a few milliseconds, that candidate is NOT on the blocking
path — its wobble is noise, reject it. Confirm only services whose
p95/p99 rose by an absolute amount of the same order as the upstream's
increase.

### Uninstrumented backing components (DB / cache)
The `tc netem` rule slows ALL of the target's packets, including its
outbound DB/cache calls — so the target's own DB-client spans get
slower. That is the **target's egress delay**, already counted on the
target; it is NOT the backing component degrading. The DB serves every
other service at normal speed. Confirm a backing component only if its
OWN resource metrics worsen, or if MULTIPLE independent callers (not
just the fault-bearing one) show slower/erroring DB calls. A single
caller's slow client spans → reject the component.
