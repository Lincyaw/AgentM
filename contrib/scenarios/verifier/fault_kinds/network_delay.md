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

## How the failure tends to propagate
This is a link-type fault. The edge `rule-bearing side → peer`
records that the side carrying the rule causes the slow / failed
communication; the peer experiences unreachability / latency from
its perspective. From the rule-bearing side, latency cascades upward
through its callers, who themselves slow down as their own
dependencies-of-this-side block on slow responses.
