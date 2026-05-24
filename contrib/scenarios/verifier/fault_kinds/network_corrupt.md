# network_corrupt

## How the injection works
A `tc netem` rule corrupts random bytes on a fraction of packets
through the target's interface. TCP catches most via checksums and
retransmits; some payloads still deserialise wrong at the receiver.

## What the data should show
Elevated retransmit counters and tail latency on traffic through the
rule-bearing side. Occasional decode / deserialisation / schema-
validation errors at the receiver. Error rate is usually below what
`loss` produces but visible.

## How the failure tends to propagate
Link-type fault. Edge: `rule-bearing side → peer`. Cascade tends to
be modest — most calls succeed — but services that retry aggressively
or fail-closed on a bad payload may surface anomalies further up.
