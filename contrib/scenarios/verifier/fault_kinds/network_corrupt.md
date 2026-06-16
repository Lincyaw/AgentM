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

### When the target looks healthy but traffic dropped

Heavy corruption can cause TCP to drop/retransmit so many packets
that callers time out. Requests that survive arrive intact and
complete normally at the target. When the target's latency and
error rate look fine but span volume dropped significantly, check
the caller side — the same methodology as network_partition and
network_loss applies.

## How to observe on a neighbour
Like network_loss, corruption causes retransmissions that appear as
**tail-latency spikes** (p99/max). Average latency may look flat.
Additionally, corrupted payloads that survive TCP checksums can
cause deserialization errors at the receiver — check error logs for
decode/schema/parse failures. On a neighbour, always check p99
latency AND error logs, not just aggregate averages.

## How the failure tends to propagate
Link-type fault. Edge: `rule-bearing side → peer`. Cascade tends to
be modest — most calls succeed — but services that retry aggressively
or fail-closed on a bad payload may surface anomalies further up.
