# network_duplicate

## How the injection works
A `tc netem` rule causes random duplication of packets on the
target's interface. TCP normally dedupes transparently, so the
application layer sees little or nothing.

## What the data should show
Often subtle. Slight throughput overhead and bandwidth waste; rarely
a clear application-level signal. Some idempotent / UDP traffic may
show duplicate processing.

## How the failure tends to propagate
Link-type fault. Edge: `rule-bearing side → peer`. Cascade is
usually nil or ambiguous — be ready to mark `injection_effective:
ambiguous` if no signal materialises.
