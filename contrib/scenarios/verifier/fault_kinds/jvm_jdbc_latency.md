# jvm_jdbc_latency

## How the injection works
A JVM agent inserts latency into the target's JDBC layer, so every
outbound DB query on the target is slow. The database itself is
unaffected — the wait is inside the target's client code.

## What the data should show
Target's DB-call sub-spans show elevated duration; database-side
metrics (server latency, connection pool, query plans) are normal.
Target's inbound spans slow whenever they touch the DB code path.

## How to observe on a neighbour
Every endpoint on the target that touches the DB is slowed, so the
signal is broader than single-method faults — but callers that only
exercise non-DB paths are still unaffected. On a caller, check
both the service aggregate and per-`span_name` latency to see
which call paths are affected.

## How the failure tends to propagate
In-pod fault — target is the `from`, NOT the DB. Cascade reaches
callers whose requests serialise on slow DB responses through the
target. Don't attribute the edge to the DB.
