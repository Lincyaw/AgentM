# jvm_jdbc_latency

## How the injection works
A JVM agent inserts latency into the target's JDBC layer, so every
outbound DB query on the target is slow. The database itself is
unaffected — the wait is inside the target's client code.

## What the data should show
Target's DB-call sub-spans show elevated duration; database-side
metrics (server latency, connection pool, query plans) are normal.
Target's inbound spans slow whenever they touch the DB code path.

## How the failure tends to propagate
In-pod fault — target is the `from`, NOT the DB. Cascade reaches
callers whose requests serialise on slow DB responses through the
target. Don't attribute the edge to the DB.
