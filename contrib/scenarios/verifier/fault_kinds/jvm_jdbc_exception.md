# jvm_jdbc_exception

## How the injection works
A JVM agent rewrites the target's JDBC code path to throw, so the
target's outbound database calls fail. The database server itself
is healthy — the fault is in the target's client-side stack.

## What the data should show
Target's DB-call sub-spans show exceptions; the database service
shows no anomalies of its own (no elevated latency, no error
codes returned from the DB). Target's inbound spans may fail or
succeed depending on whether the failed DB call is recoverable.

## How the failure tends to propagate
In-pod fault — target (the JVM service) is the `from`, NOT the DB.
A common mistake is to attribute the edge to the DB; the DB is
healthy. Cascade extends from the target through callers whose
requests depend on DB-backed responses.
