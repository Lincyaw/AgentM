# http_response_status_modified

## How the injection works
The chaos proxy at the target rewrites HTTP response status codes
on matching paths — typically 2xx → 5xx. The body may be untouched
but the status changes.

## Root-cause granularity
This is often a relationship/path fault, not a broken application
process. The proxy-bearing service may keep normal CPU, memory,
logs, response bodies, and unrelated endpoints. If the injection
metadata names a peer service, treat the root cause as the affected
HTTP link (`link:source->target`). If no peer is named, decide from
evidence: rewritten status limited to callers of one endpoint with
healthy target internals points to a link/path-scoped root; broad
target-side status failures point to a service-scoped anomaly.

If an upstream caller later sends fewer requests to unrelated
downstream dependencies, treat that reduced demand as evidence for
the upstream interrupted path, not as a final node for every healthy
callee. Promote a downstream `flow_interrupted` node only when the
interrupted endpoint is itself an alarm/user-visible path or has
timeout/error/fail-fast evidence.

## What the data should show
Inbound spans on the matched endpoint carry the rewritten status.
Callers treat the response as a failure.

## How the failure tends to propagate
Rule-bearing side is `app_name`. Callers of the affected endpoint
see RPC failures and may surface their own anomalies. Cascade
follows callers upward, gated by whether each layer fails on the
rewritten status.
