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

When metadata has `app_name`/rule-bearing service plus
`server_address`/peer service, the configured `route` is usually the
peer endpoint seen on the child span, not a route served by the
rule-bearing service itself. Establish the normal path with
`app_name -> server_address` parent-span joins before searching for the
route. Do not reject because the route is absent from the app service's
own span names.

If an upstream caller later sends fewer requests to unrelated
downstream dependencies, treat that reduced demand as evidence for
the upstream interrupted path, not as a final node for every healthy
callee. Promote a downstream `flow_interrupted` node only when the
interrupted endpoint is itself an alarm/user-visible path or has
timeout/error/fail-fast evidence.

## What the data should show
Inbound spans on the matched endpoint carry the rewritten status.
Callers treat the response as a failure.

For request path replacement variants that are normalized into this family,
the original matched endpoint may have zero abnormal spans because the proxy
rewrote the request path before the target application routed it. Check for
new span names, wildcard/error-handler routes such as `GET /**` or
`BasicErrorController.error`, HTTP 404/5xx on the peer, and caller-side 5xx or
exceptions that mention the rewritten path. Original path disappearance plus a
new 404/error path is a positive signal, not no evidence.

## How the failure tends to propagate
Rule-bearing side is `app_name`. Callers of the affected endpoint
see RPC failures and may surface their own anomalies. Cascade
follows callers upward, gated by whether each layer fails on the
rewritten status.
