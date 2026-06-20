# http_aborted

## How the injection works
A chaos proxy attached to the target intercepts HTTP traffic and
aborts requests matching the configured path / method, returning
5xx or resetting the connection. Only requests whose endpoint
matches the rule are affected; sibling endpoints on the same
service pass through.

## Root-cause granularity
This is often a relationship/path fault, not a broken application
process. The proxy-bearing service may keep normal CPU, memory,
logs, and unrelated endpoints. If the injection metadata names a
peer service, treat the root cause as the affected HTTP link
(`link:source->target`). If no peer is named, decide from evidence:
caller-side failures on one endpoint with healthy target internals
point to a link/path-scoped root; broad target-side failures point
to a service-scoped anomaly.

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
Inbound spans on the target on the matched path show 5xx / connection
errors. The error is emitted by the proxy at the target — no
deeper downstream is involved. Sibling endpoints stay clean.

### When the target looks healthy but traffic dropped

If the proxy aborts requests before the application processes them,
the target's own application-level spans may not record the failure
(the proxy rejects at the transport layer). In that case the
target looks "healthy but idle." Check the caller side: callers
should show 5xx errors or connection resets on the specific
endpoint that matches the abort rule.

## How the failure tends to propagate
The rule-bearing side is `app_name` (the proxy-bearing service);
edge `app_name → peer-or-caller`. Callers of the failing endpoint
see RPC errors and may themselves surface anomalies on requests
that ultimately depend on this endpoint. Cascade follows callers
of the matched endpoint upward.
