# http_payload_modified

## How the injection works
The chaos proxy at the target mutates response body bytes on matching
paths. The transport-level status code may stay 2xx; only the body
is altered.

## Root-cause granularity
This is often a relationship/path fault, not a broken application
process. The proxy-bearing service may keep normal CPU, memory,
logs, status codes, and unrelated endpoints. If the injection
metadata names a peer service, treat the root cause as the affected
HTTP link (`link:source->target`). If no peer is named, decide from
evidence: malformed-response or validation errors limited to callers
of one endpoint with healthy target internals point to a link/path-
scoped root; broad target-side data corruption points to a service-
scoped anomaly.

If an upstream caller later sends fewer requests to unrelated
downstream dependencies, treat that reduced demand as evidence for
the upstream interrupted path, not as a final node for every healthy
callee. Promote a downstream `flow_interrupted` node only when the
interrupted endpoint is itself an alarm/user-visible path or has
timeout/error/fail-fast evidence.

## What the data should show
Inbound spans on the matched endpoint may show 2xx status but
callers downstream of those responses log deserialisation /
validation errors. Status-code-only views may hide the fault.

## How the failure tends to propagate
Rule-bearing side is `app_name`. Callers that parse the mutated
body fail; their callers see RPC errors in turn. Cascade follows
the call-graph from the affected endpoint outward, gated by whether
each layer fails strictly on malformed input.
