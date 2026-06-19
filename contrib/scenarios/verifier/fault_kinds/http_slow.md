# http_slow

## How the injection works
The chaos proxy at the target delays HTTP responses on matching
path / method. The target's compute is unchanged; only the proxy
inserts the delay.

## Root-cause granularity
This is often a relationship/path fault, not a broken application
process. The proxy-bearing service may keep normal CPU, memory,
logs, and unrelated endpoints. If the injection metadata names a
peer service, treat the root cause as the affected HTTP link
(`link:source->target`). If no peer is named, decide from evidence:
caller-side latency on one endpoint with healthy target internals
points to a link/path-scoped root; broad target-side latency points
to a service-scoped anomaly.

If an upstream caller later sends fewer requests to unrelated
downstream dependencies, treat that reduced demand as evidence for
the upstream interrupted path, not as a final node for every healthy
callee. Promote a downstream `flow_interrupted` node only when the
interrupted endpoint is itself an alarm/user-visible path or has
timeout/error/fail-fast evidence.

## What the data should show
Inbound spans on the matched endpoint show a flat latency shift
equal to the injected delay. Error rate usually stable unless the
shift exceeds caller timeouts. Sibling endpoints unchanged.

### When the target looks healthy but traffic dropped

If the injected delay pushes callers past their timeout, callers
give up before the response returns. The target still processes the
request (and its spans show the delay), but fewer NEW requests
arrive because callers are blocked waiting. When the target's
traffic dropped but its own latency shows the expected delay, the
injection is confirmed from the target itself. But if the target's
spans don't capture the proxy delay, check the caller side for
timeout signals — same approach as network faults.

## How the failure tends to propagate
Rule-bearing side is `app_name`. Callers of the slowed endpoint
inherit the latency on requests that wait for that endpoint, and
their own callers may then be slowed in turn. Cascade is gated by
whether each caller's own latency budget tolerates the delay.
