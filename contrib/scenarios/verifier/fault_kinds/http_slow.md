# http_slow

## How the injection works
The chaos proxy at the target delays HTTP responses on matching
path / method. The target's compute is unchanged; only the proxy
inserts the delay.

## What the data should show
Inbound spans on the matched endpoint show a flat latency shift
equal to the injected delay. Error rate usually stable unless the
shift exceeds caller timeouts. Sibling endpoints unchanged.

## How the failure tends to propagate
Rule-bearing side is `app_name`. Callers of the slowed endpoint
inherit the latency on requests that wait for that endpoint, and
their own callers may then be slowed in turn. Cascade is gated by
whether each caller's own latency budget tolerates the delay.
