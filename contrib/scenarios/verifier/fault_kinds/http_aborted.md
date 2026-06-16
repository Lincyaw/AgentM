# http_aborted

## How the injection works
A chaos proxy attached to the target intercepts HTTP traffic and
aborts requests matching the configured path / method, returning
5xx or resetting the connection. Only requests whose endpoint
matches the rule are affected; sibling endpoints on the same
service pass through.

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
