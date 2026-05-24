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

## How the failure tends to propagate
The rule-bearing side is `app_name` (the proxy-bearing service);
edge `app_name → peer-or-caller`. Callers of the failing endpoint
see RPC errors and may themselves surface anomalies on requests
that ultimately depend on this endpoint. Cascade follows callers
of the matched endpoint upward.
