# http_response_status_modified

## How the injection works
The chaos proxy at the target rewrites HTTP response status codes
on matching paths — typically 2xx → 5xx. The body may be untouched
but the status changes.

## What the data should show
Inbound spans on the matched endpoint carry the rewritten status.
Callers treat the response as a failure.

## How the failure tends to propagate
Rule-bearing side is `app_name`. Callers of the affected endpoint
see RPC failures and may surface their own anomalies. Cascade
follows callers upward, gated by whether each layer fails on the
rewritten status.
