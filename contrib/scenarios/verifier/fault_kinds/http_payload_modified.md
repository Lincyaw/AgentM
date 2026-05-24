# http_payload_modified

## How the injection works
The chaos proxy at the target mutates response body bytes on matching
paths. The transport-level status code may stay 2xx; only the body
is altered.

## What the data should show
Inbound spans on the matched endpoint may show 2xx status but
callers downstream of those responses log deserialisation /
validation errors. Status-code-only views may hide the fault.

## How the failure tends to propagate
Rule-bearing side is `app_name`. Callers that parse the mutated
body fail; their callers see RPC errors in turn. Cascade follows
the call-graph from the affected endpoint outward, gated by whether
each layer fails strictly on malformed input.
