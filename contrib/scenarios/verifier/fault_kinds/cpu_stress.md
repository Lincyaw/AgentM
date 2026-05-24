# cpu_stress

## How the injection works
Stress-ng (or equivalent) spawns busy threads inside the target pod,
saturating CPU. The target's processes still respond, but slowly —
schedulers can't give request handlers enough cycles.

## What the data should show
Target CPU utilisation spikes; inbound spans on the target show
elevated latency across all endpoints (the bottleneck is CPU, not
any one code path). Error rate rises if callers' timeouts trip.

## How the failure tends to propagate
In-pod fault — target is the `from`. Direct callers of the target
see latency / timeout on requests that wait for it; their own
latency budget determines whether they themselves become anomalous.
Cascade extends through callers as long as each layer's anomaly is
attributable to its slow dependency.
