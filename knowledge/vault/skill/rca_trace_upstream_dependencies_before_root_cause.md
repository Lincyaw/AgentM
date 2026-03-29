---
confidence: pattern
source_trajectories:
- '43258'
- '43261'
tags:
- rca
- dependency-tracing
- best-practice
type: skill
---
# RCA Best Practice: Trace Upstream Dependencies Before Declaring Root Cause
## Application Rule
For every service identified as having observable anomalies (errors, latency, saturation) during RCA:
1. First query the service dependency graph to identify all directly and indirectly upstream dependent services
2. Collect anomaly signals (metrics, logs, traces) for all upstream services to check for earlier or concurrent failures
3. Verify the timestamp of the earliest anomaly for each service in the dependency chain
4. Only classify a service as root cause if no upstream service has anomalies occurring within a configurable 1-5 second time window before the service's first anomaly
5. If near-simultaneous anomalies are detected across multiple upstream services, trace request flows to identify the failure origin

## Rationale
Microservice request chains often produce cascading failures where downstream services exhibit anomalies first, even though the root cause is located in an upstream service. Skipping upstream dependency tracing leads to anchoring on intermediate symptomatic services rather than actual root causes.

## Success Metrics
Reduces false positive root cause identification of intermediate symptomatic services by eliminating premature termination of investigation before full dependency chain analysis is complete