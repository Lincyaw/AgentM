---
confidence: pattern
source_trajectories:
- '43258'
- '43261'
tags:
- rca
- anchoring-bias
- premature-termination
- dependency-tracing-gap
type: failure-pattern
---
# Failure Pattern: RCA Agent Anchors on First Detected Anomalous Service Without Dependency Tracing
## Mechanism
When performing microservice root cause analysis, the agent immediately classifies the first service with observable anomalies (high error rate, latency spike) as the root cause, without:
1. Tracing upstream dependencies to confirm if the anomalies originate from an earlier failure in the service chain
2. Checking if the anomalous service is only an intermediate symptom rather than the source of the failure
3. Verifying anomaly timestamps across the full dependency chain to identify the earliest failure point

## Evidence
Observed in 2 independent RCA trajectories:
1. Case 43258: Agent anchored on `ts-travel2-service` anomalies, failed to investigate upstream `ts-basic-service` and `ts-price-service` which were the actual root causes
2. Case 43261: Agent anchored on `ts-travel2-service` anomalies, failed to investigate upstream `ts-config-service` and its backing `mysql` database which were the actual root causes

In both cases, no dependency discovery or tracing queries were executed for the anomalous service before declaring root cause.

## Mitigation Recommendations
1. Mandate a dependency tracing workflow step for every detected anomalous service before root cause declaration
2. Implement a guardrail requiring confirmation that no upstream service has anomalies occurring within a 1-5 second time window before the detected service anomaly
3. Ensure agents have access to dependency mapping tooling to identify upstream/downstream service relationships for any service