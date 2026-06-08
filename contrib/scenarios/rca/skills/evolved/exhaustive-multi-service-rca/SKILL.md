---
name: exhaustive-multi-service-rca
description: 'Systematic multi-service investigation protocol to avoid premature single-cause conclusions when multiple services are failing simultaneously.'
tags: ["rca", "premature_conclusion", "insufficient_evidence", "multi-service", "microservice"]
trigger_patterns: ["When the incident involves a request flow through multiple microservices", "When a single service anomaly is found but the error could propagate from upstream/downstream services", "When symptoms suggest a distributed failure (e.g., timeout, degraded response)", "When the dependency graph shows multiple candidate services in the call chain"]
type: skill
confidence: evolved
version: 1
evidence:
  train_cases: 11
  pattern_frequency: 8
---

## Exhaustive Multi-Service RCA Protocol

Microservice incidents frequently involve **multiple independent faults** across different services. Never stop at the first anomaly. Follow this protocol:

### Step 1: Map the dependency chain
Identify ALL services in the call path of the failing endpoint. List them as candidates.

### Step 2: Investigate every candidate service
For each service in the dependency chain, gather evidence:
- **Metrics**: Check error rates, latency, CPU/memory for each service.
- **Logs**: Scan recent error logs for each service.
- **Fault injection**: Check if any fault tool is active on each service.

### Step 3: Apply the "Two-Fault Rule"
If you find one faulty service, **do not conclude**. Assume there may be a second co-fault until you have explicitly ruled out every other service in the dependency chain. The most common pattern is two services failing with **different fault types** (e.g., JVMRuntimeMutator on one, NetworkPartition on another).

### Step 4: Build a complete fault matrix
Before finalizing, construct a table:
| Service | Fault Found? | Fault Type | Evidence Source |
If any cell in the "Fault Found?" column is unchecked, you have NOT finished investigating.

### Step 5: Verify completeness
Ask: "Could there be another service also contributing to this incident?" If yes, investigate it. Only conclude when ALL services in the dependency chain have been examined and accounted for.

**Key rule**: Zero tool calls = zero confidence. Always gather evidence before concluding.
