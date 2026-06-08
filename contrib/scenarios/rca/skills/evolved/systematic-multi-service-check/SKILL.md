---
name: systematic-multi-service-check
description: 'After finding one root cause service, systematically check all related services for additional faults before concluding.'
tags: ["rca", "insufficient_evidence", "premature_conclusion", "multi_service", "fault_isolation"]
trigger_patterns: ["When investigating a microservice incident", "After identifying one failing service", "When multiple services are involved in a failure", "When symptoms suggest cascading or co-occurring faults"]
type: skill
confidence: evolved
version: 1
evidence:
  train_cases: 6
  pattern_frequency: 6
---

## Systematic Multi-Service Check

When investigating a microservice incident, **never stop after finding the first failing service**. Incidents frequently involve multiple services failing simultaneously from different fault types (e.g., JVMRuntimeMutator on one service, NetworkLoss/NetworkPartition on another).

### Required Investigation Protocol

1. **Start with zero assumptions.** Always query metrics, logs, and service topology before concluding anything. Never skip investigative tool calls.

2. **When you find a failing service, ask: "What other services could also be failing?"** Specifically:
   - Check all upstream and downstream dependencies of the failing service.
   - Check all services in the same request path or call chain.
   - Check services that share infrastructure (same host, same namespace, same cluster).

3. **For each candidate service, check for DIFFERENT fault types.** The pattern across all cases is: Service A has a compute/memory fault (JVMRuntimeMutator, CPUStress, MemoryStress) while Service B has a network fault (NetworkLoss, NetworkPartition, NetworkCorrupt). These are independent — finding one does not rule out the other.

4. **Produce a complete fault matrix.** Before concluding, compile a list of all services with anomalies, noting the specific fault type for each. Only submit your conclusion when you have checked every service in the affected call chain.

5. **If you have zero tool calls in your trajectory, you have failed.** Every RCA must include at least 2-3 investigative queries to different data sources before a conclusion is valid.
