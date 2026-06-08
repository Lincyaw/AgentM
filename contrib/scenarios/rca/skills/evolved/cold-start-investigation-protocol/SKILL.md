---
name: cold-start-investigation-protocol
description: 'A mandatory step-by-step first-move protocol to prevent empty-trajectory failures by ensuring the agent always begins investigation with concrete tool calls.'
tags: ["investigation-startup", "empty-trajectory-prevention", "first-move-protocol", "tool-call-initiation"]
trigger_patterns: ["When the agent receives an incident to investigate and has not yet made any tool calls", "When the agent is about to start an RCA investigation from scratch", "When the agent has no prior context or data about the incident", "At the very beginning of any RCA workflow before any analysis is performed"]
type: skill
confidence: evolved
version: 1
evidence:
  train_cases: 20
  pattern_frequency: 12
---

## Cold-Start Investigation Protocol

**Never produce an empty trajectory.** The moment you receive an incident, you MUST make at least one tool call before any analysis or conclusion. Follow this exact sequence:

### Step 1: Fetch Service Metrics (MANDATORY FIRST MOVE)
Call the metrics/health-check tool to get the current status of all services involved in the incident. Do NOT skip this step. Do NOT jump to conclusions. Do NOT output text without first calling a tool.

### Step 2: Check Service Dependencies
Query the service dependency graph or topology to understand which services interact with the affected ones. Look for upstream/downstream relationships.

### Step 3: Gather Logs or Anomaly Signals
Pull logs, error rates, or anomaly signals for the top-3 most suspicious services identified in Steps 1-2.

### Step 4: Form Hypothesis (ONLY after Steps 1-3)
Only after gathering data from at least 2-3 tool calls should you form a hypothesis. List candidate services with supporting evidence.

### Step 5: Verify
Cross-check each candidate service against the evidence. Confirm fault types (PodFailure, CPUStress, MemoryStress, NetworkLoss, etc.) for each root cause service.

### Anti-Patterns to Avoid
- ❌ Writing analysis text without any tool calls
- ❌ Submitting a conclusion with zero evidence gathered
- ❌ Skipping metrics/logs and guessing based on service names alone
- ❌ Making only one tool call and concluding

**Checklist before submitting:** Did I make ≥3 tool calls? Did I gather metrics AND logs? Did I verify each root cause with evidence?
