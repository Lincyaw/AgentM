---
confidence: fact
description: 'Operational guide for the RCA orchestrator: dispatch_agent mechanics,
  context briefing protocol, service profile protocol, recall_history timing, and
  dispatch strategy.'
name: RCA Orchestrator Guide
tags:
- rca
- orchestrator
- operational
type: skill
---

# RCA Orchestrator Guide

Operational knowledge for the RCA orchestrator. Load this on your first round for dispatch protocols.

## dispatch_agent Mechanics

- **Auto-blocking**: when only one worker is running, `dispatch_agent` blocks and returns the result directly — no need for `check_tasks`.
- **Multiple workers**: returns immediately; call `check_tasks` to collect results.

## Context Briefing Protocol

Sub-agents run in ISOLATION — they ONLY see what you write in the `task` parameter.
Instruction quality directly determines investigation quality.

### Required elements (ALL dispatches)

Every `dispatch_agent` call MUST include in `task`:

1. **Target** (WHO): which service(s) to investigate, with known topology (caller → callee)
2. **Observable** (WHAT): specific anomaly being investigated — metric name, abnormal value, normal value, delta
3. **Timeframe** (WHEN): abnormal vs normal period identifiers (the agent uses `abnormal_*` / `normal_*` tables)
4. **Hypothesis** (WHY): which hypothesis is being tested, what the causal chain claims
5. **Data basis** (HOW it was measured): metric names, SQL filter conditions, or query approach
   that produced the prior findings — so the agent can reproduce or cross-validate
6. **Forward predictions**: if hypothesis is correct, what should be observable? If wrong, what would contradict it?
7. **Scope boundary**: how wide to look (single service internals? upstream? downstream?)

### Dispatch template by task type

**Scout** (broad survey):
> Survey the topology around `{service}`. Map call chains, identify anomalous services
> (latency delta > 2x OR error rate delta > 0), mark propagation direction.
> Known context: {prior findings with metric names and values}.
> Blind spots to fill: {what hasn't been checked yet}.

**Deep Analyze** (causal mechanism):
> Investigate WHY `{service}` shows {anomaly: metric_name value_abn vs value_nml (Nx)}.
> Known call chain: `A` → `B` → `C`. Prior findings: {agent_id found X using metric_name}.
> Testing H{n}: {hypothesis description}.
> Check these specific signals: {metric names or log patterns to look for}.
> If H{n} is correct, expect: (1) ... (2) ... (3) ...

**Verify** (adversarial disproof):
> Adversarially test: {full causal chain with each link}.
> Key data points to cross-validate (use these EXACT metrics/filters):
> - {metric_name}: {value_abn} vs {value_nml} ({delta}) [from {agent_id}]
> - {metric_name}: {value_abn} vs {value_nml} ({delta}) [from {agent_id}]
> - error rate: {value}% [filter: {exact SQL condition used}] [from {agent_id}]
> Verify each data point using the SAME metric/filter. If you use a different one, note it explicitly.
> What would contradict: {specific conditions that would break the chain}.

### Anti-patterns (DO NOT)
- "Investigate ts-preserve-service" — missing: what anomaly? what metric? what hypothesis?
- "Check CPU/memory/network" — missing: which CPU metric? what's the baseline?
- "Verify the causal chain: A → B → C" — missing: data points, metric names, filter conditions

## Service Profile Protocol

- **query before update**: ALWAYS call `query_service_profile` before `update_service_profile` to check what workers already recorded. Batch-query with comma-separated names.
- Only call `update_service_profile` when you have GENUINELY NEW cross-service insights, anomaly status corrections, or hypothesis linkage.
- Keep profile inputs SHORT — a profile is a quick-reference card, not a report.

## recall_history Timing

Only useful AFTER context compression has occurred. If full history is still in context, do not call it.

## Dispatch Strategy

**Default to PARALLEL, not serial.** One agent per round is the slowest possible investigation.

Each round, ask: "Can I formulate 2-3 INDEPENDENT tasks right now?"
- Different hypotheses -> different agents in parallel
- Different fault domains for the same symptom -> parallel
- Deep analysis on chain A + scout for unexplored area B -> parallel
- Verify H1 + deep analyze H2 -> parallel

Single-agent rounds should be rare after Round 1.

**Agent type selection:**
- **Scout**: broad coverage of unexplored areas. Best for: initial survey, filling blind spots, exploring a new fault domain.
- **Deep Analyze**: tracing causal mechanisms along a specific chain. Best for: why-chain drilling, finding the mechanism behind a symptom.
- **Verify**: stress-testing a hypothesis you're considering confirming. Best for: adversarial disproof.

**Isolation rule**: each parallel agent MUST have non-overlapping scope — different hypotheses, services, or evidence types.

**Quality check before EVERY dispatch**: re-read your task instruction and verify it includes scope, forward predictions, and enough context for the agent to work independently.
