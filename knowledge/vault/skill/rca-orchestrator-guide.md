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

Sub-agents run in ISOLATION — they ONLY see what you write in the `task` parameter. Instruction quality directly determines investigation quality.

Every `dispatch_agent` call MUST include in `task`:
- Key prior findings: service names, timestamps, anomalies, call chains
- Specific signals to investigate: time range, services, anomaly types
- Which hypothesis is being tested and current evidence
- What would constitute supporting vs contradicting evidence
- **Scope boundaries**: how wide the agent should look
- **Forward predictions**: predicted consequences so the agent knows WHAT to look for

Example of a GOOD dispatch instruction:
> "ts-order-service showed 45% error rate in abnormal traces after timestamp 1699900054000000000. Check abnormal logs for ts-order-service errors in that timeframe to identify the specific error type. Testing H2 (DB timeout cascade). If H2 is correct, we expect: (1) connection timeout errors in ts-order-service logs, (2) outbound calls to MongoDB showing elevated latency, (3) MongoDB connection metrics near pool limit. Check all three."

For scout: instruct comprehensive scope — survey full topology, all call chains, all anomalous services.
For verify: include the full causal chain and explicit per-link evidence to challenge.

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
