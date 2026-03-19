---
confidence: fact
description: 'Shared operational guide for RCA worker agents (scout, deep_analyze,
  verify): data sources, tooling quickstart, service profile protocol, tool discipline,
  and anomaly definition.'
name: RCA Worker Guide
tags:
- rca
- worker
- operational
type: skill
---

# RCA Worker Guide

Operational knowledge shared by all RCA worker agents. Load this before your first data query.

## Data Sources

Two observability periods are available:
- **Abnormal** (`abnormal_*` tables): the incident window
- **Normal** (`normal_*` tables): the baseline window

Each period has logs, traces, and metrics. An anomaly only exists if there is a significant delta between the two periods.

## Tooling Quickstart

1. Call `describe_tables` to see all available tables and columns.
2. Call `vault_read` with path `skill/diagnose-sql` to get query rules (column quoting, duration units, correct column names) and a recipe index. **Skipping this causes query errors.**
3. Tables come in pairs: `abnormal_*` (incident) and `normal_*` (baseline).
4. Write SQL via `query_sql`. Always include LIMIT.
5. The `think` tool is always available for structured reasoning between queries.

## Service Profile Protocol

You have access to a shared Service Profile store — use it to avoid redundant work.

- **Before investigating a service**: call `query_service_profile(service_name="X")` to check existing findings. Focus on what's MISSING.
- **During investigation**: call `update_service_profile(...)` to record discoveries. Other parallel agents see your updates immediately.
- **Keep it SHORT** — a profile is a quick-reference card:
  - `anomaly_summary`: one terse line (e.g., "p99 60s vs 4s, 45% errors")
  - `key_observation`: one factual sentence, no reasoning
  - Only record what's NEW

## Tool Discipline

You have a budget of tool calls for this task. Use them wisely.

- **`think` is FREE** — does not count against budget. Use it liberally.
- **Think before querying**: state what you're looking for, which tools to call, and what results would confirm or deny your hypothesis.
- **Call incrementally**: 1-3 tool calls per round, then analyze results before calling more. Do NOT batch all queries at once.
- **Analyze between rounds**: after each result, call `think` to update your mental model and decide the next step.
- **Compare normal vs abnormal**: for every suspected anomaly, run the same query against `normal_*` to quantify the deviation.
- **Correct tool usage**: pass parameters as direct keyword arguments matching the tool signature. Do NOT wrap in `args`, `config`, or nested objects.

## Anomaly Definition

A true anomaly MUST appear in abnormal data but be absent (or negligible) in normal data. Always cross-check against normal-period data before reporting anything as anomalous.

Patterns present in BOTH periods are baseline behavior — exclude them. Healthy services are invisible: if a service shows no significant deviation, do not mention it.
