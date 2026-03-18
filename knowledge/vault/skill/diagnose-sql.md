---
confidence: fact
description: 'DuckDB query guide for RCA: ground rules (column quoting, duration units,
  delta pattern), common pitfalls, and index of signal-specific recipe collections
  for traces, logs, metrics, and cross-signal correlation.'
name: Diagnostic SQL Cookbook
tags:
- sql
- duckdb
- rca
- traces
- logs
- metrics
trigger_patterns:
- query_sql
- describe_tables
- duckdb
- sql query
type: skill
---

# Diagnostic SQL Cookbook

Guide for querying OpenTelemetry observability data in DuckDB during root cause analysis.
All data comes in **paired tables**: `abnormal_*` (incident window) vs `normal_*` (baseline).
An anomaly only exists if the delta between the two periods is significant.

## Sub-Skills (load on demand)

| Skill path | Signal | Use when |
|------------|--------|----------|
| [[skill/diagnose-sql/traces]] | Traces | Building topology, latency delta, error rates, span drill-down |
| [[skill/diagnose-sql/logs]] | Logs | Error scanning, pattern grouping, keyword search, log-trace joins |
| [[skill/diagnose-sql/metrics]] | Metrics | CPU/memory, JVM, DB pools, network (Hubble), HTTP latency |
| [[skill/diagnose-sql/correlation]] | Cross-signal | Triangulating cause vs victim, multi-signal drill-down |

Use `skill_load` with the path (e.g. `skill/diagnose-sql/traces`) to load the full recipe collection.

## Ground Rules

### Column quoting (CRITICAL)
Columns with dots MUST be double-quoted. Unquoted dotted names cause parse errors.
```sql
-- WRONG
SELECT attr.k8s.pod.name FROM abnormal_traces

-- CORRECT
SELECT "attr.k8s.pod.name" FROM abnormal_traces
```
Always run `describe_tables` first and use the `sql_ref` field — it is already correctly quoted.

### Column name: `metric` not `metric_name`
The metrics tables use the column `metric`, not `metric_name`.

### Duration is in nanoseconds
Trace `duration` is stored as nanoseconds (UBIGINT). Divide for readability:
- `/1e6` → milliseconds
- `/1e9` → seconds

### Always LIMIT your queries
Raw tables can have 100K+ rows. Use `LIMIT`, aggregations, or `WHERE` filters.
The tool enforces a 5000-token output cap — oversized results get truncated.

### Delta = the only truth
A 500ms latency means nothing in isolation. Only the ratio `abnormal / normal` matters.
Always compare against the corresponding `normal_*` table before calling anything anomalous.

### Three metric tables, not one
| Table | Type | Value columns | Use for |
|-------|------|--------------|---------|
| `*_metrics` | Gauge | `value` | Point-in-time: CPU, memory, pod status |
| `*_metrics_sum` | Sum | `value` | Cumulative: connections, request totals |
| `*_metrics_histogram` | Histogram | `count`, `sum`, `min`, `max` | Distributions: durations, latencies |

For histograms, average = `sum / nullif(count, 0)`.

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Unquoted dotted columns | Always double-quote: `"attr.k8s.pod.name"` |
| Using `metric_name` | Column is called `metric` |
| Treating raw duration as ms | `duration` is nanoseconds; divide by 1e6 |
| Reporting anomalies without baseline | Always query `normal_*` table and compute delta |
| SELECT * on large tables | Use aggregations + LIMIT; tool truncates at 5000 tokens |
| Ignoring histogram metrics | `sum/count` = average; `max` = worst case; don't ignore `_histogram` tables |
| Mixing up service_name vs pod name | Metrics use pod-level `"attr.k8s.pod.name"`, traces use `service_name` |

## DuckDB Tips

- **CTEs**: Chain multiple analysis steps with `WITH ... AS`
- **FILTER**: `count(*) FILTER (WHERE condition)` — conditional aggregation without CASE
- **percentile_cont**: `percentile_cont(0.99) WITHIN GROUP (ORDER BY duration)` for P99
- **ILIKE**: Case-insensitive pattern matching for log search
- **regexp_matches**: `WHERE regexp_matches(message, 'pattern')` for regex log search
- **INTERVAL arithmetic**: `WHERE time > TIMESTAMP '2025-01-01' - INTERVAL '5 minutes'`
- **list_agg**: `list_agg(DISTINCT service_name)` to collect unique values in a group
