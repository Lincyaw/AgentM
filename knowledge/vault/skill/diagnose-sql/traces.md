---
confidence: fact
description: 'DuckDB recipes for distributed tracing: service call graph, latency
  delta, error rate delta, child span drill-down, percentile distribution, single
  trace path.'
name: Trace Query Recipes
tags:
- sql
- duckdb
- traces
- rca
type: skill
---

# Trace Query Recipes

Recipes for querying OpenTelemetry span data in DuckDB.
Tables: `abnormal_traces` (incident) / `normal_traces` (baseline).

Key columns: `trace_id`, `span_id`, `parent_span_id`, `service_name`, `span_name`, `duration` (nanoseconds), `"attr.status_code"`, `"attr.http.response.status_code"`, `"attr.k8s.pod.name"`.

## Service call graph (topology map)

The foundation of any RCA — who calls whom, how often, how fast.
```sql
SELECT parent.service_name AS caller,
       child.service_name  AS callee,
       count(*)            AS calls,
       round(avg(child.duration)/1e6, 2) AS avg_ms
FROM abnormal_traces child
JOIN abnormal_traces parent
  ON child.trace_id = parent.trace_id
 AND child.parent_span_id = parent.span_id
WHERE parent.service_name != child.service_name
GROUP BY caller, callee
ORDER BY calls DESC
LIMIT 50
```
Run on both `abnormal_traces` and `normal_traces` to compare topology changes (disappeared or new edges indicate failures).

## Latency delta by service

Side-by-side comparison to find which services deviate most.
```sql
WITH abn AS (
  SELECT service_name,
         round(avg(duration)/1e6, 2) AS avg_ms,
         count(*) AS cnt
  FROM abnormal_traces GROUP BY service_name
), nml AS (
  SELECT service_name,
         round(avg(duration)/1e6, 2) AS avg_ms,
         count(*) AS cnt
  FROM normal_traces GROUP BY service_name
)
SELECT coalesce(a.service_name, n.service_name) AS service,
       a.avg_ms  AS abn_ms,
       n.avg_ms  AS nml_ms,
       round(a.avg_ms / nullif(n.avg_ms, 0), 1) AS ratio,
       a.cnt AS abn_calls,
       n.cnt AS nml_calls
FROM abn a
FULL OUTER JOIN nml n USING (service_name)
ORDER BY ratio DESC NULLS FIRST
LIMIT 30
```

## Error rate delta by service

```sql
WITH calc AS (
  SELECT service_name, source,
         count(*) AS total,
         count(*) FILTER (WHERE "attr.status_code" = 'Error'
                          OR "attr.http.response.status_code" >= 500) AS errors
  FROM {TABLE} GROUP BY service_name, source
)
SELECT service_name,
       total,
       errors,
       round(100.0 * errors / nullif(total, 0), 2) AS error_pct
FROM calc
WHERE errors > 0
ORDER BY error_pct DESC
LIMIT 20
```
Replace `{TABLE}` with `abnormal_traces` then `normal_traces`. Compare `error_pct`.

## Drill into a slow service (child span breakdown)

When you know which service is slow, find WHERE inside it the time is spent.
```sql
SELECT child.service_name,
       child.span_name,
       round(avg(child.duration)/1e6, 2) AS avg_ms,
       count(*) AS cnt,
       round(max(child.duration)/1e6, 2) AS max_ms
FROM abnormal_traces child
JOIN abnormal_traces parent
  ON child.trace_id = parent.trace_id
 AND child.parent_span_id = parent.span_id
WHERE parent.service_name = 'TARGET_SERVICE'
GROUP BY child.service_name, child.span_name
ORDER BY avg_ms DESC
LIMIT 20
```

## Latency distribution (percentiles)

```sql
SELECT service_name,
       round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY duration)/1e6, 2) AS p50_ms,
       round(percentile_cont(0.90) WITHIN GROUP (ORDER BY duration)/1e6, 2) AS p90_ms,
       round(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration)/1e6, 2) AS p99_ms,
       count(*) AS cnt
FROM abnormal_traces
WHERE service_name = 'TARGET_SERVICE'
GROUP BY service_name
```

## Trace path: follow a single slow trace end-to-end

```sql
WITH slow_trace AS (
  SELECT trace_id FROM abnormal_traces
  WHERE service_name = 'TARGET_SERVICE'
  ORDER BY duration DESC LIMIT 1
)
SELECT t.service_name, t.span_name,
       round(t.duration/1e6, 2) AS ms,
       t.span_id, t.parent_span_id,
       "attr.status_code", "attr.http.response.status_code"
FROM abnormal_traces t
JOIN slow_trace s ON t.trace_id = s.trace_id
ORDER BY t.time
```
