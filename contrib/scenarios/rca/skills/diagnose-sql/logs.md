---
confidence: fact
description: 'DuckDB recipes for application logs: error scanning, pattern grouping,
  error delta, keyword search, and log-trace correlation joins.'
name: Log Query Recipes
tags:
- sql
- duckdb
- logs
- rca
type: skill
---

# Log Query Recipes

Recipes for querying OpenTelemetry log data in DuckDB.
Tables: `abnormal_logs` (incident) / `normal_logs` (baseline).

Key columns: `time`, `trace_id`, `span_id`, `level`, `service_name`, `message`, `"attr.k8s.pod.name"`.

Levels in order of severity: `DEBUG`, `INFO`, `WARN`, `ERROR`. Focus on `ERROR` and `WARN` for RCA.

## Error log scan (first pass)

```sql
SELECT time, level, service_name,
       left(message, 200) AS msg_preview
FROM abnormal_logs
WHERE level IN ('ERROR', 'WARN')
ORDER BY time
LIMIT 50
```

## Error pattern grouping

Instead of reading individual logs, group by pattern to find dominant errors.
```sql
SELECT service_name,
       left(message, 120) AS pattern,
       count(*) AS cnt
FROM abnormal_logs
WHERE level = 'ERROR'
GROUP BY service_name, pattern
ORDER BY cnt DESC
LIMIT 20
```

## Error delta (abnormal vs normal)

```sql
WITH abn AS (
  SELECT service_name, count(*) AS error_cnt
  FROM abnormal_logs WHERE level = 'ERROR'
  GROUP BY service_name
), nml AS (
  SELECT service_name, count(*) AS error_cnt
  FROM normal_logs WHERE level = 'ERROR'
  GROUP BY service_name
)
SELECT coalesce(a.service_name, n.service_name) AS service,
       coalesce(a.error_cnt, 0) AS abn_errors,
       coalesce(n.error_cnt, 0) AS nml_errors,
       coalesce(a.error_cnt, 0) - coalesce(n.error_cnt, 0) AS delta
FROM abn a
FULL OUTER JOIN nml n USING (service_name)
ORDER BY delta DESC
LIMIT 20
```

## Keyword search in logs

```sql
SELECT time, service_name, level, message
FROM abnormal_logs
WHERE message ILIKE '%timeout%'
   OR message ILIKE '%connection refused%'
   OR message ILIKE '%pool exhausted%'
ORDER BY time
LIMIT 30
```
Common fault keywords: `timeout`, `refused`, `reset`, `broken pipe`, `OOM`, `pool exhausted`, `deadlock`, `too many connections`, `circuit breaker`.

## Log-trace correlation

When you find a suspicious log, join with traces via trace_id to see the full request context.
```sql
SELECT l.time, l.service_name, l.level, left(l.message, 150) AS msg,
       t.span_name, round(t.duration/1e6, 2) AS span_ms,
       "attr.http.response.status_code" AS http_status
FROM abnormal_logs l
JOIN abnormal_traces t
  ON l.trace_id = t.trace_id AND l.span_id = t.span_id
WHERE l.level = 'ERROR'
ORDER BY l.time
LIMIT 30
```
