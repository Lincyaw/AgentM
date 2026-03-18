---
confidence: fact
description: 'DuckDB recipes for infrastructure and application metrics: CPU/memory,
  JVM GC/heap, DB connection pools, network (Hubble/Cilium), HTTP latency percentiles,
  and metric delta patterns.'
name: Metric Query Recipes
tags:
- sql
- duckdb
- metrics
- rca
type: skill
---

# Metric Query Recipes

Recipes for querying OpenTelemetry metric data in DuckDB.

Three metric tables per period — each has different structure:

| Table | Type | Value columns | Use for |
|-------|------|--------------|---------|
| `abnormal_metrics` | Gauge | `value` | Point-in-time: CPU, memory, pod status, queue sizes |
| `abnormal_metrics_sum` | Sum/Counter | `value` | Cumulative: connection counts, request totals, byte counts |
| `abnormal_metrics_histogram` | Histogram | `count`, `sum`, `min`, `max` | Distributions: request duration, GC duration, connection times |

Common columns: `time`, `metric`, `service_name`, `"attr.k8s.pod.name"`, `"attr.k8s.container.name"`.

**Important**: The column is `metric`, not `metric_name`.

## Discover available metrics

Always check what's available before querying. Different environments export different metrics.
```sql
SELECT DISTINCT metric FROM abnormal_metrics ORDER BY metric
```
Repeat for `abnormal_metrics_sum` and `abnormal_metrics_histogram`.

## Resource usage by pod (CPU, memory)

```sql
SELECT "attr.k8s.pod.name" AS pod,
       metric,
       round(avg(value), 4)  AS avg_val,
       round(max(value), 4)  AS max_val
FROM abnormal_metrics
WHERE metric IN ('container.cpu.usage', 'container.memory.usage',
                 'container.memory.working_set')
  AND "attr.k8s.pod.name" ILIKE '%TARGET%'
GROUP BY pod, metric
ORDER BY avg_val DESC
LIMIT 20
```

## JVM metrics (GC, threads, heap)

```sql
-- GC duration from histogram
SELECT service_name,
       "attr.jvm.gc.action" AS gc_action,
       round(sum / nullif(count, 0), 4) AS avg_gc_sec,
       max AS max_gc_sec,
       count AS gc_events
FROM abnormal_metrics_histogram
WHERE metric = 'jvm.gc.duration'
ORDER BY avg_gc_sec DESC
LIMIT 15
```

```sql
-- Heap usage from sum metrics
SELECT service_name, metric,
       round(avg(value) / 1e6, 2) AS avg_mb
FROM abnormal_metrics_sum
WHERE metric IN ('jvm.memory.used', 'jvm.memory.committed', 'jvm.memory.limit')
GROUP BY service_name, metric
ORDER BY avg_mb DESC
LIMIT 20
```

```sql
-- Thread count
SELECT service_name, round(avg(value), 0) AS avg_threads
FROM abnormal_metrics_sum
WHERE metric = 'jvm.thread.count'
GROUP BY service_name
ORDER BY avg_threads DESC
```

## DB connection pool health

```sql
-- Pool usage and pending requests
SELECT service_name, metric, round(avg(value), 2) AS avg_val
FROM abnormal_metrics_sum
WHERE metric LIKE 'db.client.connections%'
GROUP BY service_name, metric
ORDER BY service_name, metric
LIMIT 30
```

```sql
-- Connection timing from histogram (create, wait, use)
SELECT service_name, metric,
       round(sum / nullif(count, 0), 2) AS avg_ms,
       round(max, 2) AS max_ms
FROM abnormal_metrics_histogram
WHERE metric LIKE 'db.client.connections%'
ORDER BY avg_ms DESC
LIMIT 15
```

Key signals:
- `create_time` spike → DB unreachable or slow DNS
- `wait_time` spike → pool exhaustion (all connections busy)
- `use_time` spike → slow queries or transactions

## Network-level metrics (Hubble/Cilium)

```sql
SELECT "attr.source_workload" AS src,
       "attr.destination_workload" AS dst,
       metric,
       round(avg(value), 2) AS avg_val
FROM abnormal_metrics_sum
WHERE metric LIKE 'hubble%'
  AND "attr.source_workload" IS NOT NULL
GROUP BY src, dst, metric
ORDER BY avg_val DESC
LIMIT 20
```

## HTTP latency percentiles (Hubble gauges)

```sql
SELECT service_name,
       max(CASE WHEN metric = 'hubble_http_request_duration_p50_seconds' THEN value END) AS p50,
       max(CASE WHEN metric = 'hubble_http_request_duration_p90_seconds' THEN value END) AS p90,
       max(CASE WHEN metric = 'hubble_http_request_duration_p99_seconds' THEN value END) AS p99
FROM abnormal_metrics
WHERE metric LIKE 'hubble_http_request_duration_p%'
GROUP BY service_name
ORDER BY p99 DESC NULLS LAST
LIMIT 20
```

## Metric delta between periods

```sql
WITH abn AS (
  SELECT "attr.k8s.pod.name" AS pod, metric, round(avg(value), 4) AS val
  FROM abnormal_metrics
  WHERE metric = 'container.cpu.usage'
  GROUP BY pod, metric
), nml AS (
  SELECT "attr.k8s.pod.name" AS pod, metric, round(avg(value), 4) AS val
  FROM normal_metrics
  WHERE metric = 'container.cpu.usage'
  GROUP BY pod, metric
)
SELECT coalesce(a.pod, n.pod) AS pod,
       a.val AS abn_val, n.val AS nml_val,
       round(a.val / nullif(n.val, 0), 2) AS ratio
FROM abn a FULL OUTER JOIN nml n USING (pod, metric)
ORDER BY ratio DESC NULLS FIRST
LIMIT 20
```
