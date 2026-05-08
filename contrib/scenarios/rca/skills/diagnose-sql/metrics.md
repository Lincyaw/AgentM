---
confidence: fact
description: 'DuckDB recipes for infrastructure and application metrics. Organized as:
  mandatory full-scan first, then targeted drill-down recipes, with domain knowledge
  for interpreting compound signals.'
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

## Step 1: Full metric scan (MANDATORY for every investigated service)

Do NOT pre-select which metrics to query. Let the data show you what's anomalous.

```sql
-- Gauge metrics: full delta ranking for a service
SELECT metric,
       round(avg(a.value), 4) AS abn_avg,
       round(avg(n.value), 4) AS nml_avg,
       round(avg(a.value) / nullif(avg(n.value), 0), 2) AS ratio
FROM abnormal_metrics a
FULL OUTER JOIN normal_metrics n USING (metric, service_name)
WHERE a.service_name = 'TARGET' OR n.service_name = 'TARGET'
GROUP BY metric
ORDER BY ratio DESC NULLS FIRST
LIMIT 30
```

If the query above fails due to row-level FULL JOIN on unpaired tables, use this alternative:
```sql
WITH abn AS (
  SELECT metric, round(avg(value), 4) AS val
  FROM abnormal_metrics WHERE service_name = 'TARGET'
  GROUP BY metric
), nml AS (
  SELECT metric, round(avg(value), 4) AS val
  FROM normal_metrics WHERE service_name = 'TARGET'
  GROUP BY metric
)
SELECT coalesce(a.metric, n.metric) AS metric,
       a.val AS abn_avg, n.val AS nml_avg,
       round(a.val / nullif(n.val, 0), 2) AS ratio
FROM abn a FULL OUTER JOIN nml n USING (metric)
ORDER BY ratio DESC NULLS FIRST
LIMIT 30
```

After the scan, use `think` to answer:
1. Which metrics have the largest deviation (both increases AND decreases)?
2. Do any of the top deviations relate to each other? (see Domain Knowledge below)
3. Can these resource signals explain the latency or error symptoms being investigated?

## Step 2: Targeted drill-down recipes

Use these AFTER Step 1 identifies which area to investigate deeper.

### JVM GC duration (histogram)

```sql
-- GC duration delta between periods
WITH abn AS (
  SELECT "attr.jvm.gc.action" AS action,
         round(sum / nullif(count, 0), 4) AS avg_sec,
         count AS events
  FROM abnormal_metrics_histogram
  WHERE metric = 'jvm.gc.duration' AND service_name = 'TARGET'
), nml AS (
  SELECT "attr.jvm.gc.action" AS action,
         round(sum / nullif(count, 0), 4) AS avg_sec,
         count AS events
  FROM normal_metrics_histogram
  WHERE metric = 'jvm.gc.duration' AND service_name = 'TARGET'
)
SELECT coalesce(a.action, n.action) AS gc_action,
       a.avg_sec AS abn_avg, n.avg_sec AS nml_avg,
       a.events AS abn_events, n.events AS nml_events
FROM abn a FULL OUTER JOIN nml n USING (action)
```

### JVM heap and threads (sum metrics)

```sql
WITH abn AS (
  SELECT metric, round(avg(value), 2) AS val
  FROM abnormal_metrics_sum
  WHERE service_name = 'TARGET'
    AND metric IN ('jvm.memory.used', 'jvm.memory.committed',
                   'jvm.memory.limit', 'jvm.thread.count')
  GROUP BY metric
), nml AS (
  SELECT metric, round(avg(value), 2) AS val
  FROM normal_metrics_sum
  WHERE service_name = 'TARGET'
    AND metric IN ('jvm.memory.used', 'jvm.memory.committed',
                   'jvm.memory.limit', 'jvm.thread.count')
  GROUP BY metric
)
SELECT coalesce(a.metric, n.metric) AS metric,
       a.val AS abn_val, n.val AS nml_val,
       round(a.val / nullif(n.val, 0), 2) AS ratio
FROM abn a FULL OUTER JOIN nml n USING (metric)
ORDER BY ratio DESC NULLS FIRST
```

### DB connection pool health

```sql
-- Pool usage and pending requests (delta)
WITH abn AS (
  SELECT metric, round(avg(value), 2) AS val
  FROM abnormal_metrics_sum
  WHERE metric LIKE 'db.client.connections%' AND service_name = 'TARGET'
  GROUP BY metric
), nml AS (
  SELECT metric, round(avg(value), 2) AS val
  FROM normal_metrics_sum
  WHERE metric LIKE 'db.client.connections%' AND service_name = 'TARGET'
  GROUP BY metric
)
SELECT coalesce(a.metric, n.metric) AS metric,
       a.val AS abn_val, n.val AS nml_val,
       round(a.val / nullif(n.val, 0), 2) AS ratio
FROM abn a FULL OUTER JOIN nml n USING (metric)
ORDER BY ratio DESC NULLS FIRST
LIMIT 20
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

### Network-level metrics (Hubble/Cilium)

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

### HTTP latency percentiles (Hubble gauges)

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

## Domain Knowledge: interpreting compound signals

Individual metrics can mislead. The diagnostic value comes from reading metrics TOGETHER.

**CPU and memory are linked through GC.** In JVM services, high CPU often comes from garbage
collection, not application logic. When you see elevated CPU, always check GC metrics before
concluding "CPU saturation." Frequent GC keeps heap usage LOW (because it keeps reclaiming
memory) while burning CPU — so low `jvm.memory.used` alongside high CPU is a signal of memory
pressure, not a sign of health.

**page_faults reveal what memory.usage hides.** `container.memory.usage` shows how much memory
the OS has allocated to the container. It can appear stable even under severe memory pressure —
the OS pages memory in and out instead of growing the allocation. `page_faults` (both
`container.memory.page_faults` and `k8s.pod.memory.page_faults`) expose this hidden thrashing.

**Average latency hides tail problems.** Resource exhaustion (GC pauses, pool waits, page faults)
typically causes intermittent stalls — a few requests get hit hard while most are fine.
This shows as normal avg latency but elevated p99/p999. If a service has resource anomalies
but normal avg latency, check latency percentiles before ruling it out.

**DB connection pool signals have a specific causal order:**
- `create_time` spike → DB unreachable or slow DNS
- `wait_time` spike → pool exhaustion (all connections busy)
- `use_time` spike → slow queries or long-held transactions

**Absence can be a signal.** A metric present in normal but missing in abnormal (or vice versa)
may indicate a pod restart, deployment change, or crash. Null ratios in the full scan
deserve attention, not just high ratios.
