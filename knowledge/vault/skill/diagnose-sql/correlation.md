---
confidence: fact
description: 'Strategies for triangulating trace, log, and metric signals: slow-service
  drill-down, upstream-vs-downstream causality, network fault detection.'
name: Cross-Signal Correlation
tags:
- sql
- duckdb
- correlation
- rca
type: skill
---

# Cross-Signal Correlation

Strategies and recipes for combining trace, log, and metric data to establish causality.

## Principle: triangulate with three signals

A single signal can mislead. Strong RCA evidence comes from correlation across signals:
- **Trace** tells you WHAT is slow and WHO calls whom
- **Log** tells you WHY (error messages, exceptions, timeouts)
- **Metric** tells you HOW (resource exhaustion, saturation, limits)

## Strategy: trace-first, log-second, metric-third

1. **Trace topology** → identify which edges/services are anomalous
2. **Log scan** → check if anomalous services emit new errors in the abnormal period
3. **Metric check** → for services with errors/latency, look at resource indicators

Each step narrows the scope. Don't query metrics for all services — only for those flagged by traces and logs.

## Recipe: slow service with errors

```sql
-- Step 1: Find the service with highest latency ratio
-- (use the latency delta query from trace recipes)

-- Step 2: Check its error logs
SELECT time, level, left(message, 200) AS msg
FROM abnormal_logs
WHERE service_name = 'TARGET_SERVICE' AND level IN ('ERROR', 'WARN')
ORDER BY time LIMIT 20

-- Step 3: Check resource metrics for that service's pod
SELECT metric, round(avg(value), 4) AS avg_val
FROM abnormal_metrics
WHERE "attr.k8s.pod.name" ILIKE '%target-service%'
  AND metric IN ('container.cpu.usage', 'container.memory.working_set')
GROUP BY metric

-- Step 4: Check DB connection health if DB-backed
SELECT metric, round(sum/nullif(count,0), 2) AS avg_ms
FROM abnormal_metrics_histogram
WHERE service_name = 'TARGET_SERVICE'
  AND metric LIKE 'db.client%'
```

## Recipe: upstream vs downstream (cause vs victim)

The key question in any RCA: is the suspect service the CAUSE or a VICTIM?

```sql
-- Step 1: For a suspect service, find its upstream callers
SELECT parent.service_name AS upstream,
       round(avg(parent.duration)/1e6, 2) AS upstream_ms,
       round(avg(child.duration)/1e6, 2)  AS downstream_ms,
       count(*) AS calls
FROM abnormal_traces child
JOIN abnormal_traces parent
  ON child.trace_id = parent.trace_id
 AND child.parent_span_id = parent.span_id
WHERE child.service_name = 'SUSPECT_SERVICE'
GROUP BY upstream
ORDER BY upstream_ms DESC LIMIT 10

-- Step 2: Decision logic
-- If upstream is also slow → suspect may be a VICTIM (propagated latency)
-- If upstream is fast but downstream is slow → suspect is the CAUSE

-- Step 3: Check both directions' error logs for confirmation
```

## Recipe: network fault detection

```sql
-- Hubble drop/TCP reset counts between services
SELECT "attr.source_workload" AS src,
       "attr.destination_workload" AS dst,
       metric, round(avg(value), 2) AS avg_val
FROM abnormal_metrics_sum
WHERE metric IN ('hubble_drop_total', 'hubble_tcp_flags_total')
  AND "attr.source_workload" IS NOT NULL
GROUP BY src, dst, metric
HAVING avg_val > 0
ORDER BY avg_val DESC LIMIT 20
```

## Recipe: temporal correlation

When two services show anomalies, check if they are correlated in time:
```sql
-- Bucket by minute and compare patterns
SELECT date_trunc('minute', time) AS minute,
       service_name,
       count(*) AS span_count,
       round(avg(duration)/1e6, 2) AS avg_ms
FROM abnormal_traces
WHERE service_name IN ('SERVICE_A', 'SERVICE_B')
GROUP BY minute, service_name
ORDER BY minute
LIMIT 60
```
If SERVICE_A's latency spike precedes SERVICE_B's by N minutes, A is likely upstream cause.
