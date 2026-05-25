---
confidence: fact
description: 'DuckDB query patterns for observability parquets: efficient aggregation techniques and how to combine them to minimise round-trips.'
name: query-patterns
tags:
- verifier
- sql
- duckdb
- query
type: skill
trigger_patterns:
- query_sql
- list_tables
---

# Query Patterns

Each `query_sql` call costs budget. The patterns below let you answer
more questions per call. They are composable — combine freely.

## 1. Side-by-side window comparison

Compare normal and abnormal in ONE query. Never run two separate queries
for the two windows.

```sql
SELECT 'normal' AS win, <aggregates>
FROM normal_traces WHERE <filter>
UNION ALL
SELECT 'abnormal' AS win, <aggregates>
FROM abnormal_traces WHERE <filter>
```

## 2. Multi-service in one pass

Query several services at once with `GROUP BY`, not one query per
service.

```sql
SELECT service_name, COUNT(*) AS cnt, AVG(duration) AS avg_dur
FROM abnormal_traces
WHERE service_name IN ('svcA','svcB','svcC')
GROUP BY service_name
```

Combine with pattern 1 (window + multi-service):

```sql
SELECT 'normal' AS win, service_name, COUNT(*) AS cnt, AVG(duration) AS avg
FROM normal_traces
WHERE service_name IN ('svcA','svcB','svcC')
GROUP BY service_name
UNION ALL
SELECT 'abnormal' AS win, service_name, COUNT(*) AS cnt, AVG(duration) AS avg
FROM abnormal_traces
WHERE service_name IN ('svcA','svcB','svcC')
GROUP BY service_name
ORDER BY service_name, win
```

## 3. Full distribution in one query

Never scan thresholds one by one (`duration > 50ms`, then `> 100ms`,
then `> 500ms` …). Use quantiles or bucketed counts in a single pass:

```sql
SELECT service_name,
       COUNT(*) AS total,
       APPROX_QUANTILE(duration, 0.5)  AS p50,
       APPROX_QUANTILE(duration, 0.9)  AS p90,
       APPROX_QUANTILE(duration, 0.95) AS p95,
       APPROX_QUANTILE(duration, 0.99) AS p99,
       MAX(duration) AS max_dur
FROM abnormal_traces
WHERE service_name IN ('svcA','svcB')
GROUP BY service_name
```

If you need bucket counts (e.g. how many spans exceed various
thresholds), do them all in one row:

```sql
SELECT service_name,
       COUNT(*) AS total,
       SUM(CASE WHEN duration > 100000000  THEN 1 ELSE 0 END) AS gt100ms,
       SUM(CASE WHEN duration > 1000000000 THEN 1 ELSE 0 END) AS gt1s,
       SUM(CASE WHEN duration > 5000000000 THEN 1 ELSE 0 END) AS gt5s
FROM abnormal_traces
WHERE service_name IN ('svcA','svcB')
GROUP BY service_name
```

## 4. Call-graph discovery

Find all direct caller→callee pairs in one query (join on
`parent_span_id`). Repeat for both windows to see what changed:

```sql
SELECT 'normal' AS win,
       parent.service_name AS caller,
       child.service_name  AS callee,
       COUNT(*) AS calls
FROM normal_traces child
JOIN normal_traces parent
  ON child.parent_span_id = parent.span_id
WHERE child.service_name <> parent.service_name
GROUP BY parent.service_name, child.service_name
UNION ALL
SELECT 'abnormal' AS win,
       parent.service_name AS caller,
       child.service_name  AS callee,
       COUNT(*) AS calls
FROM abnormal_traces child
JOIN abnormal_traces parent
  ON child.parent_span_id = parent.span_id
WHERE child.service_name <> parent.service_name
GROUP BY parent.service_name, child.service_name
ORDER BY callee, caller, win
```

To verify a SPECIFIC edge (e.g. svcA calls svcB), filter both sides:

```sql
SELECT 'normal' AS win, COUNT(*) AS calls
FROM normal_traces child
JOIN normal_traces parent
  ON child.parent_span_id = parent.span_id
WHERE parent.service_name = 'svcA'
  AND child.service_name  = 'svcB'
UNION ALL
SELECT 'abnormal' AS win, COUNT(*) AS calls
FROM abnormal_traces child
JOIN abnormal_traces parent
  ON child.parent_span_id = parent.span_id
WHERE parent.service_name = 'svcA'
  AND child.service_name  = 'svcB'
```

## 5. Multi-signal in one query

Combine throughput, latency, and errors from the same table:

```sql
SELECT 'normal' AS win, service_name,
       COUNT(*) AS spans,
       AVG(duration) AS avg_dur,
       APPROX_QUANTILE(duration, 0.99) AS p99,
       SUM(CASE WHEN "attr.status_code" = 'Error' THEN 1 ELSE 0 END) AS errors
FROM normal_traces
WHERE service_name IN ('svcA','svcB')
GROUP BY service_name
UNION ALL
SELECT 'abnormal' AS win, service_name,
       COUNT(*) AS spans,
       AVG(duration) AS avg_dur,
       APPROX_QUANTILE(duration, 0.99) AS p99,
       SUM(CASE WHEN "attr.status_code" = 'Error' THEN 1 ELSE 0 END) AS errors
FROM abnormal_traces
WHERE service_name IN ('svcA','svcB')
GROUP BY service_name
ORDER BY service_name, win
```

## 6. Metric tables — pivot multiple metrics

Metrics tables have a `metric` column. Pivot in one query instead of
querying one metric at a time:

```sql
SELECT 'normal' AS win, service_name, metric, AVG(value) AS avg_val
FROM normal_metrics
WHERE service_name IN ('svcA','svcB')
  AND metric IN ('container.cpu.usage','container.memory.usage')
GROUP BY service_name, metric
UNION ALL
SELECT 'abnormal' AS win, service_name, metric, AVG(value) AS avg_val
FROM abnormal_metrics
WHERE service_name IN ('svcA','svcB')
  AND metric IN ('container.cpu.usage','container.memory.usage')
GROUP BY service_name, metric
ORDER BY service_name, metric, win
```

## Composing a plan

A typical investigation needs ~5-10 well-composed queries, not 100+
narrow ones:

1. **Overview** — pattern 5 (multi-signal) across all candidate
   services, both windows → one query, shows who changed.
2. **Call graph** — pattern 4 (full graph discovery) → one query, shows
   the topology.
3. **Metrics survey** — pattern 6 (pivot) for resource and app metrics →
   one or two queries.
4. **Targeted deep-dive** — pattern 3 (distribution) for services that
   look anomalous → one query per signal type.
5. **Edge verification** — pattern 4 (specific edge) for each candidate
   edge → one query each.

If you find yourself writing a query you have already run, stop — the
answer is already in your context. Re-read it instead of re-querying.
