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

**CRITICAL: Do NOT add WHERE service_name IN (...) to this query.** Run it unfiltered.

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

**CRITICAL: Do NOT add WHERE service_name IN (...) to this query.** Run it unfiltered against
ALL services. A service you think is healthy may have a high error rate that you haven't checked.
Filtering to a pre-selected list is the #1 cause of missed root causes in RCA.

```sql
WITH abn AS (
  SELECT service_name,
         count(*) AS total,
         count(*) FILTER (WHERE "attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
                          OR "attr.http.response.status_code" >= 400) AS errors
  FROM abnormal_traces GROUP BY service_name
), nml AS (
  SELECT service_name,
         count(*) AS total,
         count(*) FILTER (WHERE "attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
                          OR "attr.http.response.status_code" >= 400) AS errors
  FROM normal_traces GROUP BY service_name
)
SELECT coalesce(a.service_name, n.service_name) AS service_name,
       round(100.0 * a.errors / nullif(a.total, 0), 2) AS abn_error_pct,
       round(100.0 * n.errors / nullif(n.total, 0), 2) AS nml_error_pct,
       a.errors AS abn_errors, a.total AS abn_total,
       n.total AS nml_total
FROM abn a
FULL OUTER JOIN nml n USING (service_name)
ORDER BY abn_error_pct DESC NULLS LAST
LIMIT 30
```
This returns ALL services with their error rates in both periods — no pre-filtering.

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

## Internal vs downstream attribution (fan-out aware)

When determining how much of a service's latency is "internal processing" vs "waiting on
downstream calls," you MUST sum ALL child span durations — not compare a single child to the
parent. A parent span with 7 sequential children, each taking 8s, has ~0% internal time even
though each individual child (8s) is shorter than the parent (56s).

```sql
WITH parent_spans AS (
  SELECT trace_id, span_id, duration AS parent_dur
  FROM abnormal_traces
  WHERE service_name = 'TARGET_SERVICE'
    AND span_name = 'TARGET_SPAN'  -- optional: filter to specific endpoint
),
child_totals AS (
  SELECT p.trace_id, p.span_id,
         p.parent_dur,
         coalesce(sum(c.duration), 0) AS total_child_dur,
         count(c.span_id) AS child_count
  FROM parent_spans p
  LEFT JOIN abnormal_traces c
    ON c.trace_id = p.trace_id
   AND c.parent_span_id = p.span_id
  GROUP BY p.trace_id, p.span_id, p.parent_dur
)
SELECT round(avg(parent_dur)/1e6, 2)     AS avg_parent_ms,
       round(avg(total_child_dur)/1e6, 2) AS avg_total_child_ms,
       round(avg(parent_dur - total_child_dur)/1e6, 2) AS avg_internal_ms,
       round(100.0 * avg(parent_dur - total_child_dur) / nullif(avg(parent_dur), 0), 1)
         AS internal_pct,
       round(avg(child_count), 1) AS avg_fan_out
FROM child_totals
```

**Interpretation:**
- `internal_pct` > 70% AND `avg_fan_out` ≈ 0 → truly internal (no downstream calls)
- `internal_pct` > 70% AND `avg_fan_out` > 1 → SUSPICIOUS: children may overlap (parallel calls).
  Check if children are sequential or concurrent before concluding "internal bottleneck."
- `internal_pct` < 30% → most time spent in downstream calls; investigate children
- Always compare against the same query on `normal_traces` to see if internal_pct changed.

**Beware of vanishing children.** High `internal_pct` does not always mean the service itself is slow. If the same span type normally has downstream calls (fan_out > 0 in baseline) but those calls disappear in the abnormal period (fan_out drops toward 0), the "internal time" is likely spent waiting for a downstream service that is too overloaded to respond — the child span is missing because the callee never processed the request, not because the caller didn't make the call. Compare fan_out between abnormal and normal periods; a significant drop combined with error spans is a strong signal that the downstream service is unreachable.

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
