---
confidence: fact
description: 'Strategies for triangulating trace, log, and metric signals: choosing
  the right entry point, upstream-vs-downstream causality, network fault detection,
  and cross-signal reasoning.'
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

No single signal is always the right starting point. Choose based on what you already know.

## Choosing your entry signal

**Your lead is "service X has high latency"** → start with traces
1. Trace topology → identify which edges/services are anomalous
2. Child span breakdown → where is time spent? Use the fan-out aware attribution query
   (traces.md "Internal vs downstream attribution") to correctly separate internal vs downstream time.
3. If no slow child spans found → do NOT conclude "internal processing bottleneck."
   Absence of child spans means traces cannot explain the latency. Switch to metrics
   (resource scan) and logs (error scan) for this service.

**Your lead is "service X has resource anomaly (CPU/memory/GC)"** → start with metrics
1. Full metric scan for the service (see metrics.md Step 1)
2. Read the compound signals together (see metrics.md Domain Knowledge)
3. Then check traces: does this resource anomaly correlate with latency tail (p99)?
4. Then check logs: do error logs mention the resource problem (OOM, GC overhead, pool exhaustion)?

**Your lead is "service X has errors"** → start with logs
1. Error pattern grouping → what are the dominant error messages?
2. Then check traces: which request paths trigger the errors?
3. Then check metrics: is a resource issue driving the errors?

**You have no clear lead yet** → start broad
1. Trace latency delta across all services → find highest-deviation services
2. Error rate delta across all services → find services with new errors
3. Call volume delta across all services → find services with traffic changes
4. For EACH anomalous service (in ANY dimension), run the metric full scan
5. A service is only "healthy" when ALL four dimensions (latency, error rate, call volume,
   resources) have been checked and show no deviation. Until then, it is "partially observed."

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

**Important caveat**: this recipe uses average latency. If the suspect has a resource problem
that causes intermittent stalls (GC pauses, page faults), average latency may look normal
while p99 is elevated. When resource metrics are anomalous, also check latency percentiles:
```sql
SELECT service_name,
       round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY duration)/1e6, 2) AS p50_ms,
       round(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration)/1e6, 2) AS p99_ms,
       count(*) AS cnt
FROM abnormal_traces
WHERE service_name = 'SUSPECT_SERVICE'
GROUP BY service_name
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

## Recipe: asymmetric error detection (link-level faults)

When service A has errors but its downstream service B does not, the fault may be in
the network link between them — not in either service. This recipe detects such asymmetry.

```sql
-- Step 1: Find which downstream services the caller's error spans are targeting
-- (join caller error spans with their child spans to see which callee was being called)
WITH caller_errors AS (
  SELECT trace_id, span_id, span_name, duration
  FROM abnormal_traces
  WHERE service_name = 'CALLER_SERVICE'
    AND ("attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
         OR "attr.http.response.status_code" >= 400)
)
SELECT child.service_name AS target_callee,
       count(*) AS error_calls,
       round(avg(child.duration)/1e6, 2) AS callee_avg_ms,
       count(*) FILTER (WHERE child."attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
                         OR child."attr.http.response.status_code" >= 400) AS callee_errors
FROM caller_errors ce
JOIN abnormal_traces child
  ON child.trace_id = ce.trace_id
 AND child.parent_span_id = ce.span_id
GROUP BY child.service_name
ORDER BY error_calls DESC
```

Interpretation:
- `callee_errors` = 0 but `error_calls` > 0 → callee succeeded but caller still failed = link fault
- `callee_errors` ≈ `error_calls` → callee is actually failing = callee is the problem

```sql
-- Step 2: Check error span status codes for the caller
-- null status_code = connection-level failure (abort/reset/timeout), not application error
SELECT "attr.http.response.status_code" AS status_code,
       "attr.status_code" AS span_status,
       count(*) AS cnt
FROM abnormal_traces
WHERE service_name = 'CALLER_SERVICE'
  AND ("attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
       OR "attr.http.response.status_code" >= 400
       OR "attr.http.response.status_code" IS NULL)
GROUP BY status_code, span_status
ORDER BY cnt DESC LIMIT 20
```

Interpretation:
- Many `status_code = null` with `span_status = Error` → connection severed before response
- `status_code = 502/503` → upstream proxy/LB error, callee may be unreachable
- `status_code = 500` → application error (callee returned an error response)

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

## Cross-signal reasoning principles

**Traces and metrics can tell different stories — and both can be right.**
A service with normal trace latency but abnormal resource metrics is not healthy — it may be
causing intermittent problems that don't show in averages. Always consider whether the resource
anomaly could be causing the symptoms you see in OTHER services' traces.

**Absence of trace evidence is not evidence of absence.**
When traces show a latency gap (time unaccounted for by child spans), the explanation
lives outside the trace data — in metrics (resource stalls), logs (errors, retries),
or infrastructure (network, scheduling). Do not fill the gap with speculation;
switch signals and investigate.

**The strongest causal argument combines all three signals.**
- Trace: shows WHERE the latency lives (which service, which edge)
- Metric: shows WHAT resource is stressed (CPU, memory, connections)
- Log: shows the specific MECHANISM (error message, exception, timeout)
If you can connect all three for the same service in the same timeframe, you have strong evidence.
If you only have one signal, flag it as a lead, not a conclusion.

**Asymmetric errors point to link-level faults.**
When service A has errors calling service B, but B shows 0% error rate, do NOT conclude
"A has an internal problem" or "all downstream services are healthy." The fault is likely
in the link between them — network, proxy, or infrastructure. Use the asymmetric error
detection recipe above to confirm. The combination of: (1) caller errors with null status
codes, (2) callee showing 0% errors, and (3) no application-level error logs in the caller
is the classic fingerprint of a network-level fault (connection abort, packet drop, TCP reset).
