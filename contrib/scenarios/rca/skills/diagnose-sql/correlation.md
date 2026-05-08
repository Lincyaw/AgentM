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

## Understanding network and link-level faults

Network faults between services come in many forms. Recognize the pattern by its observability signature:

**Pattern A: Connection-level failure** — caller error spans with null HTTP status, callee has
no corresponding span. Network metrics: hubble_drop_total elevated, TCP reset flags.

**Pattern B: Response abort** — callee span shows SUCCESS but caller gets error (asymmetric).
Callee processed the request; the response was lost in transit.

**Pattern C: HTTP-level selective fault** — specific HTTP errors (502/503) only on certain
paths/endpoints, other requests to the same callee succeed. May be proxy/sidecar injecting errors.

**Pattern D: Latency injection** — elevated latency between two services, no errors, no resource
anomalies on either side. The delay is added in transit, not by either service's processing.

**Network faults are path-dependent**: they affect a network path (node1→node2), not a specific
caller. Check `attr.k8s.node.name` for pod placement and correlate with hubble metrics.

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
- `callee_errors` = 0 but `error_calls` > 0 → callee succeeded but caller still failed = link fault (Pattern B)
- `callee_errors` ≈ `error_calls` → callee is actually failing = callee is the problem
- no child spans found for error calls → request never reached callee = connection-level fault (Pattern A)

```sql
-- Step 2: Check error span status codes for the caller
-- Classify errors by status code to identify the fault pattern
SELECT "attr.http.response.status_code" AS http_code,
       "attr.status_code" AS span_status,
       count(*) AS cnt,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM abnormal_traces
WHERE service_name = 'CALLER_SERVICE'
  AND ("attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
       OR "attr.http.response.status_code" >= 400)
GROUP BY http_code, span_status
ORDER BY cnt DESC LIMIT 20
```

Interpretation:
- `http_code = null` + `span_status = Error` → connection severed before response (Pattern A/B)
- `http_code = 502/503` → proxy/LB returned error, callee may be unreachable
- `http_code = 500` → application error in callee
- Mix of null and 500 → possibly both link-level and application-level failures co-occurring

```sql
-- Step 3: Check if the error is path-selective
-- (does the error only affect calls to specific endpoints/paths?)
SELECT span_name,
       count(*) AS total,
       count(*) FILTER (WHERE "attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
                         OR "attr.http.response.status_code" >= 400) AS errors,
       round(100.0 * count(*) FILTER (WHERE "attr.status_code" IN ('Error', 'STATUS_CODE_ERROR')
                         OR "attr.http.response.status_code" >= 400) / nullif(count(*), 0), 1) AS error_pct
FROM abnormal_traces
WHERE service_name = 'CALLER_SERVICE'
GROUP BY span_name
ORDER BY error_pct DESC LIMIT 20
```

Interpretation:
- All span_names have similar error_pct → fault is indiscriminate (TCP-level, Pattern A)
- Only specific span_names have errors → fault is selective (HTTP-level, Pattern C)

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

**Traces and metrics can disagree — trust metrics for resource state, traces for call flow.**
A service with normal latency but abnormal resources (6x CPU, GC storms) is NOT healthy — it
may be partially failing. Successful requests are fast; failing requests show up as timeouts
in the caller, not as high latency in the callee. When resource anomalies and latency disagree,
check callers' error patterns and call volume changes before eliminating the service.

**Missing trace evidence points to infrastructure faults, not healthy services.**
When a caller span normally has a child span to a downstream service but the child vanishes in
the abnormal period, the downstream is unreachable — not "uninvolved." The caller's inflated
"internal time" is an artifact. The attribution recipe (traces.md) shows this as `fan_out`
dropping between periods.

**The strongest causal argument combines all three signals.**
- Trace: WHERE the latency lives (which service, which edge)
- Metric: WHAT resource is stressed (CPU, memory, connections)
- Log: the specific MECHANISM (error message, exception, timeout)
One signal is a lead. Three signals for the same service in the same timeframe is strong evidence.

**Asymmetric errors point to link-level faults.**
Caller has errors calling callee, but callee shows 0% error rate → fault is in the link.
See the asymmetric error detection recipe above.
