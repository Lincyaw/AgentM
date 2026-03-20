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
2. Child span breakdown → where is time spent?
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
2. For EACH anomalous service, run the metric full scan — not just the most latency-anomalous one
3. Log error delta across all services → find new error patterns

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
