---
name: verifier-result-analysis
description: >
  Methodology for analyzing verifier batch results on the ops-lite / RCA
  dataset. Use whenever the user asks to check, review, or analyze verifier
  run results — including "看看结果", "分析下", "为什么seed没confirm",
  "这些case啥情况", or any request to interpret fpg_scenario.json /
  run_meta.json / run_summary.jsonl outputs. The core principle: never
  trust any single source (GT labels, seed verdicts, aggregate stats).
  Every judgment must be backed by your own SQL queries against the
  parquet data. Hasty conclusions from aggregate numbers are the #1
  failure mode — always trace the causal chain.
---

# Verifier Result Analysis

You are analyzing fault-propagation verifier results against an
observability dataset (traces, metrics, logs in parquet). Your job is
to form an independent judgment about whether each injection took
effect and propagated — not to rubber-stamp what the seed agent or the
ground truth says.

## Why this matters

Aggregate statistics lie. A span-count drop of 87% looks like a
smoking gun until you check that ALL services dropped 80% because the
load generator got blocked. A seed verdict of "rejected — no latency
increase" looks correct until you discover the caller timed out at 20 s
and the few requests that reached the target were the fast survivors.
GT labels like "silent" look authoritative until you find 503 errors
on the caller side proving the kill worked.

Every claim must survive your own verification against the raw data.

## Analysis protocol

### Level 0: Aggregate summary

Read `run_summary.jsonl`. Count OK / error cases. Group errors by
type. This gives orientation, not conclusions.

### Level 1: Per-case seed verdicts

For each no-seed-confirmed case, read `run_meta.json` and extract
`seed_verdicts` → `{service: {verdict, rationale}}`. Note the
rationale — it tells you what the seed agent checked and what it
concluded. Do NOT stop here.

### Level 2: Target-level data check

For each case of interest, query the parquet data directly with
DuckDB:

```sql
-- Target service: normal vs abnormal
SELECT 'normal' w, COUNT(*) spans,
       p50(duration/1e6) p50_ms, p99(duration/1e6) p99_ms,
       SUM(CASE WHEN "attr.status_code" IN ('STATUS_CODE_ERROR','ERROR')
           THEN 1 ELSE 0 END) errors
FROM normal_traces WHERE service_name = '<target>'
UNION ALL
SELECT 'abnormal', ...
FROM abnormal_traces WHERE service_name = '<target>'
```

If the target shows clear degradation, the seed verdict may be wrong.
If the target looks healthy, **do not conclude yet** — proceed to L3.

### Level 3: Global context (CRITICAL)

**This is where most analysis errors happen.** A target's span drop
means nothing in isolation. You must compare against the system-wide
baseline:

```sql
-- All services span counts
WITH n AS (SELECT service_name, COUNT(*) n FROM normal_traces GROUP BY 1),
     a AS (SELECT service_name, COUNT(*) a FROM abnormal_traces GROUP BY 1)
SELECT *, ROUND((n - COALESCE(a,0))::FLOAT/n*100, 0) drop_pct
FROM n FULL OUTER JOIN a USING (service_name)
ORDER BY drop_pct DESC
```

If all services dropped proportionally, the drop is global — but
**global does not mean irrelevant**. Proceed to L4.

### Level 4: Causal chain — WHY did global traffic drop?

Check the load generator / frontend:

```sql
-- Load generator latency (is it blocked?)
SELECT 'normal' w, count(*) spans,
       p99(duration/1e6) p99_ms, max(duration/1e6) max_ms
FROM normal_traces WHERE service_name LIKE '%load%'
UNION ALL ...
```

**Key signal: loadgen p99 or max jumping to the timeout ceiling
(typically 20 000 ms).** This means:

1. The injection caused some service to block/timeout
2. The load generator is synchronous — it waits for each request
3. While waiting 20 s for a timeout, it sends no new requests
4. Global throughput collapses as a side effect

This is the injection working, not background noise. The target may
look healthy because the requests that would have exposed degradation
never arrived — they timed out upstream.

### Level 5: Caller-side signals

Trace the call chain from frontend/loadgen to the target:

```sql
-- Who calls the target in normal window?
SELECT caller.service_name, callee.span_name, COUNT(*)
FROM normal_traces callee
JOIN normal_traces caller ON callee.parent_span_id = caller.span_id
WHERE callee.service_name = '<target>'
  AND caller.service_name != '<target>'
GROUP BY 1, 2 ORDER BY 3 DESC
```

Then check the same callers in the abnormal window:
- Did calls vanish? (partition / kill severed the link)
- Did caller latency spike to timeout? (caller blocked waiting)
- Did caller return 5xx? (target was down during the call)
- Did caller's Error spans appear? (exception in caller code)

**A caller returning 503 or timing out at 20 s on calls to the target
is direct evidence the injection worked — even if the target itself
shows no degradation.**

### Level 6: Resource metrics

For JVM / container faults, check k8s and JVM metrics:

```sql
SELECT metric, 'normal' w, AVG(value) avg, MAX(value) max
FROM normal_metrics
WHERE service_name = '<target>'
  AND metric IN ('container.memory.rss', 'container.memory.usage',
                  'k8s.pod.memory_limit_utilization',
                  'container.cpu.usage', 'k8s.container.restarts',
                  'k8s.deployment.available')
GROUP BY metric
UNION ALL ...
```

memory_limit_utilization jumping from 23% to 77% is a strong JVM heap
stress signal even when latency looks normal (GC pauses cause slow
requests to timeout — they never complete a span — while fast requests
between pauses look fine).

### Level 7: Per-endpoint and HTTP status breakdown

Aggregate service-level stats hide endpoint-level signals:

```sql
-- Frontend endpoints with errors or timeout
SELECT span_name, COUNT(*) cnt,
       p99(duration/1e6) p99_ms, max(duration/1e6) max_ms,
       SUM(CASE WHEN CAST("attr.http.response.status_code" AS INT) >= 400
           THEN 1 ELSE 0 END) http_err
FROM abnormal_traces WHERE service_name = '<frontend>'
GROUP BY 1
HAVING max(duration/1e6) > 10000
    OR SUM(CASE WHEN CAST(...) >= 400 THEN 1 ELSE 0 END) > 0
```

Then verify that the error endpoints are on the target's call path —
trace the chain through parent_span_id joins.

## Common traps

### "All services dropped proportionally → reject"
Wrong if the loadgen is synchronous. Check loadgen latency first.

### "Target latency is flat or improved → no effect"
Wrong if slow requests timed out and never completed a span. The
surviving requests are the fast ones — survivorship bias.

### "GT says silent → must be no signal"
GT "silent" means the GT labeler didn't see a signal. You might,
especially on the caller side.

### "p99 didn't change → nothing happened"
Break down by span_name. Aggregate p99 dilutes endpoint-specific
signals when healthy endpoints dominate the sample.

### "Only N% span drop → noise"
Check whether those missing spans are concentrated on specific
endpoints (especially ones that call the target). A 9% overall drop
that is 100% on one endpoint is not noise.

## Forming your verdict

After completing L0–L7, classify each case:

| Category | Criteria |
|----------|----------|
| **Injection effective, seed correct** | Seed confirmed, propagation matches data |
| **Injection effective, seed missed** | Data shows clear signal (caller timeout, 503, metrics spike) but seed rejected |
| **Injection ineffective** | No signal anywhere — target, callers, loadgen, metrics all normal |
| **Ambiguous** | Weak/mixed signals that could go either way |

For "seed missed" cases, identify specifically what the seed agent
didn't check (caller-side latency? http status? resource metrics?)
— this feeds back into fault doc improvements.

## Output format

Present results as a table first (case, fault type, category, key
evidence), then expand on interesting cases. Always show the SQL
you ran so the user can verify.
