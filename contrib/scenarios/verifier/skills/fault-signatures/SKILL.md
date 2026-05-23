---
confidence: fact
description: 'Per-fault-kind diagnostic signatures: what changes on the injection target, what propagates downstream, and the disqualifying signals that rule the kind out.'
name: fault-signatures
tags:
- verifier
- fault-kind
- diagnostics
- propagation
trigger_patterns:
- fault_kind
- pod_failure
- network_delay
- cpu_stress
- mem_stress
- http_aborted
- jvm_method_exception
type: skill
---

# Fault Signatures

Per-fault-kind reference. Pick the section matching the `fault_kind`
returned by `get_injection_spec`. Each section gives:
- **Target signature**: what should change on the injected component
- **Propagation pattern**: what typically lights up downstream
- **Disqualifying signals**: what would prove this isn't the actual fault
- **SQL stubs**: starting queries (rename columns / windows as needed)

Common-to-all conventions in the rcabench parquets:
- Tables come paired: `abnormal_*` and `normal_*` (same schema).
- Span duration is `duration` in nanoseconds — divide by `1e9` for seconds.
- Timestamps in `abnormal_traces` / `abnormal_logs` are unix-nanosecond
  `start_time` / `time_unix_nano`; metric tables use unix seconds in
  `time_unix`.
- Service identifier is `service_name`. Operation/span name is
  `span_name`. Status code lives in `attr.http.response.status_code`
  or `status_code` depending on the source.

If column names diverge, run `list_tables` once to confirm — do NOT
proceed on the wrong column names.

---

## `pod_failure` / `container_kill` (kill the pod / container)

**Target signature** — the injected `container|X` / `pod|X`:
- Span volume from that service drops to ~0 then recovers (or stays
  zero for the abnormal window).
- Container restart count gauge increments (look at
  `abnormal_metrics_sum` for restart counters; the exact metric name
  varies — search `metric_name LIKE '%restart%'`).
- If the target is a DB pod (mysql, redis, mongo): direct callers'
  spans accumulate connection errors / refused-connection logs in the
  same minute as the kill.

**Propagation pattern**:
- Immediate downstream (within seconds): every service whose spans
  named into the killed service's operations show JDBC / connection
  errors. Look in `abnormal_logs` for exception class names
  (`CommunicationsException`, `ConnectException`, `JDBCConnectionException`).
- Second-hop: services calling the first-hop services see latency
  spikes or 500s as the first-hop callers blow up.
- Recovery may happen within the window; record `affected_window` to
  the recovery point if so.

**Disqualifying signals**:
- No span-volume drop at all on the named target → injection didn't
  fire OR target is wrong (set `injection_effective="false"`).
- Span volume drops but timestamp is before `abnormal_start` → pre-
  existing failure, not this injection.

**SQL stub** (span volume cliff on target):
```sql
SELECT date_trunc('second', to_timestamp(start_time / 1e9)) AS sec,
       count(*) AS spans
FROM read_parquet('abnormal_traces.parquet')
WHERE service_name = 'mysql'
GROUP BY sec ORDER BY sec;
```

---

## `network_delay` / `network_loss` / `network_partition` / `network_corrupt`

**Target signature** — the affected `service|X` (chaos-mesh network
faults target a pod's network, so the signature shows on spans
involving that pod):
- Latency p95 on inbound spans rises sharply (`network_delay`) OR
  error rate on outbound spans rises (`network_loss` / `_partition`).
- For `network_partition`: span count for outbound calls drops to
  near-zero — the caller can't reach the partitioned target.
- For `network_corrupt`: spans complete but with truncated /
  malformed responses; error spans rise with deserialisation
  exceptions in logs.

**Propagation pattern**:
- Callers of the affected service see (a) latency increases (delay/loss
  retransmits) or (b) timeout / 500 spans (partition/corrupt).
- Loss vs delay distinguishable by error rate: `network_delay` raises
  latency without proportional errors; `network_loss` raises both.

**Disqualifying signals**:
- p95 latency on the target's spans is flat → injection didn't fire.
- Errors are evenly spread across services not connected to the
  target → coincidence / unrelated cause.

**SQL stub** (latency delta on target service inbound spans):
```sql
WITH a AS (
  SELECT service_name,
         approx_quantile(duration/1e9, 0.95) AS p95_s
  FROM read_parquet('abnormal_traces.parquet')
  WHERE service_name = 'TARGET' GROUP BY service_name
),
n AS (
  SELECT service_name,
         approx_quantile(duration/1e9, 0.95) AS p95_s
  FROM read_parquet('normal_traces.parquet')
  WHERE service_name = 'TARGET' GROUP BY service_name
)
SELECT a.service_name, n.p95_s AS normal_p95, a.p95_s AS abnormal_p95,
       a.p95_s / n.p95_s AS ratio
FROM a JOIN n USING (service_name);
```

---

## `cpu_stress` / `mem_stress` / `jvm_heap_stress` / `jvm_thread_cpu_stress`

**Target signature** — the stressed `service|X` / `container|X`:
- CPU / memory utilisation gauges spike to near 100% on the target.
  Look in `abnormal_metrics` for metrics with names containing
  `cpu`, `memory`, `heap`, `gc`. `jvm.cpu.recent_utilization`,
  `jvm.memory.used`, `process.cpu.utilization` are common.
- Latency on the target's inbound spans rises (saturated process can't
  serve fast).
- For `jvm_heap_stress` / `jvm_gc_pressure`: GC time spikes; look at
  `jvm.gc.duration` or similar.

**Propagation pattern**:
- Callers of the stressed service see latency increases proportional
  to the target's degradation.
- If the stress is severe enough to cause OOM: behaviour collapses to
  `pod_failure` pattern.

**Disqualifying signals**:
- Target CPU is flat or within normal variance → fault didn't materialise.
- Latency rises but CPU is normal → some other bottleneck (DB, lock).

**SQL stub** (CPU gauge on target):
```sql
SELECT service_name,
       avg(CASE WHEN window='normal' THEN value END) AS cpu_normal,
       avg(CASE WHEN window='abnormal' THEN value END) AS cpu_abnormal
FROM (
  SELECT 'normal' AS window, service_name, value
  FROM read_parquet('normal_metrics.parquet')
  WHERE metric_name LIKE '%cpu%utilization%' AND service_name='TARGET'
  UNION ALL
  SELECT 'abnormal', service_name, value
  FROM read_parquet('abnormal_metrics.parquet')
  WHERE metric_name LIKE '%cpu%utilization%' AND service_name='TARGET'
) GROUP BY service_name;
```

---

## `http_aborted` / `http_slow` / `http_payload_modified` / `http_response_status_modified`

**Target signature** — the targeted `service|X` (chaos-mesh modifies
HTTP at the proxy / app level on the target pod):
- `http_aborted`: inbound spans on the target show error status codes
  (often 500-class) without any backend exception in the target's logs
  (the abort happens before app code runs).
- `http_slow`: inbound span latency rises sharply with no CPU/memory
  signal on the target.
- `http_payload_modified` / `http_response_status_modified`: callers
  see unexpected response shapes — JSON parse errors in caller logs,
  or status codes the spec says shouldn't appear.

**Propagation pattern**:
- Callers calling the modified endpoint surface the modification:
  deserialisation errors, retries, fallback paths exercised.
- The modification is endpoint-scoped — only spans matching the
  modified URL pattern are affected; sibling endpoints on the same
  service may be untouched.

**Disqualifying signals**:
- Target spans look identical to normal-window — the modification
  selector didn't match anything in this window.

**SQL stub** (error rate on target's spans, abnormal vs normal):
```sql
SELECT 'abnormal' AS win,
       count(*) FILTER (WHERE "attr.http.response.status_code" >= 400)::float
       / nullif(count(*), 0) AS error_rate
FROM read_parquet('abnormal_traces.parquet')
WHERE service_name = 'TARGET'
UNION ALL
SELECT 'normal',
       count(*) FILTER (WHERE "attr.http.response.status_code" >= 400)::float
       / nullif(count(*), 0)
FROM read_parquet('normal_traces.parquet')
WHERE service_name = 'TARGET';
```

---

## `jvm_method_exception` / `jvm_jdbc_exception` / `jvm_method_latency` / `jvm_jdbc_latency`

**Target signature** — the targeted `service|X`, scoped to a specific
method (chaos-mesh uses bytecode instrumentation):
- `*_exception`: spans for the targeted method-class show error tags;
  caller logs accumulate exception traces naming the injected method
  class.
- `*_latency`: only the targeted method's spans show latency spikes;
  sibling methods on the same service do NOT.

**Propagation pattern**:
- Same as `http_*` but at the method level: only callers exercising
  the targeted method propagate. Methods on the same service that
  don't call the injected method are clean.
- JDBC exceptions surface as failed DB queries → caller falls back or
  errors out → upstream caller sees the error.

**Disqualifying signals**:
- All methods on the service show similar degradation → too broad to
  be a method-level injection; consider whether the real fault is
  service-wide (cpu_stress / pod_failure mislabeled).

**SQL stub** (find the affected method via error span_name in target):
```sql
SELECT span_name, count(*) AS errors
FROM read_parquet('abnormal_traces.parquet')
WHERE service_name = 'TARGET' AND status_code != 'OK'
GROUP BY span_name
ORDER BY errors DESC LIMIT 20;
```

---

## `dns_resolution_failed` / `dns_resolution_wrong`

**Target signature**: outbound spans from the target service fail with
DNS-related exceptions in logs. Latency may rise (timeouts) before
errors materialise.

**Propagation**: only outbound calls from the target are affected;
inbound traffic to the target proceeds normally unless the target
needs DNS to serve its own response.

---

## `clock_skew`

**Target signature**: rarely visible in spans/metrics directly. Look
at timestamps: are the target's spans' `start_time` values inconsistent
with caller's recorded "end of call" timestamps?

**Propagation**: usually subtle. May show as token expiry errors (JWT
`exp` mismatch), cache misses on time-windowed keys, or trace
reconstruction issues. **Often the injection is silent in this dataset
shape** — be ready to mark `injection_effective="ambiguous"`.

---

## Unknown / unmapped `fault_kind`

If `get_injection_spec` returns `fault_kind="unknown"` (the chaos-type
mapping didn't recognise the index), fall back to the
fault-kind-agnostic checks:
1. Span volume on the named target — drop = something happened.
2. Latency p95 on the target — rise = something happened.
3. Error-rate on the target — rise = something happened.
4. If none of the above moved, set `injection_effective="false"`.

Document the agnostic approach in `effectiveness_rationale` so the
downstream consumer knows you couldn't apply a specific signature.
