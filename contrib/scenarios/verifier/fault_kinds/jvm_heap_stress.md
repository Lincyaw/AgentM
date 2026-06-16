# jvm_heap_stress

## How the injection works
A JVM agent inside the target fills the heap, forcing GC pressure or
OutOfMemoryError. Request-handler threads stall during long GC
pauses; severe cases OOM the JVM.

## What the data should show
Target inbound latency shows multi-second spikes correlated with GC
events; heap metrics rise; GC-time fraction climbs. Possibly an
OOM-kill leading to a pod_failure-style restart gap.

### Check resource metrics first

Trace latency can be misleading under heap stress. GC pauses cause
slow requests to time out before completing a span — those requests
vanish from the trace data. The surviving requests completed between
GC pauses and look **faster than normal** (survivorship bias). When
trace latency looks flat or improved, check resource metrics:

Look for memory and CPU metrics on the target (the exact metric
names vary by dataset — discover them from the schema). Key
signals: memory usage spiking well above the normal-window
baseline, memory limit utilization climbing significantly, page
fault counts jumping, CPU usage rising (GC burns CPU). A clear
jump in memory pressure with elevated CPU is strong heap stress
evidence, even when trace latency is flat or lower.

### When the target's trace latency looks normal or improved

GC pauses block request threads. A caller waiting on the target
blocks for the entire pause duration — which often exceeds the
caller's timeout. The caller times out (20 s typical), and the
target never records a span for that request.

Result: the target's completed spans show **lower** latency (only
fast requests survived), while the caller shows **exploding**
latency. Check the caller side:

1. Find callers via `parent_span_id` join in the normal window.
2. In the abnormal window, check if those callers' latency on
   calls to the target spiked significantly beyond the normal
   range — especially toward the system's timeout ceiling.
3. Also check the frontend / load generator for similar signals.

If caller-side latency jumped and resource metrics confirm heap
pressure, the injection is effective. Mark **confirmed** with
`gc_pressure` or `resource_pool_exhausted`.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade follows GC-stall
windows: callers waiting on the target time out / slow down during
each pause. If OOM-kill occurs the cascade for that subinterval
matches pod_failure.
