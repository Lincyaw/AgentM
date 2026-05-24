# mem_stress

## How the injection works
Stress-ng allocates memory inside the target pod, pushing it toward
OOM. May trigger swap thrashing, GC thrashing (in JVM apps), or an
OOM-kill that ends the container.

## What the data should show
Target memory utilisation climbs steadily; inbound latency rises
during GC / swap pressure. If OOM-killed, the trace shows a
pod_failure-style gap when the container restarts.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade looks like cpu_stress
during the pressure phase; if OOM-kill fires, it degenerates into a
pod_failure cascade for that interval (callers see connection errors
during the restart gap).
