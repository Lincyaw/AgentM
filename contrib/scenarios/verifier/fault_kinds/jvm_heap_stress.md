# jvm_heap_stress

## How the injection works
A JVM agent inside the target fills the heap, forcing GC pressure or
OutOfMemoryError. Request-handler threads stall during long GC
pauses; severe cases OOM the JVM.

## What the data should show
Target inbound latency shows multi-second spikes correlated with GC
events; heap metrics rise; GC-time fraction climbs. Possibly an
OOM-kill leading to a pod_failure-style restart gap.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade follows GC-stall
windows: callers waiting on the target time out / slow down during
each pause. If OOM-kill occurs the cascade for that subinterval
matches pod_failure.
