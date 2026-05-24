# jvm_gc_pressure

## How the injection works
A JVM agent on the target triggers frequent full GCs (e.g.,
`System.gc()` loops or filling the old generation), inducing
repeated stop-the-world pauses.

## What the data should show
Target inbound latency shows recurring spikes aligned with GC
events. GC-time fraction in JVM metrics is markedly elevated; heap
churn is high.

## How the failure tends to propagate
In-pod fault — target is the `from`. Callers see periodic latency
bumps timed to GC; sustained pressure looks like cpu_stress. Cascade
follows the same shape as cpu_stress, with intermittent rather than
continuous symptoms.
