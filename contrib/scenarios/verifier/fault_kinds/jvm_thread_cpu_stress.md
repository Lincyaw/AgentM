# jvm_thread_cpu_stress

## How the injection works
A JVM agent attached to the target pins JVM threads to busy-loop,
burning CPU from inside the JVM. Externally indistinguishable from
generic CPU stress.

## What the data should show
Same observable shape as `cpu_stress`: target CPU spikes, inbound
span latency rises across endpoints, possible timeouts.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade behaves like
`cpu_stress`: callers slow / fail in proportion to how much they
wait on the target, then their callers in turn.
