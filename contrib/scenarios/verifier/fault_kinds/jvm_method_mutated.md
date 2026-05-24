# jvm_method_mutated

## How the injection works
A JVM agent replaces a method's body with mutated logic on the
target — semantic-behaviour change rather than outright failure
or latency.

## What the data should show
Often subtle. Status codes and latency may look normal while the
returned values are wrong; downstream may surface validation errors
or behavioural anomalies. Pure error-rate / latency views can miss
this.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade depends on how strictly
callers validate semantic correctness. If no caller flags the wrong
output, the observable cascade may be empty — mark
`injection_effective: ambiguous` in that case.
