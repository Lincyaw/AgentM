# jvm_method_latency

## How the injection works
A JVM agent inserts a sleep into a specific method on the target.
Only that method slows; sibling methods are unchanged.

## What the data should show
Target spans matching the targeted method name show a flat latency
shift equal to the injected delay; spans for other methods on the
same service look normal.

## How to observe on a neighbour
Because only one method is slowed, the signal on a caller is
concentrated on the specific endpoint that exercises that method.
The caller's service-wide aggregate may look flat if unaffected
endpoints outnumber the affected one. Always break down by
`span_name` on the caller to find the affected call path.

## How the failure tends to propagate
In-pod fault scoped to one method — target is the `from`. Cascade
reaches callers that exercise the slowed method, then their callers
that wait on those. Endpoints / callers that avoid the method are
unaffected.
