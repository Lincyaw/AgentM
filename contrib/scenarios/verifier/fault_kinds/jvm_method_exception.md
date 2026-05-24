# jvm_method_exception

## How the injection works
A JVM agent on the target rewrites a specific method's bytecode to
throw an exception. Only call sites entering that exact method
fail; sibling methods on the same service stay healthy.

## What the data should show
Target spans whose operation name matches the targeted method show
errors; spans for other methods on the same service look normal.
Error logs cite the injected exception.

## How the failure tends to propagate
In-pod fault scoped to one method — target is the `from`. Cascade
only reaches callers that exercise the affected method; their
callers see RPC errors and may surface anomalies in turn. Callers
that hit only sibling methods are unaffected.
