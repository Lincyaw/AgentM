# clock_skew

## How the injection works
Chaos-mesh shifts the target pod's system clock by a configured
amount. Consequences depend entirely on which application logic
keys off wall-clock time.

## What the data should show
Often silent in observability data. Possible symptoms: token
expiry / authentication failures, time-window cache misses, span
timestamps inconsistent with peers. Latency and error metrics may
look normal.

## How the failure tends to propagate
In-pod fault — target is the `from`. Cascade only surfaces if a
time-dependent failure produces observable downstream errors;
otherwise mark `injection_effective: ambiguous`.
