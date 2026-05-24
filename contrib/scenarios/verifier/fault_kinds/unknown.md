# unknown

## How the injection works
The injection's `fault_type` integer does not map to any known
chaos kind in the rcabench enum. Reason from the raw
`injection_point` fields instead of any pre-baked mechanism.

## What the data should show
Run fault-kind-agnostic checks on the target: abnormal vs normal
span volume, error-rate delta, latency-percentile delta. Whatever
moves is your starting point.

## How the failure tends to propagate
No mechanistic prior — derive propagation from what the data
actually shows. If no signal materialises, mark
`injection_effective: ambiguous` and explain.
