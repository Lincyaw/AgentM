---
name: verdict-audit
description: >
  Audit whether a verifier verdict (confirmed/rejected) is sound for a
  specific service in a fault-injection case. Trigger when the user asks
  to "check a verdict", "audit a rejection", "look at why X was
  rejected/confirmed", or wants to understand whether the verifier got
  it right on a particular edge. Also trigger proactively when
  investigating discrepancies between verifier output and GT labels.
---

# verdict-audit

You audit a single verifier verdict by reasoning forward from the fault
injection — not backward from the data. The question is always: "given
what was injected, should we expect to see degradation on this service,
and does the data match that expectation?"

## Why this order matters

Looking at data first and then rationalizing invites confirmation bias.
A flat error rate looks like "no problem" until you realize the fault
mechanism predicts latency spikes, not errors. Starting from the
mechanism tells you WHERE to look and WHAT to look for before you open
any parquet file.

## Method

Work through these phases in order. Each phase produces a concrete
written conclusion before moving to the next. Do not skip ahead.

### Phase 1: Understand the injection

Read `injection.json` from the case directory. For each injected fault,
identify:

- **Target service** and **target method/endpoint** (if applicable)
- **Fault type** (e.g. NetworkLoss, JVMRuntimeMutator, PodFailure)
- **Parameters** (loss percentage, mutation config, target_service for
  directed faults, etc.)
- **Direction** (for network faults: `to` means egress from the target
  to a specific peer)

Then read the corresponding fault kind doc at
`contrib/scenarios/verifier/fault_kinds/<fault_kind>.md` to understand
the mechanism. State in one paragraph: what does this fault do at the
system level, and what observable effects does it predict?

### Phase 2: Trace the expected propagation path

Starting from the injection target(s), reason through the call graph:

1. Which services directly call the injection target, or are called by it?
2. For each, does the fault mechanism predict observable degradation on
   that service? Be specific: what KIND of degradation (latency? errors?
   functional breakage?) and through what call path?
3. For the service under audit: is it on a plausible propagation path
   from the injection? Through how many hops? What is the expected
   signal?

If the service under audit has no call-graph relationship to the
injection targets, say so and note that GT may have over-labeled.

Write this as a concrete prediction BEFORE looking at any trace/log
data: "If the fault propagated to service X, I expect to see [specific
signal] because [mechanism]."

### Phase 3: Verify with data

Now query the parquet files using duckdb. Load all parquet files from
the case data directory (each `*.parquet` becomes a view named after
its stem, e.g. `normal_traces`, `abnormal_logs`).

A conclusion drawn from a single data source is not a conclusion.
The dataset has three independent modalities — traces (spans), logs,
and resource metrics — and a verdict must account for all of them
before it is trustworthy. A service that looks clean in spans may
have error logs; a service with flat latency may have spiking CPU;
a throughput number from spans alone may contradict the log trace_id
count. Check every modality for the signal your Phase 2 prediction
says should exist, and report the exact numbers from each.

### Phase 4: Render judgment

Compare your Phase 2 prediction against Phase 3 data:

- **Prediction matched**: the data shows what the mechanism predicts.
  The verdict should be confirmed. If the verifier rejected, it missed
  something — identify what.
- **Prediction not matched**: the mechanism predicts degradation but the
  data doesn't show it. Possible reasons: the fault didn't fire (method
  never called, network rule didn't trigger), the service masked it
  (retry, fallback, circuit breaker), or data collection missed it
  (span gaps). The verifier's rejection may be correct.
- **No plausible path**: the service is not on the propagation path from
  the injection. The verifier's rejection is correct regardless of what
  the data shows. GT likely over-labeled.

State clearly: "The verifier's [verdict] is [sound / unsound] because
[reason]."

## Common traps

These are patterns that have led to wrong conclusions in past audits:

- **Throughput drop alone is not degradation.** Fewer requests arriving
  at a service means callers stopped calling — that's the caller's
  problem. This includes the case where traffic drops to zero: if the
  business flow's prerequisite steps were blocked (e.g. you can't
  preserve a ticket if route queries failed), downstream services
  naturally receive no requests. That is business-flow interruption,
  not cascade failure of the downstream service.

- **"Thread pool exhaustion" is speculation without evidence.** Don't
  claim a caller's threads are exhausted unless you can show it in the
  data (e.g. the caller's OWN latency spiked for ALL endpoints, not
  just the affected one, and its request queue backed up).

- **Load generator behavior is irrelevant.** The load generator is
  external test tooling, not part of the microservice system. Its
  throughput drop doesn't prove cascade failure within the system.

- **GT is not ground truth.** GT labels are generated by a separate
  process that may over-label (marking services as "propagated" when
  they merely received fewer requests). A verifier rejection that
  contradicts GT is not automatically wrong.

- **JVMRuntimeMutator is silent.** URL mutations cause no errors on the
  target itself (404 returns fast, target looks healthy). The signal is
  on the call path: the downstream that should have received the
  request didn't, or received a malformed one.

## Input

The user provides:
- A case directory (with parquet files, injection.json, causal_graph.json)
- A verifier run directory (with all_verdicts.json, report.json)
- Optionally, a specific service to audit

If no specific service is given, pick the most interesting discrepancy
(GT says propagated, verifier says rejected) and audit that.
