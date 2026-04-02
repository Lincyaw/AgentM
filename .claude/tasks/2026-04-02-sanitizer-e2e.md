# Task: End-to-End Validation

**Plan**: [investigation-sanitizer](../plans/2026-04-02-investigation-sanitizer.md)
**Phase**: 6 — E2E Validation
**Design**: [investigation-sanitizer](../designs/investigation-sanitizer.md)
**Depends on**: [sanitizer-config-wiring](2026-04-02-sanitizer-config-wiring.md)

## Scope

Verify the full sanitizer pipeline works correctly with realistic orchestrator interactions. Uses snapshot-based testing (Layer 2) with mock LLM.

## Deliverables

### Snapshot tests — `tests/snapshot/test_sanitizer_pipeline.py`

**Test 1: Finalize blocked by C1 (skipped verify)**
- Setup: orchestrator has confirmed hypothesis, no verify task dispatched
- Model outputs `<decision>finalize</decision>`
- Assert: finalize tag stripped, `<finalize_blocked>` message injected with C1
- Model then dispatches verify, verify completes SUPPORTED
- Model outputs `<decision>finalize</decision>` again
- Assert: finalize proceeds

**Test 2: Drift detection (J2)**
- Setup: 3 consecutive dispatch_agent calls targeting same service and hypothesis
- Assert: WARN finding for J2 injected into next round's messages

**Test 3: BLOCK degradation after 3 attempts**
- Setup: C1 BLOCK fires, model attempts finalize 3 more times without dispatching verify
- Assert: first 3 attempts blocked; 4th attempt: C1 degraded to WARN with `[DEGRADED]` annotation, finalize proceeds

**Test 4: Budget exhaustion degrades coverage BLOCKs**
- Setup: orchestrator at max tool_call_count, E1 and C1 both fire
- Assert: E1 severity → WARN (budget_exhausted), C1 severity → BLOCK (process, not budget-dependent)

**Test 5: TrajectoryCollector receives sanitizer events**
- Setup: trigger a pre-finalize check that produces findings
- Assert: TrajectoryCollector has event with type="sanitizer", correct findings data

**Test 6: Periodic coverage check**
- Setup: run 5 rounds with anomalous services that have unchecked upstream
- Assert: E1 WARN finding appears at round 5

### Integration verification

- Build full RCA scenario with sanitizer enabled via config
- Verify middleware stack order: SanitizerMiddleware after DynamicContextMiddleware, before TrajectoryMiddleware
- Verify sanitizer components have correct references (same hypothesis_store/profile_store instances as the scenario tools)
