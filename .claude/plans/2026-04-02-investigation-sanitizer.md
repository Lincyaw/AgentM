# Plan: Investigation Sanitizer

**Date**: 2026-04-02
**Status**: DRAFT

## Requirements Restatement

Implement an automated investigation quality checking system (linter/sanitizer) for the RCA orchestrator. The system has two layers:

1. **CodeSanitizer** — deterministic checks against HypothesisStore, ServiceProfileStore, and a new InvestigationTracker. Runs every round (lightweight) and at pre-finalize (full suite). Zero LLM cost.
2. **CriticSanitizer** — LLM-based semantic judgment using a cheap model. Runs at pre-finalize (sync) and on dispatch (async). Checks reasoning quality that code alone cannot assess.

Both layers produce `SanitizerFinding` objects with severity levels (BLOCK/WARN/INFO). Findings are injected into the orchestrator's message stream. BLOCK findings at finalize time suppress the `<decision>finalize</decision>` tag, forcing the model to address gaps before concluding.

Key behaviors: BLOCK degrades to WARN after 3 consecutive unresolved attempts; budget-exhaustion auto-degrades coverage BLOCKs but not process BLOCKs; sanitizer events are recorded in TrajectoryCollector.

## Related Designs

- [Investigation Sanitizer](../designs/investigation-sanitizer.md) — Full design document
- [Orchestrator](../designs/orchestrator.md) — Host loop and middleware stack
- [Loop Resilience](../designs/loop-resilience.md) — Existing middleware patterns

## Implementation Phases

### Phase 1: Foundation — Data Model and InvestigationTracker

Build the data structures that all subsequent phases depend on.

**Task**: [investigation-tracker](../tasks/2026-04-02-investigation-tracker.md)

- `SanitizerFinding` frozen dataclass (`code`, `severity`, `message`, `details`)
- `Severity` enum: `BLOCK`, `WARN`, `INFO`
- `InvestigationEvent` frozen dataclass (`round`, `event_type`, `data`)
- `InvestigationTracker` class: append-only event log with query methods (`dispatches()`, `task_completions()`, `hypothesis_changes()`, `tool_calls_for()`)
- `Sanitizer` Protocol: `check(trigger, hypothesis_store, profile_store, tracker, ctx) -> list[SanitizerFinding]`
- Unit tests for InvestigationTracker: record, query by type, temporal ordering, filtering

**Location**: `src/agentm/scenarios/rca/sanitizer/models.py`, `src/agentm/scenarios/rca/sanitizer/tracker.py`

### Phase 2: CodeSanitizer — Deterministic Checks

Implement all code-based checks. Each check is a pure function: stores in, findings out.

**Task**: [code-sanitizer](../tasks/2026-04-02-code-sanitizer.md)

Individual check functions (one per error code):
- E1: `check_anchoring_bias` — anomalous services with unchecked upstream
- E2: `check_dimension_gap` — anomalous services with missing dimension coverage
- E3: `check_coverage_gap` — topology services never profiled
- E4: `check_premature_termination` — finalize with high remaining budget + open gaps
- C1: `check_skipped_verify` — confirmed hypothesis without verify task
- C2: `check_unresolved_contradiction` — CONTRADICTED verdict with no follow-up
- C4: `check_no_alternative` — single hypothesis investigated
- J2: `check_investigation_drift` — N consecutive dispatches to same target/hypothesis
- J3: `check_incomplete_chain` — anomalous services not in confirmed hypothesis
- P1: `check_hypothesis_before_scout` — hypothesis created before scout completes
- P3: `check_profile_write_without_read` — update without prior query

`CodeSanitizer` class implementing the `Sanitizer` protocol: routes `trigger` to appropriate subset of checks.

Unit tests: one test per check with controlled store/tracker state. Both "fires" and "does not fire" cases.

**Location**: `src/agentm/scenarios/rca/sanitizer/code_sanitizer.py`

### Phase 3: CriticSanitizer — LLM-Based Checks

Implement semantic checks via LLM calls.

**Task**: [critic-sanitizer](../tasks/2026-04-02-critic-sanitizer.md)

- Critic prompt templates for each check (C3, J1, J4, P2)
- Pydantic response schemas for each check's structured output
- `CriticSanitizer` class implementing `Sanitizer` protocol
  - Sync execution path (pre-finalize: C3, J1, J4)
  - Async execution path (dispatch: P2) — launches `asyncio.Task`, stores pending futures
  - `collect_async_results()` method for SanitizerMiddleware to drain completed async checks
- Uses `create_chat_model` with `with_structured_output` for each check
- Unit tests with mock LLM: verify prompt construction, response parsing, async collection

**Location**: `src/agentm/scenarios/rca/sanitizer/critic_sanitizer.py`

### Phase 4: SanitizerMiddleware — Integration Layer

Wire sanitizers into the orchestrator loop via middleware.

**Task**: [sanitizer-middleware](../tasks/2026-04-02-sanitizer-middleware.md)

`SanitizerMiddleware(MiddlewareBase)`:

- `on_tool_call`: intercept `dispatch_agent`, `update_hypothesis`, `remove_hypothesis`, `update_service_profile`, `query_service_profile` → record events in InvestigationTracker. Also trigger async P2 critic check on dispatch.
- `on_llm_end`:
  - Detect `<decision>finalize</decision>` in response content
  - If finalize detected: run all code checks + sync critic checks (pre-finalize)
  - If BLOCK findings exist: strip finalize tag from response, queue findings for injection
  - Every round: run J2 drift check
  - Every N rounds: run E1–E3 coverage checks
  - On hypothesis status change (detected via on_tool_call intercepting update_hypothesis): run C1, C2, C4, P1
- `on_llm_start`: inject accumulated findings (from previous on_llm_end + completed async critic results) as XML-formatted message
- BLOCK degradation logic: track consecutive block counts per (code, target) pair; degrade after 3
- Budget-aware severity: check LoopContext for exhaustion, degrade coverage/judgment BLOCKs
- TrajectoryCollector integration: record sanitizer events after each check cycle

Unit tests: middleware lifecycle (on_tool_call records events, on_llm_end triggers checks, on_llm_start injects findings), finalize gate (block and allow), degradation logic, budget-aware severity.

**Location**: `src/agentm/scenarios/rca/sanitizer/middleware.py`

### Phase 5: Configuration and Scenario Wiring

Connect sanitizer to config system and RCA scenario setup.

**Task**: [sanitizer-config-wiring](../tasks/2026-04-02-sanitizer-config-wiring.md)

- `SanitizerConfig` Pydantic model: `enabled`, `critic_model`, `periodic_interval`, `drift_window`, `drift_threshold`, `block_on`, `warn_on`, `disable`
- Add `sanitizer: SanitizerConfig | None` field to `OrchestratorConfig`
- Update `RCAScenario.setup()`: create InvestigationTracker, CodeSanitizer, CriticSanitizer, SanitizerMiddleware; return via `ScenarioWiring.orchestrator_middleware`
- Update `config/scenarios/rca_hypothesis/scenario.yaml` with default sanitizer config
- Config validation: block_on and warn_on codes must be valid error codes; critic_model must be set if any critic checks are enabled
- Integration test: build full scenario, verify middleware is in stack, verify config flows through

**Location**: `src/agentm/config/schema.py` (SanitizerConfig), `src/agentm/scenarios/rca/scenario.py` (wiring)

### Phase 6: End-to-End Validation

Verify the full sanitizer pipeline works correctly in realistic scenarios.

**Task**: [sanitizer-e2e](../tasks/2026-04-02-sanitizer-e2e.md)

- Snapshot test: orchestrator attempts finalize with unverified hypothesis → sanitizer blocks → findings injected → model dispatches verify → finalize allowed
- Snapshot test: drift detection fires after 3 same-target dispatches → WARN injected
- Snapshot test: BLOCK degrades after 3 consecutive blocks on same finding
- Snapshot test: budget-exhausted finalize → coverage BLOCKs degrade, C1 stays BLOCK
- Verify TrajectoryCollector receives sanitizer events with correct structure
- Verify sanitizer findings appear in trajectory output for trajectory judger consumption

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| SanitizerMiddleware ordering conflicts with existing middleware | MEDIUM | Place after DynamicContextMiddleware but before TrajectoryMiddleware; document ordering requirement |
| on_llm_end response modification breaks downstream middleware | MEDIUM | Only modify response content (strip finalize tag), preserve all other attributes; test with full middleware stack |
| CriticSanitizer LLM latency slows pre-finalize | LOW | Critic uses cheap/fast model (same as compression); only runs at finalize (rare); sync is justified by gate semantics |
| Async critic results arrive after context compression deletes relevant messages | LOW | Findings are injected as new messages, not retroactively inserted; compression preserves recent messages |
| InvestigationTracker event extraction from tool_args is fragile | MEDIUM | Extract only stable fields (tool_name, service_name, task_type); use defensive .get() with defaults |
| BLOCK degradation allows wrong conclusions through | LOW | Acceptable by design — stuck model produces no value; degradation is visible in trajectory for post-hoc analysis |

## Dependencies

- **Existing**: MiddlewareBase, HypothesisStore, ServiceProfileStore, LoopContext, TrajectoryCollector, ScenarioWiring, create_chat_model
- **No changes to SimpleAgentLoop** — all integration via middleware hooks
- **No changes to existing middleware** — SanitizerMiddleware is additive
- **Minor addition to OrchestratorConfig** — new optional `sanitizer` field with `extra="allow"` already set on FeatureGatesConfig pattern
