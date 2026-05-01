**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](../pluggable-architecture.md) and
[extension-as-scenario.md](../extension-as-scenario.md).

---

# Design: Investigation Sanitizer

**Status**: DRAFT
**Created**: 2026-04-02
**Last Updated**: 2026-04-02

---

## 1. Overview

The Investigation Sanitizer is a system of automated checks that detect reasoning errors and process violations in the orchestrator's investigation flow. It operates as a feedback loop: checks run at defined trigger points, produce findings, and inject them back into the orchestrator's message stream for self-correction.

The sanitizer does **not** replace the model's reasoning or enforce a rigid pipeline. It detects when the model has deviated from sound investigative practice and provides specific, actionable feedback — like a linter for investigation logic.

---

## 2. Motivation

The current RCA scenario relies on detailed prompt engineering to guide the orchestrator's behavior. This works well with highly capable models but degrades with weaker ones. Two recurring failure modes:

1. **Premature convergence** — The model anchors on the first anomalous service and declares root cause without tracing upstream dependencies or checking all dimensions.
2. **Noise entrapment** — The model follows a misleading signal deep into an irrelevant direction, consuming budget without backtracking.

Prompt-based guidance is "advisory" — the model can ignore it. The sanitizer turns critical rules into enforceable checks without removing the model's autonomy over investigation strategy.

### Design Principle

**Detect and feedback, don't control.** The sanitizer produces findings; the orchestrator decides how to respond. The only hard gate is the finalize check — the model cannot declare completion when mandatory conditions are unmet.

---

## 3. Error Taxonomy

All checks target errors classified into four categories. Each error has a code, a detection method, and a trigger point.

### 3.1 Exploration Failures (E) — Insufficient Data Collection

| Code | Name | Description | Detection |
|------|------|-------------|-----------|
| E1 | Anchoring bias | Anomalous service's upstream never queried | Code: profile store topology vs queried services |
| E2 | Dimension gap | Service checked on some dimensions but not all (latency/error/volume/resources) | Code: profile.data_sources_queried vs required dimensions |
| E3 | Coverage gap | Services on the call chain never appear in any query | Code: topology services vs profile store entries |
| E4 | Premature termination | Finalize with significant budget remaining and open gaps | Code: remaining steps/tools ratio + open gap count |

### 3.2 Confirmation Failures (C) — Insufficient Verification

| Code | Name | Description | Detection |
|------|------|-------------|-----------|
| C1 | Skipped verify | Hypothesis confirmed without any verify task | Code: hypothesis store status history vs task dispatch log |
| C2 | Unresolved contradiction | CONTRADICTED verdict exists with no follow-up investigation | Code: verify results vs subsequent dispatches |
| C3 | Causal direction unproven | Root cause declared without internal-time or independent-anomaly evidence | Critic: check deep_analyze findings for attribution evidence |
| C4 | No alternative elimination | Only one hypothesis investigated, no alternatives tested | Code: count of rejected/investigated hypotheses |

### 3.3 Judgment Failures (J) — Flawed Reasoning

| Code | Name | Description | Detection |
|------|------|-------------|-----------|
| J1 | Signal misread | Strong hypothesis rejected over minor metric discrepancy | Critic: compare rejection reason against cumulative evidence strength |
| J2 | Investigation drift | N+ consecutive rounds focused on same hypothesis/service without progress | Code: dispatch history target diversity over sliding window |
| J3 | Incomplete causal chain | Root cause doesn't explain all observed anomalous services | Code: CausalGraph nodes vs profile store anomalous services |
| J4 | Symptom-as-cause | Downstream victim identified as root cause | Critic: check if candidate's upstream services are also anomalous |

### 3.4 Process Failures (P) — Operational Violations

| Code | Name | Description | Detection |
|------|------|-------------|-----------|
| P1 | Hypothesis before scout | Hypothesis formed before any scout task completes | Code: first hypothesis timestamp vs first scout completion |
| P2 | Low-quality dispatch | dispatch_agent task parameter missing target/metric/hypothesis | Critic: check dispatch instruction completeness |
| P3 | Profile write without read | update_service_profile called without prior query_service_profile for that service | Code: tool call sequence analysis |

---

## 4. Architecture

### 4.1 Two-Layer Design

```
Orchestrator Loop (SimpleAgentLoop)
    │
    ├── on_llm_end ──→ SanitizerMiddleware
    │                      │
    │                      ├── CodeSanitizer (deterministic, every round)
    │                      │     reads: HypothesisStore, ServiceProfileStore,
    │                      │            dispatch history, tool call log
    │                      │
    │                      ├── CriticSanitizer (LLM-based, conditional trigger)
    │                      │     reads: dispatch instructions, findings text,
    │                      │            hypothesis evidence vs rejection reasons
    │                      │
    │                      └── Aggregated findings
    │                              │
    └── on_llm_start ◀────────────┘  (injected into next round's messages)
```

**CodeSanitizer** — Deterministic checks against store state and tool call history. Zero cost, runs every round. Covers: E1–E4, C1–C2, C4, J2–J3, P1, P3.

**CriticSanitizer** — LLM call (cheap/fast model) for checks requiring semantic judgment. Runs conditionally at trigger points. Covers: C3, J1, J4, P2.

### 4.2 Trigger Points

| Trigger | Which sanitizers run | Rationale |
|---------|---------------------|-----------|
| **Every round** (on_llm_end) | CodeSanitizer: drift detection (J2) only | Lightweight, catches drift early |
| **Hypothesis status change** | CodeSanitizer: C1, C2, C4, P1 | Status transitions are decision points |
| **Pre-finalize** (decision=finalize detected) | CodeSanitizer: ALL code checks | Hard gate — block finalize if critical checks fail |
| **Pre-finalize** | CriticSanitizer: C3, J1, J4 | Semantic validation before conclusion |
| **Every N rounds** (configurable, default 5) | CodeSanitizer: E1–E3 coverage checks | Periodic nudge for exploration gaps |
| **dispatch_agent called** | CriticSanitizer: P2 | Check instruction quality at dispatch time |

### 4.3 Finding Severity Levels

| Severity | Behavior | Example |
|----------|----------|---------|
| **BLOCK** | Reject finalize, force continuation | C1 (no verify before confirm), J3 (unexplained anomalies) |
| **WARN** | Inject finding as advisory message | E2 (dimension gap), J2 (drift detected) |
| **INFO** | Include in context but no urgency | P3 (profile write order) |

Only BLOCK-level findings can prevent finalize. WARN and INFO are advisory — the model may choose to address them or explain why they are not relevant.

### 4.4 Finalize Gate Logic

When `<decision>finalize</decision>` is detected in `on_llm_end`:

1. Run ALL CodeSanitizer checks
2. Run CriticSanitizer pre-finalize checks
3. Collect BLOCK-level findings
4. If any BLOCK findings exist:
   - Suppress the finalize decision (return modified response without the tag)
   - Inject findings as a structured message listing each blocker
   - The orchestrator continues its loop and must address the blockers
5. If no BLOCK findings: allow finalize to proceed

This integrates with the existing `should_terminate` mechanism. The sanitizer middleware sits before termination evaluation in the `on_llm_end` chain, so it can modify the response before `should_terminate` sees it.

---

## 5. Data Dependencies

The sanitizer reads from existing stores and tracking state. Some tracking state is new.

### 5.1 Existing (no changes needed)

- **HypothesisStore** — hypothesis status, evidence, parent chains
- **ServiceProfileStore** — anomalous services, topology, data_sources_queried, observations
- **LoopContext** — step count, tool_call_count, max_steps

### 5.2 New: InvestigationTracker

A lightweight event log that records investigation-level events. The sanitizer reads from it; tool wrappers and middleware write to it.

```python
@dataclass(frozen=True)
class InvestigationEvent:
    round: int
    event_type: str          # "dispatch", "task_complete", "hypothesis_change", "tool_call"
    data: dict[str, Any]     # event-specific payload

class InvestigationTracker:
    """Append-only event log for investigation-level events."""

    def record(self, round: int, event_type: str, data: dict) -> None: ...
    def dispatches(self) -> list[InvestigationEvent]: ...
    def task_completions(self) -> list[InvestigationEvent]: ...
    def hypothesis_changes(self) -> list[InvestigationEvent]: ...
    def tool_calls_for(self, tool_name: str) -> list[InvestigationEvent]: ...
```

Events captured:
- `dispatch`: agent_id, task_type, target_services, hypothesis_id, task_instruction
- `task_complete`: agent_id, task_type, verdict (for verify), findings_summary
- `hypothesis_change`: hypothesis_id, old_status, new_status
- `tool_call`: tool_name, key args (for P3 sequence checking)

The tracker is wired into the scenario via `ScenarioWiring` — the RCA scenario creates it and passes it to both the sanitizer and the tool wrappers that record events.

---

## 6. Interface Definition

### 6.1 SanitizerFinding

```python
@dataclass(frozen=True)
class SanitizerFinding:
    code: str               # "E1", "C2", "J3", etc.
    severity: str           # "BLOCK", "WARN", "INFO"
    message: str            # Human-readable description
    details: dict[str, Any] # Structured data (e.g., which services are uncovered)
```

### 6.2 Sanitizer Protocol

```python
class Sanitizer(Protocol):
    def check(
        self,
        trigger: str,               # "every_round", "pre_finalize", "hypothesis_change", "dispatch"
        hypothesis_store: HypothesisStore,
        profile_store: ServiceProfileStore,
        tracker: InvestigationTracker,
        ctx: LoopContext,
    ) -> list[SanitizerFinding]: ...
```

Both CodeSanitizer and CriticSanitizer implement this protocol. The CriticSanitizer additionally takes a model reference for LLM calls.

### 6.3 SanitizerMiddleware

A `MiddlewareBase` subclass that:

- `on_llm_end`: runs trigger-appropriate checks, modifies response if BLOCK findings exist at finalize
- `on_llm_start`: injects accumulated findings from previous round into messages
- `on_tool_call`: records events to InvestigationTracker (dispatch_agent, update_hypothesis, update/query_service_profile)

### 6.4 Integration with ScenarioWiring

The RCA scenario creates the sanitizer components and returns them via `ScenarioWiring.orchestrator_middleware`:

```python
# In RCAScenario.setup():
tracker = InvestigationTracker()
code_sanitizer = CodeSanitizer(tracker=tracker)
critic_sanitizer = CriticSanitizer(tracker=tracker, model=critic_model)
sanitizer_mw = SanitizerMiddleware(
    sanitizers=[code_sanitizer, critic_sanitizer],
    tracker=tracker,
    hypothesis_store=hypothesis_store,
    profile_store=profile_store,
)
# Returned in ScenarioWiring.orchestrator_middleware
```

---

## 7. CodeSanitizer Check Specifications

### E1: Anchoring Bias

```
FOR each service S where profile.is_anomalous == True:
    upstream = profile.upstream_services
    FOR each U in upstream:
        IF U not in profile_store OR profile(U).observations is empty:
            WARN "Anomalous service {S} has unchecked upstream {U}"
```

### E2: Dimension Gap

```
REQUIRED_DIMENSIONS = {"latency", "error_rate", "call_volume", "resources"}

FOR each service S where profile.is_anomalous == True:
    queried = set(profile.data_sources_queried)
    missing = REQUIRED_DIMENSIONS - queried
    IF missing:
        WARN "Anomalous service {S} missing dimensions: {missing}"
```

Note: dimension names are approximate — the check maps data_sources_queried entries to dimension categories via a configurable mapping.

### E3: Coverage Gap

```
all_topology_services = union of all profile upstream + downstream + service_name
profiled_services = set(profile_store.get_all().keys())
uncovered = all_topology_services - profiled_services
IF uncovered:
    WARN "Services on topology but never profiled: {uncovered}"
```

### C1: Skipped Verify

```
FOR each hypothesis H where status == "confirmed":
    verify_tasks = tracker.task_completions()
        .filter(task_type="verify", hypothesis_id=H.id)
    IF verify_tasks is empty:
        BLOCK "Hypothesis {H.id} confirmed without verify task"
```

### C2: Unresolved Contradiction

```
FOR each verify completion V where verdict == "CONTRADICTED":
    subsequent = tracker.dispatches().after(V.round)
    IF no subsequent dispatch references V.hypothesis_id or V.contradicting_services:
        BLOCK "CONTRADICTED verdict for {V.hypothesis_id} has no follow-up"
```

### C4: No Alternative Elimination

```
total_investigated = count hypotheses with status in {investigating, confirmed, rejected, refined, inconclusive}
IF total_investigated <= 1 AND any hypothesis is confirmed:
    WARN "Only one hypothesis investigated — no alternatives eliminated"
```

### J2: Investigation Drift

```
WINDOW = 3 rounds
recent_dispatches = tracker.dispatches().last(WINDOW)
target_set = unique(d.target_services for d in recent_dispatches)
hypothesis_set = unique(d.hypothesis_id for d in recent_dispatches)
IF len(target_set) == 1 AND len(hypothesis_set) == 1 AND WINDOW dispatches exist:
    WARN "Investigation drift: {WINDOW} consecutive rounds targeting {target_set} on {hypothesis_set}"
```

### J3: Incomplete Causal Chain (pre-finalize only)

```
anomalous_services = {s.service_name for s in profile_store.query(anomalous_only=True)}
# The causal chain is not yet available as a data structure at this point,
# so we check hypothesis descriptions and evidence for service mentions.
explained_services = services mentioned in confirmed hypothesis evidence + description
unexplained = anomalous_services - explained_services
IF unexplained:
    BLOCK "Anomalous services not explained by confirmed hypothesis: {unexplained}"
```

### P1: Hypothesis Before Scout

```
first_hypothesis = min(tracker.hypothesis_changes(), key=round)
first_scout_complete = min(tracker.task_completions().filter(task_type="scout"), key=round)
IF first_hypothesis.round <= first_scout_complete.round:
    WARN "Hypothesis formed at round {first_hypothesis.round} before scout completed at round {first_scout_complete.round}"
```

### P3: Profile Write Without Read

```
FOR each update_service_profile call U:
    prior_queries = tracker.tool_calls_for("query_service_profile")
        .before(U.round)
        .filter(service_name overlaps U.service_name)
    IF prior_queries is empty:
        INFO "update_service_profile({U.service_name}) without prior query"
```

---

## 8. CriticSanitizer Check Specifications

The critic sanitizer uses a small/fast LLM (e.g., the compression model already configured) with focused prompts. Each check is a single LLM call with structured output.

### C3: Causal Direction Unproven

**Trigger**: pre-finalize

**Input to critic**: confirmed hypothesis description + all deep_analyze findings for that hypothesis

**Prompt**: "Does the evidence include internal-time attribution or independent-anomaly analysis for the root cause service? A root cause must have anomalies NOT explainable by its own dependencies."

**Output**: { proven: bool, missing_evidence: str }

### J1: Signal Misread

**Trigger**: pre-finalize, when any hypothesis was rejected after being confirmed or having strong evidence

**Input to critic**: the hypothesis's full evidence list + the rejection reason

**Prompt**: "Was this hypothesis rejected based on a minor discrepancy in a single metric while multiple independent signals supported it? Compare the strength of supporting evidence against the contradicting evidence."

**Output**: { misread: bool, reasoning: str }

### J4: Symptom-as-Cause

**Trigger**: pre-finalize

**Input to critic**: confirmed root cause service name + profile store data for that service and its upstream services

**Prompt**: "Is the confirmed root cause's upstream also anomalous? If so, the confirmed service may be a downstream victim, not the true origin."

**Output**: { likely_symptom: bool, suspect_upstream: list[str] }

### P2: Low-Quality Dispatch

**Trigger**: on dispatch_agent tool call

**Input to critic**: the task instruction text

**Prompt**: "Does this dispatch instruction include: (1) target service names, (2) specific metric values from prior findings, (3) the hypothesis being tested, (4) forward predictions? List what is missing."

**Output**: { quality: "good" | "incomplete" | "poor", missing: list[str] }

---

## 9. Finding Injection Format

Findings are injected as a structured message before the next LLM call:

```xml
<sanitizer_report round="N">
<finding code="E1" severity="WARN">
Anomalous service `ts-travel2-service` has unchecked upstream: `ts-basic-service`, `ts-price-service`.
These services appear in the topology but have no observations in the profile store.
</finding>
<finding code="C1" severity="BLOCK">
Hypothesis H3 is confirmed but has no verify task. A verify worker must test this hypothesis
before finalization is allowed.
</finding>
</sanitizer_report>
```

For BLOCK findings during finalize, the message explicitly states what must be resolved:

```xml
<finalize_blocked reason="2 BLOCK findings">
You attempted to finalize but the following conditions are not met:
1. [C1] Hypothesis H3 confirmed without verify — dispatch a verify worker.
2. [J3] `ts-config-service` is anomalous but not explained by your causal chain.
Address these before attempting to finalize again.
</finalize_blocked>
```

---

## 10. Configuration

Sanitizer behavior is configurable in the scenario YAML:

```yaml
orchestrator:
  sanitizer:
    enabled: true
    critic_model: "gpt-5.1-mini"    # model for CriticSanitizer
    periodic_interval: 5             # run coverage checks every N rounds
    drift_window: 3                  # rounds to detect drift
    drift_threshold: 3               # consecutive same-target dispatches
    block_on:                        # which codes are BLOCK-level
      - C1   # skipped verify
      - C2   # unresolved contradiction
      - J3   # incomplete causal chain
    warn_on:                         # which codes are WARN-level (default for unlisted)
      - E1
      - E2
      - E3
      - C4
      - J2
      - P1
    disable:                         # turn off specific checks
      - P3
```

---

## 11. Constraints and Decisions

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| Middleware-based, not pipeline stage | Integrates with existing architecture; no changes to SimpleAgentLoop | Separate pre/post processor outside the loop |
| Two-layer (code + critic) | Code checks are free and fast; critic checks add cost but catch semantic issues | All-code (misses semantic checks) or all-LLM (expensive) |
| Findings as advisory except at finalize | Preserves model autonomy during investigation; only hard-gate the conclusion | Hard-gate every round (too restrictive, slows investigation) |
| InvestigationTracker as new component | Existing stores lack temporal ordering needed for sequence checks | Extend existing stores (violates SRP) |
| Critic uses scenario's compression model | Already configured, cheap, no extra model config needed | Dedicated critic model (adds config complexity) |
| Sanitizer is scenario-specific, not SDK | Different scenarios need different checks; RCA checks don't apply to general_purpose | SDK-level sanitizer (over-generalized) |

---

## 12. Related Concepts

- [Middleware System](middleware-system.md) — SanitizerMiddleware extends MiddlewareBase
- [Orchestrator](orchestrator.md) — Sanitizer operates within the orchestrator's SimpleAgentLoop
- [Scenario Protocol](sdk-consistency.md) — Sanitizer is wired via ScenarioWiring.orchestrator_middleware
- [Loop Resilience](loop-resilience.md) — Sanitizer complements existing resilience mechanisms (budget, loop detection)
- [Trajectory Judger](trajectory-judger.md) — Error taxonomy aligns with judger's classification categories

---

## 13. Resolved Design Questions

### Q1: CriticSanitizer Sync vs Async — Split by Trigger

CriticSanitizer execution mode depends on the trigger point:

| Trigger | Mode | Rationale |
|---------|------|-----------|
| **Pre-finalize** (C3, J1, J4) | **Synchronous** | Hard gate — must have results before deciding to allow finalize |
| **dispatch_agent** (P2) | **Asynchronous** | Advisory feedback; one-round delay is acceptable; dispatch frequency is high, blocking would significantly slow investigation |
| **Periodic** (J1, J4) | **Asynchronous** | Advisory; non-urgent |

Async implementation: CriticSanitizer launches the LLM call as a background task. SanitizerMiddleware collects completed async results on the next `on_llm_start` and injects them alongside any new synchronous findings.

### Q2: BLOCK Retry Limit — Degrade After 3 Attempts

After a BLOCK finding has blocked finalize **3 consecutive times** without resolution (same error code, same target):

1. The finding is **downgraded from BLOCK to WARN**
2. The finding message is annotated with `[DEGRADED: unresolved after 3 attempts]`
3. Finalize is allowed to proceed
4. The degradation event is recorded in InvestigationTracker for post-hoc analysis

Rationale: if the model cannot resolve a blocker after 3 tries, it has exceeded its capability for this check. Further blocking produces no new information and risks infinite loops. The WARN annotation ensures the issue remains visible in the output and in trajectory analysis.

Non-budget-related BLOCK codes (C1, C2) follow the same degradation rule — even process violations eventually yield, because a stuck agent produces no value.

### Q3: InvestigationTracker → TrajectoryCollector — Yes

InvestigationTracker events are written to TrajectoryCollector via a dedicated event type `"sanitizer"`. This enables:

- **Trajectory judger** can see sanitizer interactions (block → model response → resolution) as part of classification evidence
- **Trajectory analysis** can aggregate sanitizer findings across runs to identify systematic weaknesses
- **Post-hoc debugging** can trace exactly which checks fired, when, and how the model responded

Implementation: SanitizerMiddleware holds a reference to TrajectoryCollector (already available via SetupContext) and calls `trajectory.record()` after each check cycle with:
```python
{
    "event_type": "sanitizer",
    "data": {
        "trigger": "pre_finalize",
        "findings": [finding.to_dict() for finding in findings],
        "block_count": N,
        "degraded_codes": ["C1"],  # if any were degraded this round
    }
}
```

### Q4: Budget Exhaustion vs Coverage Gaps — Context-Dependent Degradation

When the orchestrator's tool/step budget is exhausted (detected via LoopContext):

| Finding category | Behavior | Rationale |
|-----------------|----------|-----------|
| **Coverage checks** (E1–E4) | **Degrade to WARN**, annotate `[budget_exhausted]` | Cannot dispatch more workers; blocking is pointless |
| **Process checks** (C1, C2, C4) | **Keep BLOCK** | These are not resource-constrained — e.g., C1 means no verify was ever dispatched, which is a process gap that existed before budget ran out |
| **Judgment checks** (J1, J3, J4) | **Degrade to WARN**, annotate `[budget_exhausted]` | Model cannot take corrective action; findings serve as output caveats |

The degradation is automatic: SanitizerMiddleware checks `ctx.tool_call_count >= tool_call_budget` (or step equivalent) before applying BLOCK severity. The original severity is preserved in the finding's `details` dict for trajectory analysis.

This means even a budget-exhausted investigation can still be blocked by C1/C2 — but only if the model tries to finalize without having done basic verification steps. In practice, if the model burned its entire budget without a single verify task, that is a meaningful signal worth blocking on.