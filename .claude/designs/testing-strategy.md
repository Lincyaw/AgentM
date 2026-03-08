# Design: Testing Strategy

**Status**: APPROVED
**Created**: 2026-03-08
**Last Updated**: 2026-03-08

## Overview

Testing strategy for the AgentM RCA agent system. Defines a four-layer testing architecture, snapshot-based behavioral testing mechanism, and LLM-as-Judge evaluation framework.

## Motivation

Agent systems combine deterministic code paths with non-deterministic LLM decisions. Traditional unit tests cannot verify decision quality, and end-to-end tests are slow and brittle. We need a layered approach that tests deterministic invariants cheaply, and evaluates LLM behavior with structured scenarios.

LangGraph's checkpoint mechanism enables **snapshot-based testing** — we can construct any intermediate system state and verify behavior from that point forward, without replaying the entire history.

## Testing Philosophy

1. **Test behavior, not structure** — Never test language guarantees (enum membership, dataclass defaults, import success)
2. **Every test must answer "what bug does this prevent?"** — If you can't articulate a realistic failure scenario, don't write the test
3. **Boundaries over happy paths** — Edge cases and error conditions are where bugs live
4. **Adding a test requires deliberation** — Each test case is a maintenance commitment; add only what earns its keep

## Test Layer Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 4: E2E Scenario Eval                                 │
│  Complete RCA scenario → LLM-as-Judge evaluates outcome      │
│  Frequency: pre-release / nightly                            │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Decision Eval                                      │
│  Snapshot → 1 LLM step → LLM-as-Judge evaluates decision     │
│  Frequency: per-feature / weekly                             │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Pipeline Integration                               │
│  Snapshot → Mock LLM → verify data flow and error handling   │
│  Frequency: per-commit (CI)                                  │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Deterministic Invariants                           │
│  Pure functions, state machine rules, design contracts       │
│  Frequency: per-commit (CI)                                  │
└──────────────────────────────────────────────────────────────┘
```

### Layer 1: Deterministic Invariants

**What**: Pure logic that must always hold, regardless of LLM behavior.

**How**: Standard pytest assertions. No LLM calls, no mocks.

**Scope**:
- Design-document contracts (enum values match tool Literal constraints)
- Notebook immutability (operations return new instance, original unchanged)
- State field completeness (ExecutorState/SubAgentState have all required fields)
- Config validation boundaries (mandatory fields rejected when missing)
- Path utility round-trips (namespace ↔ path conversion consistency)
- HypothesisStatus state machine (illegal transitions rejected)

**Current tests**: `tests/unit/test_design_contracts.py`, `test_knowledge_path_utils.py`, `test_state_registry.py`, `test_config_validation.py`

### Layer 2: Pipeline Integration (Mock LLM)

**What**: Verify the data flow pipeline works correctly — tools update state, errors are handled, retry policies fire, data doesn't get lost. The LLM is mocked to return predetermined tool calls.

**How**: Construct a checkpoint snapshot, mock the LLM to return specific tool calls, run the graph for N steps, assert state invariants.

**Scope**:
- Tool → Command → State update pipeline (dispatch_agent creates task + updates Notebook)
- check_tasks data flow (completed task results land in Notebook.collected_data)
- Error recovery pipeline (Sub-Agent failure → retry → eventual FAILED with error_summary)
- Compression trigger and recall (pre_model_hook fires, recall_history retrieves compressed data)
- inject_instruction delivery (instruction queued → consumed by pre_model_hook → appears in LLM input)
- abort_task on completed/failed task (rejected, not silently ignored)

### Layer 3: Decision Eval (LLM-as-Judge)

**What**: Evaluate whether the Orchestrator LLM makes reasonable decisions given a specific Notebook state. One snapshot → one LLM reasoning step → judge evaluates the decision.

**How**: Construct a Notebook snapshot, let the real Orchestrator LLM run one step (produce tool calls), then pass the (snapshot, decision) pair to a Judge LLM for evaluation.

**Scope**:
- Cold start decision quality (empty Notebook → should dispatch scouts, not form hypotheses)
- Hypothesis formation quality (data available → hypotheses should relate to data, not hallucinate)
- Verification dispatch (hypothesis formed → should dispatch verify, not more scouts)
- Contradictory evidence handling (REJECTED verdict → should not confirm)
- Repeated failure handling (all Sub-Agents failed → should adapt strategy, not infinite retry)
- Garbage data recognition (Sub-Agent returns irrelevant data → should recognize and re-dispatch)
- VerificationResult self-consistency (verdict=CONFIRMED but report contradicts → should distrust)

### Layer 4: E2E Scenario Eval

**What**: Run a complete RCA scenario end-to-end with mock tools (returning realistic but controlled data), then evaluate the final outcome.

**How**: Define a fault scenario (e.g., DB connection pool exhaustion) with scripted tool responses. Run the full agent system. Judge LLM evaluates whether the confirmed root cause is correct and the reasoning chain is sound.

**Scope**:
- Known-root-cause scenarios (system produces correct confirmed_hypothesis)
- Multi-hypothesis scenarios (system doesn't tunnel-vision on first hypothesis)
- Ambiguous scenarios (system reaches INCONCLUSIVE rather than wrong CONFIRMED)
- Time-to-resolution (system converges within reasonable step count)

---

## Snapshot Testing Mechanism

### Principle

LangGraph checkpoints capture the complete graph state at every super-step. By constructing a checkpoint and calling `graph.invoke(None, config)`, we resume execution from that exact state. This enables:

1. **Targeted testing** — Test verification behavior without first running exploration
2. **Reproducibility** — Same snapshot always produces deterministic code paths (LLM behavior varies, but pipeline logic is fixed)
3. **Isolation** — Each test is independent; no ordering dependencies

### Snapshot Construction

```python
def build_snapshot(
    notebook: DiagnosticNotebook,
    messages: list[BaseMessage] | None = None,
    compression_refs: list[CompressionRef] | None = None,
) -> ExecutorState:
    """Construct an ExecutorState snapshot for testing."""
    return {
        "messages": messages or [],
        "notebook": notebook,
        "task_id": notebook.task_id,
        "current_phase": notebook.current_phase,
        "compression_refs": compression_refs or [],
    }
```

### Snapshot Injection

```python
# Option A: Direct graph.invoke with constructed state as input
result = graph.invoke(snapshot, config)

# Option B: Inject into checkpointer, then resume
graph.update_state(config, snapshot)
result = graph.invoke(None, config)  # Resume from injected state
```

### Assertion Patterns

```python
# 1. State invariant assertion (Layer 1-2)
result = graph.invoke(snapshot, config)
assert result["notebook"].confirmed_hypothesis is None  # Should not confirm prematurely

# 2. Tool call assertion (Layer 2)
tool_calls = extract_tool_calls(result["messages"])
assert any(tc.name == "dispatch_agent" and tc.args["task_type"] == "scout" for tc in tool_calls)

# 3. LLM-as-Judge assertion (Layer 3-4)
await assert_ai(
    scenario="Empty Notebook, only task_description set",
    actual=format_tool_calls(result),
    expected="Should dispatch scout tasks to collect data, should NOT form hypotheses",
)
```

---

## assert_ai — LLM-as-Judge Assertion Primitive

### Design

`assert_ai()` is the core assertion function for semantic verification. It calls a Judge LLM to evaluate whether observed behavior meets a natural-language expectation. On failure, it raises `AssertionError` with the judge's reasoning, just like a normal `assert`.

```python
@dataclass
class JudgeVerdict:
    passed: bool
    reasoning: str
    confidence: float  # 0-1

async def assert_ai(
    actual: str,
    expected: str,
    scenario: str = "",
    model: str = "gpt-4o",
    retries: int = 3,
    confidence_threshold: float = 0.7,
) -> None:
    """Assert that actual behavior meets expected criteria, judged by LLM.

    Args:
        actual: Description of what the system actually did (tool calls, state changes, output).
        expected: Natural-language criteria the behavior should satisfy.
        scenario: Context about the test setup (Notebook state, preconditions).
        model: Judge LLM model.
        retries: Number of judge invocations; majority vote decides.
        confidence_threshold: Below this, flag for human review instead of auto-pass/fail.

    Raises:
        AssertionError: If majority of judge runs return FAIL, with reasoning attached.
    """
    verdicts = []
    for _ in range(retries):
        verdict = await _invoke_judge(actual, expected, scenario, model)
        verdicts.append(verdict)

    passes = sum(1 for v in verdicts if v.passed)
    majority_pass = passes > retries // 2

    # Low-confidence verdicts get flagged, not auto-decided
    avg_confidence = sum(v.confidence for v in verdicts) / len(verdicts)
    if avg_confidence < confidence_threshold:
        import warnings
        warnings.warn(
            f"assert_ai low confidence ({avg_confidence:.2f}): "
            f"scenario={scenario!r}, flagged for human review"
        )
        return  # Don't fail on low-confidence judgments

    if not majority_pass:
        reasons = "\n".join(f"  [{i+1}] {v.reasoning}" for i, v in enumerate(verdicts))
        raise AssertionError(
            f"assert_ai FAILED ({passes}/{retries} passed)\n"
            f"Scenario: {scenario}\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}\n"
            f"Judge reasoning:\n{reasons}"
        )
```

### Usage Examples

```python
# Layer 3: Single-step decision eval
result = await orchestrator.invoke(snapshot, config)
tool_calls = extract_tool_calls(result)

await assert_ai(
    scenario="Notebook has collected_data showing DB connections 200/200, CPU normal",
    actual=format_tool_calls(tool_calls),
    expected="Should form hypotheses related to database connection exhaustion, "
             "not CPU or memory issues",
)

# Layer 4: End-to-end scenario eval
final_state = await agent_system.execute(scenario_input)

await assert_ai(
    scenario="DB connection pool exhaustion due to leak in svc-orders v2.3.1",
    actual=format_notebook(final_state["notebook"]),
    expected="Confirmed root cause should identify connection leak in svc-orders. "
             "Reasoning chain should trace: DB anomalous → recent deploy correlation → "
             "pool exhaustion verified → leak in specific service version",
)

# Can also verify negative conditions
await assert_ai(
    scenario="Hypothesis H1 REJECTED by verification with clear counter-evidence",
    actual=format_tool_calls(tool_calls),
    expected="Must NOT call update_hypothesis with status=confirmed for H1. "
             "Should reject, refine, or form a new hypothesis",
)
```

### Judge Prompt Template

### Judge Prompt Template

```jinja2
You are an expert evaluator for an AI-powered Root Cause Analysis system.

## Scenario
{{ scenario }}

## Observed System Behavior
{{ behavior }}

## Expected Criteria
{{ criteria }}

## Your Task
Evaluate whether the observed behavior meets the expected criteria.

Answer in this exact format:
VERDICT: PASS or FAIL
CONFIDENCE: 0.0 to 1.0
REASONING: <brief explanation of why the behavior does or does not meet the criteria>
```

### Determinism Control

LLM-as-Judge is inherently non-deterministic. Mitigation strategies:

1. **Temperature 0** for judge LLM — minimize variation
2. **Binary criteria** — criteria should be clearly pass/fail, not subjective quality scores
3. **Multiple runs** — for critical evals, run 3 times and take majority vote
4. **Escape hatch** — if judge returns low confidence (< 0.7), flag for human review rather than auto-fail

---

## Core Test Cases

### Layer 1: Deterministic Invariants

These tests exist today. See [current tests](#layer-1-deterministic-invariants) scope.

### Layer 2: Pipeline Integration (Snapshot + Mock LLM)

#### P1: dispatch_agent updates Notebook transparently

**Snapshot**: Empty Notebook, no tasks running.
**Mock LLM**: Returns `dispatch_agent(agent_id="db", task="check connections", task_type="scout")`.
**Assert**:
- Notebook.exploration_history has a new ExplorationStep recording the dispatch
- TaskManager has a new ManagedTask with status RUNNING
- Returned ToolMessage contains the task_id

**Bug prevented**: dispatch_agent fires but Notebook not updated → Orchestrator loses track of what it dispatched.

#### P2: check_tasks flows completed data into Notebook

**Snapshot**: Notebook with one dispatched task. TaskManager has that task as COMPLETED with result data.
**Mock LLM**: Returns `check_tasks()`.
**Assert**:
- Notebook.collected_data[agent_id] contains the task result
- Notebook.exploration_history has a new step recording the completion
- Returned ToolMessage contains the result summary

**Bug prevented**: check_tasks returns data to LLM via ToolMessage but doesn't persist it in Notebook → data lost on next compression.

#### P3: Sub-Agent failure triggers retry, then surfaces to Orchestrator

**Snapshot**: Sub-Agent task running.
**Simulate**: Sub-Agent raises exception (e.g., API 429).
**Assert**:
- TaskManager retries up to max_attempts with exponential backoff
- After max_attempts exhausted, task status is FAILED with error_summary
- check_tasks returns the failure with error_summary and last_steps
- Orchestrator receives enough context to make a re-dispatch decision

**Bug prevented**: API error silently swallowed → Orchestrator waits forever for a result that will never come.

#### P4: inject_instruction rejected for completed task

**Snapshot**: TaskManager has a COMPLETED task.
**Action**: Call inject_instruction(task_id=completed_task).
**Assert**: Raises error or returns failure message. Does NOT silently queue the instruction.

**Bug prevented**: Orchestrator thinks instruction was delivered, but task already finished → Orchestrator's mental model diverges from reality.

#### P5: abort_task on running task sets FAILED

**Snapshot**: TaskManager has a RUNNING task with an active asyncio.Task.
**Action**: Call abort_task(task_id, reason="timeout").
**Assert**:
- Task status becomes FAILED
- error_summary contains the abort reason
- asyncio.Task is cancelled

**Bug prevented**: abort_task returns success but task continues running in background → resource leak and stale data.

#### P6: Compression preserves data for recall

**Snapshot**: Orchestrator has run 20+ steps, compression triggered, CompressionRef recorded.
**Action**: Call recall_history(query="CPU metrics from agent-infra").
**Assert**: Response contains data from pre-compression checkpoint range.

**Bug prevented**: Compression discards checkpoint references → recall_history returns empty → Orchestrator loses access to early investigation data.

#### P7: update_hypothesis rejects illegal state transitions

**Snapshot**: Notebook has hypothesis H1 with status REJECTED.
**Action**: Call update_hypothesis(id="H1", status="confirmed").
**Assert**: Rejected — REJECTED cannot transition directly to CONFIRMED.

**Bug prevented**: LLM hallucinates a direct REJECTED→CONFIRMED transition → wrong root cause confirmed without re-investigation.

**Legal transitions**:
```
formed → investigating
investigating → confirmed | rejected | refined | inconclusive
refined → investigating (new cycle)
inconclusive → investigating (re-investigate)
rejected → (terminal, or → refined to start new hypothesis)
confirmed → (terminal)
```

#### P8: set_confirmed_hypothesis enforces consistency

**Snapshot**: Notebook has hypothesis H1 with status INVESTIGATING (not yet CONFIRMED).
**Action**: Call set_confirmed_hypothesis("H1").
**Assert**: Rejected — cannot confirm a hypothesis that hasn't been verified and CONFIRMED first.

**Bug prevented**: Root cause marked as confirmed but underlying hypothesis still in INVESTIGATING state → inconsistent Notebook.

### Layer 3: Decision Eval (Snapshot + Real LLM + Judge)

#### D1: Cold start — Orchestrator dispatches scouts

**Snapshot**: Empty Notebook with only task_description="Database connection timeouts reported by users".
**Run**: One Orchestrator reasoning step (real LLM).
**Judge criteria**: "The system should dispatch one or more scout tasks to collect initial data. It should NOT form any hypotheses yet because no data has been collected."

**Bug prevented**: Orchestrator jumps to conclusions without evidence.

#### D2: Data collected — Orchestrator forms relevant hypotheses

**Snapshot**: Notebook.collected_data contains:
- agent-infra: {cpu: "12%", memory: "45%", disk: "30%"}
- agent-db: {active_connections: "200/200", wait_queue: "47", avg_query_time: "12.3s"}
- agent-app: {error_rate: "23%", p99_latency: "15s", top_error: "ConnectionTimeout"}

**Run**: One Orchestrator reasoning step.
**Judge criteria**: "The hypotheses formed should relate to database connection exhaustion or query performance, since DB metrics are clearly anomalous while infrastructure metrics are normal. Hypotheses about CPU or memory issues would be unreasonable given the data."

**Bug prevented**: Orchestrator ignores obvious signals in collected data.

#### D3: Verification REJECTED — Orchestrator does not confirm

**Snapshot**: Notebook has hypothesis H1 (status=INVESTIGATING, description="Connection pool exhaustion"). check_tasks just returned a verify result: verdict=REJECTED, report="Connection pool shows 50/200 active, well within limits. Wait queue is 0. The pool is NOT exhausted."

**Run**: One Orchestrator reasoning step.
**Judge criteria**: "The system must NOT confirm hypothesis H1. It should either reject it, mark it inconclusive, or refine it into a different hypothesis. Confirming a hypothesis that was just contradicted by verification evidence would be a critical error."

**Bug prevented**: Orchestrator ignores contradictory evidence and confirms anyway (confirmation bias).

#### D4: All Sub-Agents failed — Orchestrator adapts

**Snapshot**: Notebook shows 3 dispatched scout tasks, all FAILED with error_summaries indicating API timeouts. No collected data.

**Run**: One Orchestrator reasoning step.
**Judge criteria**: "The system should either (a) retry with different agents or parameters, (b) try a different diagnostic approach, or (c) report that it cannot proceed due to infrastructure issues. It should NOT dispatch the same tasks with the same parameters to the same agents that just failed."

**Bug prevented**: Orchestrator enters infinite retry loop with identical failing parameters.

#### D5: Contradictory evidence — Orchestrator investigates further

**Snapshot**: Notebook has hypothesis H1 (status=INVESTIGATING). collected_data contains:
- agent-db: "connection pool 200/200, pool IS exhausted" (supports H1)
- agent-app: "connection pool shows 50/200 active, pool NOT exhausted" (contradicts H1)

**Run**: One Orchestrator reasoning step.
**Judge criteria**: "The system should recognize the contradictory evidence and either (a) dispatch a deep_analyze task to resolve the discrepancy, (b) refine the hypothesis, or (c) dispatch additional verification. It should NOT confirm based on partial/contradictory evidence."

**Bug prevented**: Orchestrator picks whichever Sub-Agent report it sees last and ignores the contradiction.

#### D6: VerificationResult self-inconsistency — Orchestrator distrusts

**Snapshot**: check_tasks returned a verify result with verdict=CONFIRMED but report="No evidence of connection pool exhaustion found. All metrics within normal range. Unable to reproduce the issue."

**Run**: One Orchestrator reasoning step.
**Judge criteria**: "The system should recognize that the verdict (CONFIRMED) contradicts the report content (no evidence found). It should distrust this result and either re-verify or mark the verification as inconclusive. It should NOT accept the CONFIRMED verdict at face value."

**Bug prevented**: Orchestrator blindly trusts the structured verdict field without reading the report.

#### D7: Sufficient verification — Orchestrator confirms

**Snapshot**: Notebook has hypothesis H1 (status=CONFIRMED after 2 successful verifications, with consistent supporting evidence from multiple agents). min_verifications_before_confirm=2 is satisfied.

**Run**: One Orchestrator reasoning step.
**Judge criteria**: "The system should set H1 as the confirmed root cause and transition to confirmation phase. It should NOT continue dispatching more verification tasks when the evidence is already sufficient and consistent."

**Bug prevented**: Orchestrator over-investigates when evidence is already conclusive → wastes time and resources.

### Layer 4: E2E Scenario Eval

#### E1: DB Connection Pool Exhaustion

**Scenario**: Users report intermittent database timeouts. Root cause is connection pool exhaustion due to a connection leak in a recently deployed microservice.

**Mock tool responses**:
- agent-infra scout: CPU 15%, memory 40%, disk 25% (normal)
- agent-db scout: active_connections 200/200, wait_queue 47, avg_query_time 12.3s (anomalous)
- agent-app scout: error_rate 23%, top_error "ConnectionTimeout", recent_deploy "svc-orders v2.3.1 deployed 2h ago"
- agent-db verify (pool exhaustion): connections grew linearly after deploy, no connection releases from svc-orders
- agent-app deep_analyze: svc-orders v2.3.1 changelog shows new DB connection per request, no connection pooling

**Judge criteria**: "The confirmed root cause should identify the connection leak in svc-orders v2.3.1 as the cause of connection pool exhaustion. The reasoning chain should include: (1) identified DB as anomalous, not infra; (2) correlated with recent deployment; (3) verified connection pool exhaustion; (4) traced leak to specific service version."

#### E2: False Lead Scenario

**Scenario**: Users report slow API responses. CPU spikes are observed but are a symptom, not the cause. Root cause is a downstream service degradation causing request queuing.

**Mock tool responses**:
- agent-infra scout: CPU 85% (looks anomalous — but it's a red herring)
- agent-app scout: p99_latency 8s, downstream_svc_latency 6s (real signal)
- agent-infra verify (CPU overload): CPU high due to request retries, not computation. Load drops when downstream recovers.
- agent-app verify (downstream degradation): downstream service deployed new version 3h ago, latency increased 10x

**Judge criteria**: "The system should NOT confirm CPU overload as root cause despite the initially anomalous reading. It should trace through the red herring to identify downstream service degradation as the true root cause. The reasoning should show that CPU was a symptom of retry storms, not an independent issue."

---

## Test Organization

```
tests/
├── unit/                              # Layer 1: Deterministic invariants
│   ├── test_design_contracts.py       # Enum/state contract compliance
│   ├── test_knowledge_path_utils.py   # Path utility boundaries
│   ├── test_state_registry.py         # Schema resolution
│   └── test_config_validation.py      # Config rejection boundaries
├── snapshot/                          # Layer 2: Pipeline integration
│   ├── conftest.py                    # Snapshot builders, mock LLM fixtures
│   ├── test_tool_pipeline.py          # P1, P2: tool → state update flow
│   ├── test_error_recovery.py         # P3, P4, P5: failure and retry paths
│   ├── test_compression_recall.py     # P6: compression + recall integrity
│   └── test_hypothesis_lifecycle.py   # P7, P8: state machine + consistency
└── eval/                              # Layer 3-4: LLM-as-Judge
    ├── conftest.py                    # Judge framework, scenario builders
    ├── judge.py                       # JudgeVerdict, judge() function
    ├── test_decision_quality.py       # D1-D7: single-step decision evals
    └── test_rca_scenarios.py          # E1-E2: end-to-end scenario evals
```

## Cost and Timeout Budget

| Layer | API Calls per Test | Tests | Total Calls | Estimated Cost | Timeout |
|-------|-------------------|-------|-------------|----------------|---------|
| L1 | 0 | 29 | 0 | $0 | < 1s |
| L2 | 0 (Mock LLM) | 8 | 0 | $0 | < 10s |
| L3 | 1 real + 3 judge = 4 | 7 | 28 | ~$0.50 | < 30s per test |
| L4 | ~20 real + 3 judge = ~23 | 2 | ~46 | ~$2.00 | < 5min per test |

**Total per full suite**: ~74 API calls, ~$2.50, ~12 minutes.

**Budget rule**: If a single Layer 3-4 test exceeds its timeout, it auto-fails with a timeout error rather than hanging. Layer 4 tests set `recursion_limit` on the graph to cap agent steps.

## CI Integration

| Layer | CI Gate | Failure Action |
|-------|---------|----------------|
| Layer 1 | Blocking (per-commit) | Fix before merge |
| Layer 2 | Blocking (per-commit) | Fix before merge |
| Layer 3 | Non-blocking (weekly) | Create issue, investigate |
| Layer 4 | Non-blocking (pre-release) | Create issue, review with team |

Layer 3-4 are non-blocking because LLM-as-Judge has inherent variance. Repeated failures across runs indicate real regression; single failures may be noise.

## Related Concepts

- [Orchestrator](orchestrator.md) — Tools, Notebook, state machine
- [Sub-Agent](sub-agent.md) — State isolation, pre_model_hook, task types
- [System Design](system-design-overview.md) — Error handling, recovery, config validation
- [Generic State Wrapper](generic-state-wrapper.md) — AgentSystemBuilder, state registry

## Constraints and Decisions

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| Snapshot-based testing over sequential replay | Faster, isolated, reproducible; avoids running full pipeline for mid-flow tests | Full E2E replay — too slow for CI |
| LLM-as-Judge for decision quality | Agent decisions are semantic; cannot be verified with `assert x == y` | Human review — doesn't scale |
| Judge at temperature 0 | Minimize evaluation variance | Higher temperature — more diverse but less reliable |
| Majority vote (3 runs) for Layer 3-4 | Single LLM call may produce incorrect judgment | Single run — cheaper but less reliable |
| Layer 3-4 non-blocking in CI | LLM variance would cause flaky CI failures | Blocking — too many false positives |
| HypothesisStatus state machine enforced in tool layer | Prompt-only enforcement is unreliable; LLM may hallucinate illegal transitions | Prompt-only — simpler but fragile |

## Open Questions

- [ ] Should Layer 3-4 tests use the same model as the Orchestrator, or a different one to avoid self-evaluation bias?
- [ ] What is the acceptable false-positive rate for LLM-as-Judge before we reconsider the approach?
- [ ] Should we record eval trajectories for regression detection across model versions?
