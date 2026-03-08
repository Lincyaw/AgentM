# Design: Orchestrator

**Status**: DRAFT
**Last Updated**: 2026-03-08

---

## Overview

The Orchestrator is the **Supervisor node** in the Root StateGraph, acting as a **Team Leader / Hypothesis Reasoner**. It maintains a `DiagnosticNotebook` as working memory, drives hypothesis-driven reasoning, and dispatches tasks to Sub-Agents who only collect data.

### Core Responsibilities

1. **Hypothesis reasoning** — Generate and verify hypotheses using LLM
2. **Async task dispatch** — Submit multiple concurrent tasks to Sub-Agents (time-multiplexing)
3. **Notebook management** — Track hypotheses, evidence, and exploration history
4. **Monitoring & intervention** — Stream Sub-Agent execution, interrupt, inject instructions
5. **Phase orchestration** — Drive the Hypothetico-Deductive state machine

---

## Theoretical Foundation: Hypothetico-Deductive Method

The Orchestrator is modeled as a **state machine** following the **Hypothetico-Deductive Method** from experimental science:

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│    OBSERVE ───→ HYPOTHESIZE ───→ EXPERIMENT ───→ ANALYZE     │
│       ↑                                            │         │
│       │                                            ↓         │
│       │                                        CONCLUDE      │
│       │                                       ╱        ╲     │
│       │                               REFUTED           CONFIRMED
│       │                                 │                  │  │
│       └──── (new cycle) ←──────────────┘                  │  │
│                                                    [terminal] │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

Mapping to our four phases:

| Scientific Method | AgentM Phase | Actor | Description |
|-------------------|-------------|-------|-------------|
| **Observe** | Phase 1: Exploration | Sub-Agents | Collect raw data from multiple sources |
| **Hypothesize** | Phase 2: Generation | Orchestrator (LLM) | Generate candidate hypotheses from observations |
| **Experiment** | Phase 3: Verification | Sub-Agents | Investigate specific data to test each hypothesis |
| **Analyze** | Phase 3: Verification | Orchestrator (LLM) | Interpret results against hypothesis |
| **Conclude** | Phase 4: Confirmation | Orchestrator | Confirm root cause or refute and loop back |

### Confirmation Bias Mitigation

LLMs are prone to **confirmation bias** — they tend to find evidence supporting their own hypotheses. We address this structurally:

1. **No prediction anchoring** — Orchestrator does NOT set expected outcomes before experiments. This avoids the LLM anchoring on a predicted result and interpreting ambiguous data in its favor.

2. **Adversarial review** (feature-gated) — After verification, a separate Sub-Agent with a "Devil's Advocate" role attempts to refute the conclusion. This uses LLM's divergent capability against its own confirmation bias. See [Feature Gates](#feature-gates).

3. **Mandatory counter-evidence** — The three-block VerificationResult requires `rejecting_reasons` as a non-optional field. Sub-Agents must actively look for evidence AGAINST the hypothesis.

4. **Role separation** — The entity that generates hypotheses (Orchestrator) is different from the entity that collects evidence (Sub-Agent), reducing self-reinforcing loops.

---

## Message Management: Mode 2 (Minimal Messages + Notebook)

The Orchestrator uses **minimal messages** (2-3) combined with a structured **Notebook** as its primary working memory. This avoids context window explosion while maintaining complete diagnostic history.

### ExecutorState

```python
from dataclasses import dataclass, field
from typing import Annotated, Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import add_messages

@dataclass
class ExecutorState:
    # Minimal message list (2-3 messages, managed via RemoveMessage)
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)

    # Primary working memory
    notebook: DiagnosticNotebook = field(default_factory=DiagnosticNotebook)

    # Auxiliary
    task_id: str = ""
    current_phase: str = "exploration"
```

Key properties:
- `messages` uses `add_messages` reducer, supports `RemoveMessage` for cleanup
- `notebook` is the structured working memory, not stored in messages
- LLM receives the formatted Notebook, not a long conversation history
- Frontend reconstructs conversation display from Notebook

---

## DiagnosticNotebook

The Notebook is the Orchestrator's complete working memory, recording the entire diagnostic process.

### Core Structure

```python
@dataclass
class DiagnosticNotebook:
    task_id: str
    task_description: str
    start_time: str

    # Data collected from Sub-Agents (Phase 1)
    collected_data: dict[str, dict] = field(default_factory=dict)
    # Example: {"infrastructure": {"cpu": 0.85, ...}, "logs": {"error_count": 45, ...}}

    # Hypothesis management (Phase 2-3)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    hypothesis_verification_order: list[str] = field(default_factory=list)
    confirmed_hypothesis: Optional[str] = None

    # Complete exploration history (all phases)
    exploration_history: list[ExplorationStep] = field(default_factory=list)

    # Current state
    current_phase: Literal["exploration", "generation", "verification", "confirmation"] = "exploration"
    current_step: int = 0
```

### Hypothesis (No Confidence Scores)

Confidence scores from LLMs are unreliable. Instead, we use a **three-value verdict system**.

```python
@dataclass
class Hypothesis:
    id: str                              # "H1", "H2", ...
    description: str                     # "Database connection pool exhaustion"
    evidence: list[str]                  # Supporting evidence
    counter_evidence: list[str]          # Rejecting evidence
    status: Literal["active", "confirmed", "rejected", "partial"]
    created_at: str
    last_updated: str
```

### ExplorationStep

Records each step across all phases, with full context for trajectory export.

```python
@dataclass
class ExplorationStep:
    step_number: int
    phase: Literal["exploration", "generation", "verification", "confirmation"]
    action: str                                      # "Initial investigation", "Verify H1"
    timestamp: str

    # Phase 1: Exploration
    target_agents: Optional[list[str]] = None

    # Phase 2: Generation
    generated_hypotheses: Optional[list[Hypothesis]] = None

    # Phase 3: Verification (three-block record)
    target_hypothesis_id: Optional[str] = None
    investigation_data: Optional[dict] = None        # Block 1: Raw data
    reasoning: Optional[VerificationReasoning] = None # Block 2: Analysis
    verdict: Optional[Literal["confirmed", "rejected", "partial"]] = None  # Block 3

    hypothesis_before: Optional[Hypothesis] = None
    hypothesis_after: Optional[Hypothesis] = None

    # Phase 4: Confirmation
    confirmed_root_cause: Optional[str] = None
    recommendations: Optional[list[str]] = None
```

---

## Sub-Agent Verification Result (Three-Block Structure)

When a Sub-Agent verifies a hypothesis, it returns **three distinct blocks**:

### VerificationResult

```python
@dataclass
class VerificationResult:
    # Block 1: Raw investigation data (uninterpreted)
    investigation_data: dict[str, any]
    # Example: {"connection_pool_status": "100/100", "waiting_connections": 45, ...}

    # Block 2: Reasoning analysis
    reasoning: VerificationReasoning

    # Block 3: Final verdict
    verdict: Literal["confirmed", "rejected", "partial"]
```

### VerificationReasoning

```python
@dataclass
class VerificationReasoning:
    supporting_reasons: list[str]
    # ["Connection pool full (100/100)", "45 connections waiting", ...]

    rejecting_reasons: list[str]
    # ["CPU usage acceptable (45%)", "Memory sufficient (60% free)"]

    neutral_observations: list[str]
    # ["Network latency within normal range"]

    refined_description: Optional[str] = None
    # Used when verdict == "partial": refined hypothesis text

    key_findings: list[str]
    # ["Root cause is pool config (max=100)", "Recommend increasing to 150-200"]
```

### Usage Example

```python
result = VerificationResult(
    investigation_data={
        "connection_pool_status": "100/100",
        "active_connections": 100,
        "waiting_connections": 45,
        "query_timeout_errors": 12,
    },
    reasoning=VerificationReasoning(
        supporting_reasons=["Pool full (100/100)", "45 connections waiting"],
        rejecting_reasons=["CPU acceptable (45%)"],
        neutral_observations=["Network latency normal (<5ms)"],
        key_findings=["Pool config insufficient", "Recommend increase to 150-200"]
    ),
    verdict="confirmed"
)
```

---

## Four-Phase Implementation

### Phase 1: Exploration

Parallel data collection from multiple Sub-Agents.

```python
def phase_exploration(state: ExecutorState) -> dict:
    notebook = state.notebook

    tasks = [
        Task(agent="infrastructure", task="Scan infrastructure", depth="overview"),
        Task(agent="logs", task="Collect error logs", time_window="last_1h"),
        Task(agent="database", task="Check connection pool", depth="overview"),
    ]

    for task in tasks:
        result = await dispatch_to_agent(task.agent, task)
        notebook.collected_data[task.agent] = result.data  # Raw data only

    notebook.current_phase = "generation"
    notebook.current_step += 1

    return {
        "notebook": notebook,
        "current_phase": "generation",
        "messages": [RemoveMessage(id="__all__")]  # Clean up messages
    }
```

**Sub-Agent constraint**: Return raw data only. No reasoning.
- OK: `{"cpu": 0.85, "memory": 0.4}`
- NOT OK: `"I think CPU is high because..."`

### Phase 2: Hypothesis Generation

Orchestrator uses LLM to generate candidate hypotheses from collected data.

```python
def phase_hypothesis_generation(state: ExecutorState) -> dict:
    notebook = state.notebook

    prompt = f"""
    You are an RCA expert. Based on the following data, generate 3-5 root cause hypotheses.
    For each hypothesis, provide: description, supporting evidence, counter evidence.

    Collected data:
    {json.dumps(notebook.collected_data, indent=2)}
    """

    hypotheses_raw = await llm.invoke(prompt)

    for i, h in enumerate(hypotheses_raw):
        hypothesis = Hypothesis(
            id=f"H{i+1}",
            description=h["description"],
            evidence=h.get("supporting_reasons", []),
            counter_evidence=h.get("rejecting_reasons", []),
            status="active",
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )
        notebook.hypotheses[hypothesis.id] = hypothesis

    notebook.hypothesis_verification_order = list(notebook.hypotheses.keys())
    notebook.current_phase = "verification"
    notebook.current_step += 1

    return {"notebook": notebook, "current_phase": "verification"}
```

### Phase 3: Hypothesis Verification

Verify hypotheses one-by-one. Sub-Agent returns the three-block VerificationResult.

```python
def phase_hypothesis_verification(state: ExecutorState) -> dict:
    notebook = state.notebook

    for hypothesis_id in notebook.hypothesis_verification_order:
        hypothesis = notebook.hypotheses[hypothesis_id]
        if hypothesis.status != "active":
            continue

        agent_name = select_agent_for_verification(hypothesis)
        result = await dispatch_to_agent(agent_name, f"Verify {hypothesis_id}")
        # result is a VerificationResult (three blocks)

        if result.verdict == "confirmed":
            hypothesis.status = "confirmed"
            hypothesis.evidence.extend(result.reasoning.supporting_reasons)
            notebook.confirmed_hypothesis = hypothesis_id
            break

        elif result.verdict == "rejected":
            hypothesis.status = "rejected"
            hypothesis.counter_evidence.extend(result.reasoning.rejecting_reasons)

        elif result.verdict == "partial":
            hypothesis.status = "partial"
            hypothesis.description = result.reasoning.refined_description or hypothesis.description
            hypothesis.evidence.extend(result.reasoning.supporting_reasons)
            hypothesis.counter_evidence.extend(result.reasoning.rejecting_reasons)

        # Record exploration step with all three blocks
        notebook.exploration_history.append(ExplorationStep(
            step_number=notebook.current_step,
            phase="verification",
            action=f"Verify {hypothesis_id}",
            timestamp=datetime.now().isoformat(),
            target_hypothesis_id=hypothesis_id,
            investigation_data=result.investigation_data,
            reasoning=result.reasoning,
            verdict=result.verdict,
        ))
        notebook.current_step += 1

    notebook.current_phase = "confirmation"
    return {"notebook": notebook, "current_phase": "confirmation"}
```

### Phase 4: Confirmation

Output the final diagnostic result.

```python
def phase_confirmation(state: ExecutorState) -> dict:
    notebook = state.notebook

    if notebook.confirmed_hypothesis:
        root_cause = notebook.hypotheses[notebook.confirmed_hypothesis]
        output = {
            "status": "SUCCESS",
            "root_cause": root_cause.description,
            "evidence": root_cause.evidence,
            "recommendations": generate_recommendations(root_cause),
        }
    else:
        output = {
            "status": "INCONCLUSIVE",
            "hypotheses_eliminated": [
                h.description for h in notebook.hypotheses.values()
                if h.status == "rejected"
            ],
        }

    return {"notebook": notebook, "current_phase": "confirmation"}
```

---

## Async Task Dispatch

The Orchestrator operates like a real team leader — it can **submit multiple concurrent tasks** and collect results asynchronously, rather than waiting for each task to complete sequentially.

### Dispatch Model

```python
# Orchestrator submits multiple tasks concurrently via LangGraph Send API
def orchestrator_dispatch(state: ExecutorState):
    notebook = state.notebook
    pending_tasks = plan_next_tasks(notebook)

    # Fan-out: dispatch all tasks in parallel
    return [
        Send(task.agent, {
            "task_id": task.id,
            "hypothesis_id": task.hypothesis_id,
            "instruction": task.instruction,
        })
        for task in pending_tasks
    ]
```

### Concurrency Patterns

| Pattern | When | LangGraph Mechanism |
|---------|------|-------------------|
| **Parallel fan-out** | Phase 1 (explore all agents at once) | `Send()` from conditional edge |
| **Parallel verification** | Multiple independent hypotheses | `Send()` to different agents |
| **Sequential with review** | Verification + adversarial review | Agent → Orchestrator → Review Agent |
| **Fire and forget** | Low-priority background checks | `Send()` with results aggregated later |

### Task Queue in Notebook

```python
@dataclass
class PendingTask:
    id: str
    agent: str
    hypothesis_id: Optional[str]
    instruction: str
    status: Literal["pending", "dispatched", "completed", "failed"]
    dispatched_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None

# DiagnosticNotebook gains a task queue
@dataclass
class DiagnosticNotebook:
    # ... existing fields ...
    pending_tasks: list[PendingTask] = field(default_factory=list)
```

The Orchestrator checks pending_tasks at each decision point, dispatches ready tasks, and processes completed results — **time-multiplexing** like a human manager.

### Results Aggregation

```python
class ExecutorState(TypedDict):
    # ... existing fields ...
    # Reducer accumulates results from parallel Sub-Agents
    agent_results: Annotated[list[dict], operator.add]
```

When multiple Sub-Agents complete, their results are accumulated via the reducer. The Orchestrator processes them in the next decision cycle.

---

## Feature Gates

Diagnostic behaviors are configurable via **feature gates** — boolean flags in config that enable/disable specific capabilities without code changes.

### Configuration

```yaml
# orchestrator.yaml
orchestrator:
  feature_gates:
    # Adversarial review: after verification, a Devil's Advocate
    # Sub-Agent attempts to refute the conclusion
    adversarial_review: true

    # Parallel hypothesis verification: test multiple hypotheses
    # concurrently instead of sequentially
    parallel_verification: false

    # Auto-refine: when verdict is "partial", automatically
    # generate a refined hypothesis and re-verify
    auto_refine_partial: true

    # Minimum verifications: require at least N hypotheses
    # to be tested before accepting a "confirmed" result
    min_verifications_before_confirm: 2

    # Exploration depth: run a second exploration round
    # if initial data is insufficient for hypothesis generation
    deep_exploration: false
```

### Adversarial Review (Devil's Advocate)

When `adversarial_review: true`, after the Orchestrator reaches a verdict, it dispatches a **separate Sub-Agent** with a Devil's Advocate role to challenge the conclusion:

```
Normal flow:
  Sub-Agent → VerificationResult → Orchestrator verdict

With adversarial review:
  Sub-Agent → VerificationResult → Orchestrator preliminary verdict
      → Devil's Advocate Agent → ChallengeResult
      → Orchestrator final verdict
```

The adversarial agent:
- Receives the raw investigation data and the preliminary verdict
- Does NOT see the original hypothesis description (prevents anchoring)
- Is prompted to find inconsistencies, alternative explanations, and overlooked evidence
- Returns a `ChallengeResult` with counter-arguments

```python
@dataclass
class ChallengeResult:
    counter_arguments: list[str]     # Arguments against the verdict
    alternative_explanations: list[str]  # Other possible causes
    overlooked_evidence: list[str]   # Data points that were ignored
    challenge_strength: Literal["weak", "moderate", "strong"]
    # "strong" challenge forces verdict downgrade (confirmed → partial)
```

This is **also a Sub-Agent task** — dispatched asynchronously like any other. The Orchestrator waits for both the verification result and the adversarial review before making the final verdict.

### Feature Gate Access in Code

```python
def phase_hypothesis_verification(state: ExecutorState, config: dict) -> dict:
    features = config["orchestrator"]["feature_gates"]

    # Choose dispatch strategy based on feature gates
    if features.get("parallel_verification"):
        # Fan-out: verify all active hypotheses in parallel
        return dispatch_parallel_verification(state)
    else:
        # Sequential: verify one by one
        return dispatch_sequential_verification(state)

def process_verification_result(state, result, config):
    features = config["orchestrator"]["feature_gates"]

    # Preliminary verdict
    verdict = result.verdict

    # Adversarial review if enabled
    if features.get("adversarial_review") and verdict == "confirmed":
        challenge = await dispatch_adversarial_review(result)
        if challenge.challenge_strength == "strong":
            verdict = "partial"  # Downgrade

    # Minimum verification check
    min_checks = features.get("min_verifications_before_confirm", 1)
    verified_count = count_verified_hypotheses(state.notebook)
    if verdict == "confirmed" and verified_count < min_checks:
        # Don't confirm yet, continue verifying others
        verdict = "partial"

    return verdict
```

---

## LLM Decision-Making

The Orchestrator uses LLM to make routing decisions. Prompts are loaded from config files (Jinja2 templates):

```python
def orchestrator_node(state: ExecutorState) -> Command[...]:
    system_prompt = load_prompt_template(
        config["prompts"]["system"],
        notebook=state.notebook,
        phase=state.current_phase,
    )

    response = model.invoke([
        SystemMessage(content=system_prompt),
        *state.messages,
    ])

    decision = parse_decision(response.content)
    return Command(goto=decision["next_agent"])
```

Prompt templates are external files — switching scenarios requires **only new config + prompt files**, zero code changes.

---

## Monitoring & Intervention

### Real-Time Streaming

```python
async def monitor_execution(graph, config, websocket):
    async for namespace, mode, data in graph.astream(
        input_data, config,
        stream_mode=["updates", "custom"],
        subgraphs=True,  # Capture Sub-Agent events
    ):
        agent_path = list(namespace) if namespace else ["orchestrator"]
        await websocket.send_json({"agent_path": agent_path, "data": data})
```

### Intervention Mechanisms

```python
# Interrupt Sub-Agent
state = graph.get_state(config)

# Inject new instruction
graph.update_state(config, {
    "messages": state.values["messages"] + [
        HumanMessage(content="[Orchestrator] New instruction: focus on Section 3")
    ],
})

# Resume execution
result = graph.invoke(None, config)

# Redirect to different Sub-Agent
result = graph.invoke(Command(goto="database_agent"), config)
```

---

## Frontend Conversation Reconstruction

Since messages are minimal, the frontend reconstructs conversation from Notebook:

```typescript
function rebuildConversation(messages: Message[], notebook: DiagnosticNotebook): Message[] {
  const conversation = [...messages];

  for (const step of notebook.exploration_history) {
    if (step.phase === "exploration") {
      conversation.push({
        role: "assistant",
        content: `[Phase 1] Dispatched to: ${step.target_agents?.join(", ")}`,
      });
    } else if (step.phase === "generation") {
      conversation.push({
        role: "assistant",
        content: `[Phase 2] Generated hypotheses:\n${
          step.generated_hypotheses?.map(h => `- ${h.description}`).join("\n")
        }`,
      });
    } else if (step.phase === "verification") {
      conversation.push({
        role: "assistant",
        content: `[Phase 3] Verify ${step.target_hypothesis_id}\n` +
          `Supporting: ${step.reasoning?.supporting_reasons?.join("; ")}\n` +
          `Rejecting: ${step.reasoning?.rejecting_reasons?.join("; ")}\n` +
          `Verdict: ${step.verdict}`,
      });
    } else if (step.phase === "confirmation") {
      conversation.push({
        role: "assistant",
        content: `[Phase 4] Root cause: ${
          notebook.hypotheses[step.confirmed_root_cause!]?.description
        }`,
      });
    }
  }

  return conversation;
}
```

---

## Configuration

```yaml
# orchestrator.yaml
orchestrator:
  model: "gpt-4"
  temperature: 0.3

  prompts:
    system: "templates/orchestrator_system.txt"
    hypothesis_generation: "templates/generate_hypotheses.txt"
    verification_task: "templates/verify_hypothesis.txt"
    adversarial_review: "templates/adversarial_review.txt"

  tools:
    - dispatch_task
    - interrupt_agent
    - inject_instruction

  monitoring:
    enabled: true
    stream_mode: ["updates", "custom"]
    max_execution_time: 3600

  intervention:
    allow_interruption: true
    allow_instruction_injection: true

  feature_gates:
    adversarial_review: true
    parallel_verification: false
    auto_refine_partial: true
    min_verifications_before_confirm: 2
    deep_exploration: false
```

---

## Related Documents

- [System Architecture](system-design-overview.md) — Overall system design
- [Sub-Agent](sub-agent.md) — Sub-Agent architecture and configuration
- [Generic State Wrapper](generic-state-wrapper.md) — SDK framework for multiple diagnostic patterns

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Hypothetico-Deductive state machine | Formal scientific method foundation; clear phase transitions |
| Orchestrator = reasoner, Sub-Agent = data collector | Clear separation; all reasoning traceable in one place |
| No confidence scores | LLM-generated confidence is unreliable; three-value verdict instead |
| No prediction anchoring | Avoids confirmation bias; Orchestrator does not set expected outcomes |
| Adversarial review via Sub-Agent | Devil's Advocate challenges conclusions; uses LLM divergence against its own bias |
| Async task dispatch (Send API) | Orchestrator submits concurrent tasks like a real team leader; time-multiplexing |
| Feature gates for behaviors | All diagnostic strategies are config-toggleable; no code changes |
| Notebook as working memory | Structured, serializable, supports RL export and failure recovery |
| Minimal messages (Mode 2) | Avoids context window explosion; Notebook is the real memory |
| Three-block verification result | Cleanly separates raw data / reasoning / verdict |
| Prompt templates (Jinja2) | Scenario switching with zero code changes |
