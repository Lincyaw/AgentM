# Design: Orchestrator

**Status**: DRAFT
**Last Updated**: 2026-03-07

---

## Overview

The Orchestrator is the **Supervisor node** in the Root StateGraph, acting as a **Team Leader / Hypothesis Reasoner**. It maintains a `DiagnosticNotebook` as working memory, drives hypothesis-driven reasoning, and dispatches tasks to Sub-Agents who only collect data.

### Core Responsibilities

1. **Hypothesis reasoning** — Generate and verify hypotheses using LLM
2. **Sub-Agent dispatch** — Route tasks via `Command(goto="agent_name")`
3. **Notebook management** — Track hypotheses, evidence, and exploration history
4. **Monitoring & intervention** — Stream Sub-Agent execution, interrupt, inject instructions
5. **Phase orchestration** — Drive the four-phase diagnostic flow

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
| Orchestrator = reasoner, Sub-Agent = data collector | Clear separation; all reasoning traceable in one place |
| No confidence scores | LLM-generated confidence is unreliable; three-value verdict instead |
| Notebook as working memory | Structured, serializable, supports RL export and failure recovery |
| Minimal messages (Mode 2) | Avoids context window explosion; Notebook is the real memory |
| Three-block verification result | Cleanly separates raw data / reasoning / verdict |
| Prompt templates (Jinja2) | Scenario switching with zero code changes |
