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

## Context Compression (Intra-Task Memory)

Even with Mode 2's minimal messages, the context window can still be exhausted during long-running RCA tasks. The **exploration_history** in DiagnosticNotebook grows unboundedly, and Sub-Agents accumulate tool call records across many ReAct iterations. Context compression handles this at two layers.

### Problem: What Grows

| Layer | What Grows | Trigger |
|-------|-----------|---------|
| **Sub-Agent** | ReAct loop messages (tool calls + responses) accumulate across many iterations | Tool call count exceeds threshold |
| **Orchestrator Notebook** | `exploration_history` entries accumulate across all phases and hypothesis cycles | History entry count exceeds threshold |
| **Orchestrator Prompt** | The formatted Notebook injected into the LLM prompt grows as data and hypotheses accumulate | Prompt token count approaches limit |

### Compression Architecture

```
┌─────────────────────────────────────────────────┐
│ Orchestrator Layer                               │
│                                                   │
│   DiagnosticNotebook                             │
│   ├─ exploration_history: [...]  ← compress when │
│   │                                 len > threshold│
│   ├─ collected_data: {...}       ← summarize      │
│   │                                 completed phases│
│   └─ phase_summaries: [...]      ← compressed     │
│                                     output stored  │
│                                     here           │
└─────────────────────────────────────────────────┘
         ↕
┌─────────────────────────────────────────────────┐
│ Sub-Agent Layer                                   │
│                                                   │
│   SubAgentState                                  │
│   ├─ messages: [...]             ← compress when │
│   │                                 tool_call_count│
│   │                                 > threshold    │
│   └─ message_summary: str        ← compressed    │
│                                     output stored │
│                                     here          │
└─────────────────────────────────────────────────┘
```

### Layer 1: Sub-Agent Message Compression

Sub-Agents run a ReAct loop that can generate many tool call → tool response cycles. When the cycle count exceeds a configurable threshold, the accumulated messages are compressed into a structured summary before the next LLM call.

**Trigger**: `tool_call_count > max_uncompressed_steps` (configurable per agent, default: 10)

**Compression Method**: 8-section structured summary via LLM call.

```python
@dataclass
class SubAgentMessageSummary:
    """Structured summary of compressed Sub-Agent message history."""

    # 1. Task & Instructions
    original_task: str               # What was the agent asked to do
    orchestrator_instructions: str   # Any injected instructions from Orchestrator

    # 2. Tools Used
    tools_called: list[dict]         # [{name, call_count, last_args}]

    # 3. Key Data Collected
    key_findings: list[str]          # Most important data points found

    # 4. Errors Encountered
    errors: list[str]                # Tool errors, retries, failures

    # 5. Current Progress
    completed_steps: list[str]       # What has been done
    pending_steps: list[str]         # What remains to do

    # 6. Running Observations
    observations: list[str]          # Notable patterns or anomalies

    # 7. Decisions Made
    decisions: list[str]             # Why certain tools/paths were chosen

    # 8. Raw Data Snapshot
    latest_raw_data: dict            # Most recent tool output (uncompressed)
```

**Implementation via `pre_model_hook`**:

```python
def sub_agent_compression_hook(state: SubAgentState) -> dict:
    """pre_model_hook: compress messages before LLM call when threshold exceeded."""
    if state["tool_call_count"] <= config.max_uncompressed_steps:
        return {"messages": state["messages"]}  # No compression needed

    # Compress all messages except the latest tool response
    messages_to_compress = state["messages"][:-1]
    latest_message = state["messages"][-1]

    summary = compress_messages_to_summary(messages_to_compress)  # LLM call

    # Replace message history with: summary + latest message
    compressed_messages = [
        SystemMessage(content=format_summary(summary)),
        latest_message,
    ]

    return {"messages": compressed_messages}

# Attach to Sub-Agent
agent = create_react_agent(
    model=model,
    tools=tools,
    pre_model_hook=sub_agent_compression_hook,  # Runs before every LLM call
)
```

**Key design choice**: Compression happens in `pre_model_hook`, which runs before every LLM call inside the ReAct loop. This means:
- The full uncompressed history remains in state (for trajectory export)
- Only the LLM's input is compressed
- The compression is transparent to the Orchestrator

### Layer 2: Orchestrator Notebook Compression

The Orchestrator's Notebook accumulates `exploration_history` entries and `collected_data` across all phases. When a phase completes, its detailed records can be compressed into a phase summary.

**Trigger**: Phase transition (exploration → generation → verification → confirmation)

**What gets compressed**: Completed phase's entries in `exploration_history` and corresponding `collected_data`.

**What is preserved**: Active phase data, all hypothesis objects, and the current working context.

```python
@dataclass
class PhaseSummary:
    """Compressed summary of a completed diagnostic phase."""
    phase: str                       # "exploration", "generation", etc.
    started_at: str
    completed_at: str

    # Compressed content
    key_data_collected: dict         # Aggregated key findings (not raw data)
    actions_taken: list[str]         # What agents did, compressed
    decisions_made: list[str]        # Why certain paths were chosen
    hypotheses_affected: list[str]   # Which hypotheses were created/updated
    anomalies_noted: list[str]       # Unexpected findings worth preserving

    # Preserved raw data (selected)
    critical_evidence: list[dict]    # Evidence directly supporting/rejecting hypotheses
    # (Raw data is NOT included — it's already in the checkpoint for replay)

# DiagnosticNotebook gains phase summaries
@dataclass
class DiagnosticNotebook:
    # ... existing fields ...

    # Compression output
    phase_summaries: list[PhaseSummary] = field(default_factory=list)
```

**Compression flow**:

```python
def compress_completed_phase(notebook: DiagnosticNotebook, completed_phase: str) -> DiagnosticNotebook:
    """Compress a completed phase's detailed records into a PhaseSummary."""

    # Extract entries for the completed phase
    phase_entries = [
        entry for entry in notebook.exploration_history
        if entry.phase == completed_phase
    ]

    # Generate summary via LLM
    summary = llm_compress_phase(phase_entries, notebook.collected_data)

    # Store summary
    notebook.phase_summaries.append(summary)

    # Remove compressed entries from exploration_history
    notebook.exploration_history = [
        entry for entry in notebook.exploration_history
        if entry.phase != completed_phase
    ]

    # Slim down collected_data for completed exploration phase
    if completed_phase == "exploration":
        notebook.collected_data = extract_key_metrics(notebook.collected_data)

    return notebook
```

### Orchestrator Prompt Assembly

When the Orchestrator calls the LLM, the prompt is assembled from Notebook data with compression-aware formatting:

```python
def format_notebook_for_llm(notebook: DiagnosticNotebook) -> str:
    """Format Notebook into LLM prompt, using summaries for compressed phases."""
    sections = []

    # 1. Compressed phases → use PhaseSummary
    for summary in notebook.phase_summaries:
        sections.append(f"## {summary.phase.title()} (completed)\n"
                       f"Key findings: {summary.key_data_collected}\n"
                       f"Decisions: {summary.decisions_made}\n"
                       f"Anomalies: {summary.anomalies_noted}")

    # 2. Active phase → use full detail
    active_entries = [e for e in notebook.exploration_history
                      if e.phase == notebook.current_phase]
    for entry in active_entries:
        sections.append(format_entry_full(entry))

    # 3. Hypotheses → always full detail (compact by nature)
    for h_id, h in notebook.hypotheses.items():
        sections.append(format_hypothesis(h))

    return "\n\n".join(sections)
```

**Principle**: Completed phases get summaries, active phase gets full detail, hypotheses are always full. This ensures the LLM has maximum detail for its current decision while staying within context limits.

### Configuration

```yaml
# orchestrator.yaml
orchestrator:
  compression:
    enabled: true

    # Sub-Agent compression
    sub_agent:
      max_uncompressed_steps: 10        # Compress after N tool calls
      compression_model: "gpt-4o-mini"  # Cheaper model for compression
      preserve_latest_n: 2              # Keep last N messages uncompressed

    # Orchestrator Notebook compression
    notebook:
      compress_on_phase_transition: true  # Auto-compress when phase changes
      max_history_entries: 20             # Force compress if history exceeds this
      compression_model: "gpt-4o-mini"
```

### Compression and Checkpoint Chain

**Key insight**: Each LangGraph checkpoint is a **full state snapshot** (not a delta). The checkpoint chain within the same `(thread_id, checkpoint_ns)` is a **linear linked list** — given any two checkpoints, there is exactly one path between them.

This means compression requires **no special checkpoint management**:

```
Checkpoint chain (append-only, never modified):

S0 → S1 → S2 → ... → S14 → S15(compression happens here)→ S16 → ...
│                      │     │
│  fine-grained        │     ├─ messages: [summary, latest]
│  history preserved   │     └─ metadata.compression_ref: {from: S0.id, to: S14.id}
│  in S0~S14           │
│                      └─ last full state before compression (recovery point)
```

**Properties**:
- `S0` through `S14`: Fine-grained checkpoints, each a complete state snapshot. Already stored, never modified (append-only).
- `S15`: The post-compression checkpoint. Its `messages` contain the summary instead of full history. Its metadata records `compression_ref` pointing to the compressed range.
- `S14`: The natural recovery point — invoke `graph.invoke(None, S14.config)` to resume from the last pre-compression state with full detail.
- Trajectory export and debug replay use checkpoints `S0~S14` directly; they are not affected by compression.

### Compression Reference in Metadata

When compression occurs, the new checkpoint's metadata records the compressed range for future analysis and drill-down:

```python
@dataclass
class CompressionRef:
    """Reference from a compressed checkpoint to its source range."""
    from_checkpoint_id: str   # First checkpoint in the compressed range
    to_checkpoint_id: str     # Last checkpoint before compression
    layer: Literal["sub_agent", "orchestrator"]  # Which layer compressed
    step_count: int           # Number of checkpoints in the range
    reason: str               # "tool_call_threshold" | "phase_transition" | "max_history"
```

Usage:
```python
# After compression, record reference in checkpoint metadata
compression_ref = CompressionRef(
    from_checkpoint_id=first_checkpoint.id,
    to_checkpoint_id=last_checkpoint.id,
    layer="sub_agent",
    step_count=15,
    reason="tool_call_threshold",
)

# Store in metadata (via custom metadata field)
# LangGraph's CheckpointMetadata supports arbitrary fields (TypedDict with total=False)
```

### Drill-Down from Compressed Checkpoint

When debugging or analyzing, use `compression_ref` to locate and traverse the fine-grained history:

```python
def drill_down_compressed_range(graph, config, compression_ref: CompressionRef):
    """Traverse the fine-grained checkpoint range behind a compression."""
    history = list(graph.get_state_history(config))
    history.reverse()  # Chronological order

    # Filter to the compressed range
    in_range = False
    fine_grained_steps = []
    for snapshot in history:
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        if checkpoint_id == compression_ref.from_checkpoint_id:
            in_range = True
        if in_range:
            fine_grained_steps.append(snapshot)
        if checkpoint_id == compression_ref.to_checkpoint_id:
            break

    return fine_grained_steps  # Full detail for S0~S14
```

### Summary of Compression vs. Checkpoint Interaction

| Concern | How It Works |
|---------|-------------|
| **Data preservation** | Checkpoint chain is append-only; compression never deletes existing checkpoints |
| **State recovery** | `graph.invoke(None, S14.config)` resumes from the last pre-compression full state |
| **Trajectory export** | Uses checkpoint chain (S0~S14), unaffected by compression |
| **Debug drill-down** | `compression_ref.from_id / to_id` locates the fine-grained range in O(1) |
| **Analysis** | `compression_ref.step_count` and `reason` enable compression pattern analysis |

---

## History Recall (Post-Compression Retrieval)

After compression, the Agent works with a summary instead of full history. But during hypothesis verification, previously unimportant details may become critical. History Recall gives compressed Agents the ability to **query their own pre-compression checkpoint history** on demand, like flipping back to original documents after reading a summary.

### What Agents Need to Recall

| Information Need | Example | Source in Checkpoint |
|-----------------|---------|---------------------|
| Raw metric data and breakdown | "What processes were using that 45% CPU?" | `ToolMessage.content` (tool return values) |
| Time series and trends | "Was connection count gradual or sudden?" | `ToolMessage.content` (time-series fields) |
| Tool call parameters | "What time window was used for slow query analysis?" | `AIMessage.tool_calls` (arguments) |
| Agent's intermediate reasoning | "Why did the agent choose to check disk before network?" | `AIMessage.content` (reasoning text) |
| Data not highlighted in summary | "Were inode stats collected during infra scan?" | `ToolMessage.content` (all fields, not just key findings) |

The retrieval target is the **complete messages sequence** within the compressed checkpoint range — because messages contain tool calls (parameters), tool responses (data), and the Agent's intermediate reasoning (context) together.

### recall_history Tool

A tool available to both Orchestrator and Sub-Agents after compression has occurred.

```python
@tool
def recall_history(
    query: str,
    scope: Literal["current_compression", "all_compressions"] = "current_compression",
) -> str:
    """Search pre-compression history for detailed information.

    Use this when you need details that were lost during context compression,
    such as: raw metric breakdowns, tool call parameters, time-series data,
    or observations not included in the summary.

    Args:
        query: Natural language description of what you're looking for.
            Examples:
            - "What were the top 5 slow queries and their execution times?"
            - "What parameters were used when checking the connection pool?"
            - "Was there any disk inode data collected during infrastructure scan?"
        scope: Search range.
            - "current_compression": Only the most recent compressed range
            - "all_compressions": All compressed ranges in this task
    """
```

### Implementation

```python
def recall_history_impl(
    query: str,
    scope: str,
    config: RunnableConfig,
    graph: CompiledGraph,
) -> str:
    # 1. Find compression references in checkpoint metadata
    compression_refs = find_compression_refs(graph, config, scope)

    if not compression_refs:
        raise ToolException(
            "No compression has occurred yet. Full history is already in your context. "
            "This tool is only available after context compression."
        )
        # ToolException is returned to the LLM as a ToolMessage with error,
        # allowing the agent to self-correct (e.g., stop trying to recall)

    # 2. Extract messages from the compressed checkpoint ranges
    all_messages = []
    for ref in compression_refs:
        fine_grained_steps = drill_down_compressed_range(graph, config, ref)
        for snapshot in fine_grained_steps:
            messages = snapshot.values.get("messages", [])
            all_messages.extend([
                {
                    "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
                    "step": snapshot.metadata.get("step"),
                    "messages": [
                        {
                            "type": type(m).__name__,       # AIMessage, ToolMessage, etc.
                            "content": m.content,
                            "tool_calls": getattr(m, "tool_calls", None),
                            "tool_call_id": getattr(m, "tool_call_id", None),
                        }
                        for m in messages
                    ],
                }
            ])

    # 3. Use LLM to find relevant information from the extracted messages
    retrieval_prompt = f"""You are searching through an agent's historical execution records.

Query: {query}

The following are chronological message records from the agent's pre-compression execution.
Each record contains the checkpoint step, message type (AIMessage = agent reasoning/tool calls,
ToolMessage = tool return values, HumanMessage = instructions from orchestrator).

Records:
{format_messages_for_retrieval(all_messages)}

Instructions:
- Find and return information relevant to the query
- Include specific data values, not just summaries
- If the query asks about tool parameters, extract them from AIMessage.tool_calls
- If the query asks about raw data, extract from ToolMessage.content
- If the information is not found, say so explicitly
"""
    result = compression_model.invoke(retrieval_prompt)
    return result.content
```

### Recall Flow

```
Agent (post-compression)
  │
  ├─ Context: [summary] + [current messages]
  │
  ├─ Needs detail → calls recall_history("What was CPU breakdown by process?")
  │                       │
  │                       ├─ 1. Find compression_ref in checkpoint metadata
  │                       ├─ 2. Load checkpoints S0~S14 (fine-grained history)
  │                       ├─ 3. Extract all messages from those checkpoints
  │                       ├─ 4. LLM retrieval: find relevant data for query
  │                       └─ 5. Return: "CPU by process: postgres 25%, java 12%, ..."
  │
  └─ Continues reasoning with the recalled detail
```

### Scope: Current Task vs. Cross-Task

`recall_history` is scoped to the **current task** (same `thread_id`). Cross-task knowledge retrieval uses the **Knowledge Store** — a filesystem-like hierarchical namespace backed by LangGraph Store with semantic search.

| Aspect | recall_history (intra-task) | Knowledge Store (cross-task) |
|--------|---------------------------|------------------------------|
| **Scope** | Current thread's compressed checkpoints | All historical task trajectories |
| **Data source** | Checkpoint chain (raw messages) | LangGraph Store (persistent, cross-thread) |
| **Query interface** | Natural language | Semantic search + path browsing + direct read |
| **Latency** | Low (local checkpoint read) | Higher (embedding search + store retrieval) |
| **Granularity** | Individual tool calls and responses | Failure patterns, skills, system knowledge |
| **Tools** | `recall_history` | `knowledge_search`, `knowledge_list`, `knowledge_read` |

> Detail: See [generic-state-wrapper.md](generic-state-wrapper.md#knowledge-store-filesystem-like-namespace) for the Knowledge Store design, namespace hierarchy, and tool APIs.

### Configuration

```yaml
# orchestrator.yaml
orchestrator:
  compression:
    # ... existing compression config ...

    recall:
      enabled: true
      model: "gpt-4o-mini"          # Model for retrieval LLM call
      max_messages_per_query: 200    # Cap messages loaded per recall
      tool_available_to:             # Which agents get the recall tool
        - orchestrator
        - sub_agents                 # All sub-agents, or list specific ones
```

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
| Two-layer context compression | Sub-Agent (pre_model_hook) + Orchestrator (phase summaries); prompt optimization only, checkpoints retain full data |
| Compression ref in checkpoint metadata | `from_id` / `to_id` enables O(1) drill-down to fine-grained history; supports analysis and debug |
| recall_history tool | Post-compression Agents can query their own pre-compression messages; retrieves raw data, tool params, and reasoning from checkpoint chain |
| Unified retrieval interface | Intra-task recall and cross-task knowledge share the same conceptual interface (natural language query → relevant data) |
| Prompt templates (Jinja2) | Scenario switching with zero code changes |
