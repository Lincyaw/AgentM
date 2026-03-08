# Design: Orchestrator

**Status**: DRAFT
**Last Updated**: 2026-03-08

---

### Code Conventions in This Document

This document contains two kinds of code:

- **Normative** (data structures) — `dataclass`, `Enum`, `TypedDict`, and interface contracts (tool signatures, WebSocket event schemas). Implementations **must** conform to these definitions: field names, types, and enum values are binding.
- **Illustrative** (logic) — Functions, algorithms, and flow examples showing one possible implementation approach. Implementations **may** use different code organization (e.g. class methods instead of free functions, different control flow, richer abstractions) as long as the **behavior and data contracts** are preserved.

**Rule of thumb**: If a code block defines a `class` with fields or an `@tool` signature, it's normative. If it defines a `def` with logic, it's illustrative.

---

## Overview

The Orchestrator is a **`create_react_agent`** instance acting as the **Team Leader / Hypothesis Reasoner**. It operates in a continuous ReAct loop (LLM → tool call → observe → repeat), using tools to dispatch Sub-Agents asynchronously, monitor their progress, and intervene when needed.

Sub-Agents are **independently compiled subgraphs**, launched as background `asyncio.Task`s by a **TaskManager**. The Orchestrator interacts with Sub-Agents exclusively through tools — never directly as graph nodes.

Phases (exploration, generation, verification, confirmation) are **not graph nodes** — they are markers in DiagnosticNotebook that the LLM updates to reflect its current focus. The LLM naturally interleaves phases like a human SRE: forming hypotheses while still exploring, looping back when evidence contradicts.

### Core Responsibilities

1. **Hypothesis reasoning** — Generate and verify hypotheses using LLM
2. **Async task dispatch** — Submit concurrent tasks to Sub-Agents via `dispatch_agent` tool (non-blocking)
3. **Monitoring & intervention** — Check task progress, inject instructions, abort stuck agents via tools
4. **Notebook management** — Track hypotheses, evidence, and exploration history
5. **Diagnostic discipline** — Prompt-guided (not graph-enforced) diagnostic methodology

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

3. **Mandatory counter-evidence** — The VerificationResult prompt guideline requires Sub-Agents to include observations that CONTRADICT the hypothesis, not just supporting evidence.

4. **Role separation** — The entity that generates hypotheses (Orchestrator) is different from the entity that collects evidence (Sub-Agent), reducing self-reinforcing loops.

---

## Message Management: Mode 2 (Minimal Messages + Notebook)

The Orchestrator uses **minimal messages** (2-3) combined with a structured **Notebook** as its primary working memory. This avoids context window explosion while maintaining complete diagnostic history.

### ExecutorState

```python
from typing import Annotated, Optional, TypedDict
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import add_messages
from dataclasses import dataclass, field


class Phase(str, Enum):
    EXPLORATION = "exploration"
    GENERATION = "generation"
    VERIFICATION = "verification"
    CONFIRMATION = "confirmation"


class HypothesisStatus(str, Enum):
    FORMED = "formed"               # Generated but not yet verified
    INVESTIGATING = "investigating" # Verification in progress (Sub-Agent dispatched)
    CONFIRMED = "confirmed"         # Verification: strong supporting evidence
    REJECTED = "rejected"           # Verification: strong contradicting evidence
    REFINED = "refined"             # Updated based on evidence (spawns child via parent_id)
    INCONCLUSIVE = "inconclusive"   # Investigated but evidence is ambiguous


class Verdict(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    PARTIAL = "partial"


class AgentRunStatus(str, Enum):
    """Runtime status of a Sub-Agent execution (used by TaskManager)."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status of a dispatched task in PendingTask."""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutorState(TypedDict):
    # Minimal message list (2-3 messages, managed via RemoveMessage)
    messages: Annotated[list[BaseMessage], add_messages]

    # Primary working memory (DiagnosticNotebook is a @dataclass, used as field value)
    notebook: DiagnosticNotebook

    # Auxiliary
    task_id: str
    current_phase: Phase

    # Compression tracking (for recall_history tool)
    compression_refs: list[CompressionRef]
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
| **Sub-Agent** | ReAct loop messages (tool calls + responses) accumulate across many iterations | Token count reaches 80% of model context limit |
| **Orchestrator Notebook** | `exploration_history` entries accumulate across all phases and hypothesis cycles | Token count reaches 80% of model context limit |
| **Orchestrator Prompt** | The formatted Notebook injected into the LLM prompt grows as data and hypotheses accumulate | Token count reaches 80% of model context limit |

### Compression Architecture

```
┌─────────────────────────────────────────────────┐
│ Orchestrator Layer                               │
│                                                   │
│   DiagnosticNotebook                             │
│   ├─ exploration_history: [...]  ← compress when │
│   │                                 tokens ≥ 80% │
│   │                                 context limit │
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
│   │                                 tokens ≥ 80% │
│   │                                 context limit │
│   └─ message_summary: str        ← compressed    │
│                                     output stored │
│                                     here          │
└─────────────────────────────────────────────────┘
```

### Layer 1: Sub-Agent Message Compression

Sub-Agents run a ReAct loop that can generate many tool call → tool response cycles. When the accumulated messages approach the model's context window limit, they are compressed into a structured summary before the next LLM call.

**Trigger**: Token count of current messages ≥ 80% of model context limit (`context_limit * compression_threshold`)

**Compression Method**: LLM-generated summary with prompt guidelines.

```python
@dataclass
class SubAgentMessageSummary:
    """Compressed summary of Sub-Agent message history.

    The summary field is natural language — the compression model decides
    what to emphasize based on the prompt guideline. The latest_raw_data
    field preserves the most recent tool output uncompressed for immediate use.
    """
    summary: str                     # Natural language summary of the agent's work so far
    latest_raw_data: dict            # Most recent tool output (uncompressed, for continuity)
```

Compression prompt guideline (not a rigid schema):
```
Summarize this agent's execution history. Include:
- What task was assigned and any instructions from the Orchestrator
- What tools were called and what key data was collected
- Any errors encountered and how they were handled
- Current progress: what's done and what remains
- Notable observations or anomalies
- Key decisions and reasoning
Keep it concise but preserve critical data points.
```

**Implementation via `pre_model_hook`**:

> **✅ LangGraph Verified**: `pre_model_hook` supports returning `llm_input_messages` which modifies only the LLM input without changing the actual `messages` state. This is the correct mechanism for compression — the full uncompressed history stays in checkpoints for trajectory export. The design below should use `llm_input_messages` instead of `messages` in the return value to preserve state integrity.

```python
def sub_agent_compression_hook(state: SubAgentState) -> dict:
    """pre_model_hook: compress messages before LLM call when token limit approached."""
    token_count = count_tokens(state["messages"], model=config.model)
    context_limit = get_model_context_limit(config.model)

    if token_count < context_limit * config.compression_threshold:  # default: 0.8
        return {"messages": state["messages"]}  # No compression needed

    # Compress all messages except the latest tool response
    messages_to_compress = state["messages"][:-1]
    latest_message = state["messages"][-1]

    summary = compress_messages_to_summary(messages_to_compress)  # LLM call

    # Replace LLM input with: summary + latest message (state unchanged)
    compressed_messages = [
        SystemMessage(content=format_summary(summary)),
        latest_message,
    ]

    # ⚠️ Implementation note: return `llm_input_messages` instead of `messages`
    # to preserve full history in state/checkpoints
    return {"llm_input_messages": compressed_messages}

# Attach to Sub-Agent (create_react_agent)
agent = create_react_agent(
    model=model,
    tools=tools,
    pre_model_hook=sub_agent_compression_hook,  # Runs before every LLM call
)
```

**Key design choice**: Compression happens in `pre_model_hook`, which runs before every LLM call inside the ReAct loop. This means:
- The full uncompressed history remains in state (for trajectory export) — **via `llm_input_messages` return key**
- Only the LLM's input is compressed
- The compression is transparent to the Orchestrator

> **⚠️ Implementation Note**: The `pre_model_hook` return must use `llm_input_messages` (not `messages`) to achieve this. Returning `messages` would **overwrite** the state's message history. With `llm_input_messages`, LangGraph sends the compressed messages to the LLM but keeps the original `messages` in the checkpoint.

### Layer 2: Orchestrator Notebook Compression

The Orchestrator's Notebook accumulates `exploration_history` entries and `collected_data` across all phases. When the formatted Notebook content approaches the model's context window limit, completed phase records are compressed into phase summaries.

**Trigger**: Token count of formatted Notebook ≥ 80% of model context limit

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
    if completed_phase == Phase.EXPLORATION:
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
    compression_threshold: 0.8      # Compress when token usage ≥ 80% of context limit

    # Sub-Agent compression
    sub_agent:
      compression_model: "gpt-4o-mini"  # Cheaper model for compression
      preserve_latest_n: 2              # Keep last N messages uncompressed

    # Orchestrator Notebook compression
    notebook:
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
│  history preserved   │     └─ state.compression_refs: [{from: S0.id, to: S14.id}]
│  in S0~S14           │
│                      └─ last full state before compression (recovery point)
```

**Properties**:
- `S0` through `S14`: Fine-grained checkpoints, each a complete state snapshot. Already stored, never modified (append-only).
- `S15`: The post-compression checkpoint. Its `messages` contain the summary instead of full history. Its `compression_refs` state field records the compressed range.
- `S14`: The natural recovery point — invoke `graph.invoke(None, S14.config)` to resume from the last pre-compression state with full detail.
- Trajectory export and debug replay use checkpoints `S0~S14` directly; they are not affected by compression.

### Compression Reference in Metadata

When compression occurs, the `CompressionRef` is appended to the `compression_refs` field in state:

```python
@dataclass
class CompressionRef:
    """Reference from a compressed state to its source checkpoint range."""
    from_checkpoint_id: str   # First checkpoint in the compressed range
    to_checkpoint_id: str     # Last checkpoint before compression
    layer: Literal["sub_agent", "orchestrator"]  # Which layer compressed
    step_count: int           # Number of checkpoints in the range
    reason: str               # "token_threshold"
```

Usage:
```python
# After compression, append to state's compression_refs
compression_ref = CompressionRef(
    from_checkpoint_id=first_checkpoint.id,
    to_checkpoint_id=last_checkpoint.id,
    layer="sub_agent",
    step_count=15,
    reason="token_threshold",
)

# Stored in state (ExecutorState.compression_refs or SubAgentState.compression_refs)
# Accessible via state["compression_refs"] — no metadata hacks needed
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

> **Decision**: `CompressionRef` is stored in state (`compression_refs: list[CompressionRef]` field in `ExecutorState` / `SubAgentState`). This is the simplest approach — no custom metadata hacks needed. The `recall_history` tool reads `state["compression_refs"]` directly.

```python
def recall_history_impl(
    query: str,
    scope: str,
    config: RunnableConfig,
    graph: CompiledGraph,
) -> str:
    # 1. Find compression references in state
    compression_refs = state.get("compression_refs", [])

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
  │                       ├─ 1. Find compression_refs in state
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
    current_phase: Phase = Phase.EXPLORATION
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
    status: HypothesisStatus = HypothesisStatus.FORMED
    # Lifecycle: FORMED → INVESTIGATING → CONFIRMED / REJECTED / REFINED / INCONCLUSIVE
    created_at: str
    last_updated: str
```

### ExplorationStep

Records each step across all phases. Structured fields for indexing and routing, natural language `content` for the actual diagnostic information.

> **Design principle**: `phase`, `action`, `verdict`, `target_hypothesis_id` are structured — the system uses them for state machine logic, filtering, and trajectory indexing. `content` is natural language — it carries the rich diagnostic narrative from Sub-Agents or the Orchestrator's reasoning.

```python
@dataclass
class ExplorationStep:
    # Structured index fields (for routing, filtering, trajectory)
    step_number: int
    phase: Phase
    action: str                                      # "Explore: infra, db, logs", "Verify H1", "Generate hypotheses"
    timestamp: str

    # Content (natural language — Sub-Agent reports, Orchestrator reasoning, etc.)
    content: str                                     # The diagnostic narrative for this step

    # Structured fields for specific phases (used by system logic)
    target_agents: Optional[list[str]] = None        # Phase 1: which agents were dispatched
    target_hypothesis_id: Optional[str] = None       # Phase 3: which hypothesis is being verified
    verdict: Optional[Verdict] = None                # Phase 3: verification outcome (for state machine)
    confirmed_root_cause: Optional[str] = None       # Phase 4: hypothesis ID of confirmed root cause

    # Agent execution metadata (structured, for monitoring/timeline)
    agent_outcomes: Optional[dict[str, AgentOutcome]] = None  # Per-agent completion info

@dataclass
class AgentOutcome:
    """Execution metadata for a Sub-Agent within an ExplorationStep."""
    agent_id: str
    task_id: str                          # TaskManager task_id (for trajectory linking)
    status: AgentRunStatus                # COMPLETED or FAILED
    duration_seconds: Optional[float] = None
    error: Optional[str] = None           # Error message if failed
```

---

## Sub-Agent Verification Result

When a Sub-Agent verifies a hypothesis, it returns a `VerificationResult` with a structured `verdict` for system logic and a natural language `report` for diagnostic content:

### VerificationResult

> **Design principle**: Structured fields for routing/judgment, natural language for content. The `verdict` field is structured so the Orchestrator can branch on it programmatically. The `report` field is natural language so the Sub-Agent can express rich diagnostic reasoning without being constrained to predefined field lists.

```python
@dataclass
class VerificationResult:
    verdict: Verdict                    # Structured — for Orchestrator branching logic
    report: str                         # Natural language — full diagnostic report
    refined_description: Optional[str] = None  # When verdict == PARTIAL: suggested refinement
```

The `report` field contains the Sub-Agent's complete diagnostic narrative. The prompt guides Sub-Agents to include investigation data, supporting/rejecting observations, and key findings — but the exact structure is up to the agent.

### Prompt Guideline for Verification Report

Sub-Agent prompts (Jinja2 templates) include a reporting guideline, not a rigid schema:

```
Write a diagnostic report for this verification task. Include:
- What you investigated and what data you collected
- Observations that SUPPORT the hypothesis
- Observations that CONTRADICT the hypothesis
- Any neutral or unexpected findings
- Your overall assessment and key conclusions
- Relevant raw data points (inline, not as a separate section)

End with a verdict: confirmed / rejected / partial
```

### Usage Example

```python
result = VerificationResult(
    verdict=Verdict.CONFIRMED,
    report="""Investigated database connection pool status for H1 (pool exhaustion).

Connection pool is at capacity: 100/100 active connections, 45 requests in wait queue.
Slow query analysis found 3 queries exceeding 2s, the worst being a JOIN on orders table (4.2s).
Lock wait time averages 320ms across active transactions.

Supporting: Pool completely saturated, significant wait queue, slow queries holding connections.
Contradicting: Primary CPU at 42% (not a CPU bottleneck), replica lag only 0.8s (acceptable).
Unexpected: Connection timeout errors spiked from 0 to 12 in the last 15 minutes — suggests recent onset.

Key conclusion: Connection pool exhaustion is the primary bottleneck. Slow queries are holding
connections too long, causing queue buildup. Recommend increasing pool_size to 150-200 and
adding a connection timeout of 30s.

Raw data: pool_size=100, active=100, waiting=45, slow_queries=[{sql: "SELECT * FROM orders JOIN ...",
time: 4.2s}, {sql: "...", time: 2.8s}, {sql: "...", time: 2.1s}], primary_cpu=42%, replica_lag=0.8s""",
)
```

---

## Orchestrator Behavior: ReAct Loop with Notebook

The Orchestrator is a single `create_react_agent` running a continuous ReAct loop. There are no separate phase nodes or explicit phase transitions — the LLM freely interweaves exploration, hypothesis generation, verification, and confirmation as it sees fit, guided by its system prompt.

### Phase as Notebook Marker, Not Graph Control

`current_phase` in DiagnosticNotebook is a **descriptive label**, not a routing signal. The Orchestrator LLM updates it to reflect what it's currently focused on. This serves:
- **Trajectory recording** — which phase an action belongs to
- **Frontend display** — phase indicator in topology/conversation view
- **Compression** — can compress older phase data when approaching token limits

The LLM is free to interleave phases naturally — e.g., it might start forming hypotheses while exploration agents are still running, or loop back to exploration after a verification fails.

### Notebook I/O: How the Orchestrator Reads and Writes Notebook

**Read**: `prompt` callable injects Notebook content before every LLM call.

```python
def build_orchestrator_prompt(system_prompt_template: str):
    """Build a prompt callable that injects Notebook into LLM context."""

    def prompt(state: ExecutorState):
        notebook_text = format_notebook(state["notebook"])
        system_prompt = render_template(system_prompt_template, notebook=notebook_text)
        return [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]

    return prompt

orchestrator = create_react_agent(
    model=model,
    tools=tools,
    prompt=build_orchestrator_prompt("prompts/orchestrator_system.j2"),
    state_schema=ExecutorState,
)
```

**Write**: Two modes, split by what's being updated:

| Notebook field | Update mode | Who controls | Why |
|---|---|---|---|
| `collected_data` | **Transparent** (tool internal) | check_tasks tool | Factual data — no judgment needed |
| `exploration_history` | **Transparent** (tool internal) | dispatch_agent / check_tasks | Action recording — mechanical |
| `hypotheses` | **LLM explicit** (update_hypothesis tool) | Orchestrator LLM | Reasoning judgment — LLM decides what hypotheses to form/update/reject |
| `current_phase` | **LLM explicit** (update_hypothesis tool) | Orchestrator LLM | LLM knows its own focus |

**Transparent updates** happen inside tools via `Command(update={"notebook": ...})` — LLM never sees this:

```python
# Inside check_tasks tool: auto-record completed results
notebook.collected_data[completed["agent_id"]] = completed["result"]
notebook.exploration_history.append(ExplorationStep(...))
```

**Explicit updates** happen via dedicated tools — LLM actively calls these as a reasoning step:

```python
# LLM explicitly manages hypotheses
update_hypothesis(id="H1", description="Connection pool exhaustion",
                  status="formed", evidence_summary="Pool 100/100, 45 waiting")
update_hypothesis(id="H1", status="confirmed",
                  evidence_summary="Slow queries holding connections confirmed")
remove_hypothesis(id="H3")  # Discard irrelevant hypothesis
```

### System Prompt Guidelines (Not Enforced by Graph)

The Orchestrator's system prompt provides diagnostic discipline without rigid enforcement:

```jinja2
{# prompts/orchestrator_system.j2 #}

You are an RCA specialist investigating a production incident.

Your working memory is the DiagnosticNotebook. It is shown to you at the beginning of each reasoning step.

<tools>
- **dispatch_agent(agent_id, task, task_type)**: Launch a background Sub-Agent.
  task_type: scout (initial recon), verify (test hypothesis), deep_analyze (focused deep dive).
  Results are NOT delivered automatically — call `check_tasks` to collect them.
- **check_tasks(wait_seconds=10)**: Check status of all dispatched tasks + collect completed results.
  If all tasks are still running, waits internally up to wait_seconds before returning.
- **update_hypothesis(id, description, status, evidence_summary?, parent_id?)**: Create
  or update a hypothesis. Status: formed, investigating, confirmed, rejected, refined, inconclusive.
- **remove_hypothesis(id)**: Remove a hypothesis from the board.
- **inject_instruction(task_id, msg)**: Redirect a running agent.
- **abort_task(task_id, reason)**: Stop a running agent.
- **knowledge_search/list/read**: Query historical knowledge base.
- **recall_history**: Retrieve compressed history details.
</tools>

<hypothesis_lifecycle>
- **formed**: Not yet investigated
- **investigating**: Sub-agent is gathering evidence
- **confirmed**: Strong evidence supports this hypothesis
- **rejected**: Strong evidence contradicts this hypothesis
- **refined**: Updated based on evidence (spawns child hypothesis via parent_id)
- **inconclusive**: Investigated but evidence is ambiguous
</hypothesis_lifecycle>

<task_types>
- **scout**: Initial reconnaissance — discover data sources, identify anomalies, map topology
- **verify**: Test a specific hypothesis — gather supporting or contradicting evidence
- **deep_analyze**: Focused deep dive into a specific data source, time range, or service
</task_types>

<diagnostic_approach>
1. **Start broad**: Round 1 MUST dispatch scout tasks to multiple agents. Don't deep-dive
   before you've scanned the landscape.
2. **Form hypotheses early**: As data comes in, use `update_hypothesis` to record them.
   Don't wait for all agents to finish before thinking.
3. **Verify with evidence**: For each hypothesis, dispatch verify tasks. Actively look
   for BOTH supporting AND contradicting evidence.
4. **Consider alternatives**: Before confirming, ensure you've investigated at least 2
   alternative explanations. Keep hypotheses bounded (max 10) — reject or merge weak ones.
5. **Conclude decisively**: When evidence strongly supports one hypothesis and alternatives
   are eliminated, set status to confirmed and stop.
</diagnostic_approach>

<critical_evaluation>
Sub-agent reports may be partial, biased, or incorrect. You MUST:

- **Cross-validate**: Confirm findings across multiple sources. One source alone cannot
  confirm a hypothesis.
- **Causation ≠ correlation**: Temporal co-occurrence does NOT prove causation. Require
  the actual propagation mechanism.
- **Symptoms ≠ causes**: "Service X has high error rate" is a symptom. Ask WHY one level deeper.
- **Evidence hierarchy**: Infrastructure failures precede application effects; upstream
  failures cause downstream ones; config changes and deployments are common root causes.

Treat sub-agent reports as data points, not verdicts — each agent sees only a narrow slice.
</critical_evaluation>

<context_briefing>
Sub-agents run in isolation — they ONLY see what you write in the `task` instruction.

Every dispatch_agent task MUST include:
- Key prior findings (service names, timestamps, anomalies)
- Specific signals to investigate (time range, components, anomaly types)
- Which hypothesis is being tested and current evidence status

BAD: "Check if the database has errors"
GOOD: "DB connection pool was 100/100 with 45 waiting at 14:30 UTC. Check slow queries
holding connections, lock wait times, and connection sources. Testing H2 (pool exhaustion)."
</context_briefing>

<per_round_task>
Each time you reason, follow this sequence:

1. **Analyze**: Review evidence from the latest sub-agent reports (from check_tasks results).
2. **Update hypotheses**: Use `update_hypothesis` to add new ones, change status, record evidence.
   Use `remove_hypothesis` to discard irrelevant ones.
3. **Decide**: Continue investigating or finalize.
   - **Continue**: Dispatch new tasks (scout/verify/deep_analyze), then check_tasks to collect.
   - **Finalize**: Set the confirmed hypothesis and provide root cause + recommendations.
</per_round_task>

<patience>
Agent investigations are long-running tasks (10s to minutes). This is normal.
- Do NOT abort a task unless you see clear failure signals (errors, stuck at same step).
- When check_tasks shows all agents still running, review existing data and refine thinking.
- Do NOT rush to conclusions. Thorough investigation beats fast guessing.
</patience>
```

### Example: Realistic RCA Flow

This shows how the Orchestrator naturally interleaves phases in a ReAct loop:

```
Orchestrator LLM (ReAct loop)
  │
  │ # Round 1: Scout
  ├─ dispatch_agent("infrastructure", "Scan CPU, memory, disk, network for anomalies",
  │    task_type="scout")
  ├─ dispatch_agent("database", "Check connection pool, slow queries, replication",
  │    task_type="scout")
  ├─ dispatch_agent("logs", "Collect error logs from last hour, identify spikes",
  │    task_type="scout")
  │
  │ # Wait for results
  ├─ check_tasks(wait_seconds=30)
  │  → infra completed: "CPU 85% (postgres 45%, java 28%). Memory 40%. Disk ok."
  │  → db: running (step 3/20)
  │  → logs: running (step 2/15)
  │
  │ # Form first hypothesis from partial data
  ├─ update_hypothesis(id="H1", description="High CPU causing slow responses",
  │    status="formed", evidence_summary="CPU 85%, postgres 45%")
  │
  │ # Collect remaining results
  ├─ check_tasks(wait_seconds=30)
  │  → db completed: "Pool 100/100, 45 waiting. 3 slow queries > 2s."
  │  → logs completed: "12 connection timeout errors in last 15 min."
  │
  │ # Revise hypothesis board
  ├─ update_hypothesis(id="H1", status="rejected",
  │    evidence_summary="CPU is symptom. DB pool is full — more likely cause.")
  ├─ update_hypothesis(id="H2", description="DB connection pool exhaustion",
  │    status="formed", evidence_summary="Pool 100/100, 45 waiting, timeout errors")
  │
  │ # Round 2: Verify H2 + check alternative
  ├─ dispatch_agent("database",
  │    "H2: DB connection pool exhaustion. Pool was 100/100 with 45 waiting. "
  │    "Investigate: connection sources, lock wait times, which queries hold "
  │    "connections longest. Check if slow queries are the cause.",
  │    task_type="verify", hypothesis_id="H2")
  ├─ dispatch_agent("infrastructure",
  │    "Check if CPU spike is from postgres or app. Any recent deployments? "
  │    "Testing whether CPU is root cause or symptom of DB issue.",
  │    task_type="verify", hypothesis_id="H1")
  │
  ├─ check_tasks(wait_seconds=30)
  │  → db completed: "Top connection holder: JOIN on orders (4.2s). Lock wait 320ms.
  │    All 100 connections from app-server-01. Verdict: SUPPORTED."
  │  → infra completed: "CPU from postgres (slow queries). No deployments.
  │    Verdict: CONTRADICTED — CPU is effect, not cause."
  │
  │ # Update hypotheses with verification evidence
  ├─ update_hypothesis(id="H2", status="confirmed",
  │    evidence_summary="Slow queries holding all 100 connections. Verified causal chain.")
  │
  └─ done: "Root cause: Connection pool exhaustion from slow queries (orders JOIN).
     Recommendations: increase pool_size, add 30s query timeout, optimize JOIN."
```

**Key observations**:
- LLM explicitly manages hypotheses via `update_hypothesis` (reasoning step)
- `task_type` guides Sub-Agent behavior (scout → broad, verify → focused)
- Context briefing in dispatch instructions includes prior findings and hypothesis
- Notebook data collection happens transparently via check_tasks
- The flow is natural, not rigidly phased — H1 formed during exploration, rejected immediately

---

## Execution Model: Async TaskManager

The Orchestrator operates like a real team leader — it **submits tasks asynchronously** and **monitors progress** while Sub-Agents execute concurrently. This is implemented via a **TaskManager** that runs Sub-Agent subgraphs as background `asyncio.Task`s.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Root StateGraph                                           │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Orchestrator (create_react_agent, ReAct loop)       │ │
│  │                                                       │ │
│  │  LLM → dispatch_agent("db", task) → {task_id: "t1"} │ │
│  │  LLM → check_tasks() → {running: [...],             │ │
│  │         completed: [{t2, result: {cpu: 0.85}}]}      │ │
│  │  LLM → inject_instruction("t1", "skip replica")     │ │
│  │  LLM → abort_task("t3", "no longer needed")         │ │
│  └──────────────────────────────────────────────────────┘ │
│           ↕ (tool calls)                                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  TaskManager (shared, injected into tools)            │ │
│  │                                                       │ │
│  │  _tasks: dict[str, ManagedTask]                       │ │
│  │                                                       │ │
│  │  submit() → asyncio.create_task(subgraph.ainvoke())  │ │
│  │  get_all_status() → status + completed results        │ │
│  │  inject() → queue instruction (pre_model_hook)        │ │
│  │  abort() → asyncio.Task.cancel()                      │ │
│  │                                                       │ │
│  │  Also: stream events → WebSocket for frontend         │ │
│  └──────────────────────────────────────────────────────┘ │
│           ↕ (asyncio.Tasks)                                │
│  ┌────────┬────────┬────────┐                              │
│  │ infra  │   db   │  logs  │  independently compiled      │
│  │ agent  │ agent  │ agent  │  subgraphs                   │
│  └────────┴────────┴────────┘                              │
└──────────────────────────────────────────────────────────┘
            ↓ stream events
     WebSocket → Frontend
```

### Why Not `Send()` API

The `Send()` API dispatches parallel workers but the Supervisor node **cannot execute during fan-out** — it returns `[Send(...)]` and is inactive until all workers complete. This prevents the Orchestrator from monitoring progress, injecting instructions, or aborting agents mid-execution.

The Async TaskManager model solves this: Sub-Agents run as independent `asyncio.Task`s, while the Orchestrator remains active in its ReAct loop, calling tools to check status and intervene.

### Concurrency Patterns

| Pattern | When | Mechanism |
|---------|------|-----------|
| **Parallel dispatch** | Phase 1 (explore all agents at once) | Multiple `dispatch_agent` tool calls in one LLM turn |
| **Parallel verification** | Multiple independent hypotheses | Multiple `dispatch_agent` calls for different hypotheses |
| **Monitor and intervene** | Agent stuck or off-track | `check_tasks` → `inject_instruction` or `abort_task` |
| **Sequential with review** | Verification + adversarial review | `dispatch_agent` → `check_tasks` → `dispatch_agent` (adversarial) |

### Sub-Agent Lifecycle

```
dispatch_agent("db", task)
  → TaskManager.submit("db", task)
  → asyncio.create_task(db_subgraph.ainvoke(input, config))
  → return {task_id: "t1", status: "running"}

# Sub-Agent executes asynchronously...
# Orchestrator continues its ReAct loop, calling other tools

check_tasks()
  → TaskManager.get_all_status()
  → return {
      running: [{task_id: "t1", step: 5, max_steps: 30, summary: "..."}],
      completed: [],
      failed: []
    }

# Later...
check_tasks()
  → return {
      running: [],
      completed: [{task_id: "t1", duration: 18.3,
                    result: {pool_size: 100, active: 100, waiting: 45}}],
      failed: []
    }
# Orchestrator LLM now has the full result directly
```

---

## Trajectory Registry

Since Sub-Agents run as independent subgraphs with their own `thread_id`s, the Orchestrator and Sub-Agent checkpoint chains are **separate**. The TaskManager maintains a **trajectory registry** that records the parent-child relationship, enabling hierarchical trajectory reconstruction at export time.

### Registry Data

```python
@dataclass
class TaskTraceRef:
    """Links a Sub-Agent's checkpoint chain to the Orchestrator's timeline."""
    task_id: str
    agent_id: str
    agent_thread_id: str           # Sub-Agent's own thread_id for get_state_history()
    parent_thread_id: str          # Orchestrator's thread_id
    parent_dispatch_step: int      # Orchestrator checkpoint step when dispatch_agent was called
    hypothesis_id: Optional[str] = None
```

The TaskManager populates this on `submit()` and updates it when the task completes.

### Trajectory Export

At export time, the hierarchical trajectory is reconstructed by combining:

1. **Orchestrator trace**: `get_state_history(orchestrator_thread_id)` — the main decision trace
2. **Sub-Agent traces**: `get_state_history(agent_thread_id)` per task — fine-grained tool call traces
3. **Registry**: Links each Sub-Agent trace to its dispatch point in the Orchestrator trace

```python
# Export format (JSONL)
{
    "orchestrator": {
        "thread_id": "rca-042",
        "steps": [...]                    # From get_state_history
    },
    "sub_agents": {
        "task-001": {
            "thread_id": "task-001-database",
            "agent_id": "database",
            "parent_dispatch_step": 1,
            "parent_result_step": 7,
            "hypothesis_id": "H1",
            "steps": [...]                # From get_state_history
        },
        "task-002": { ... }
    }
}
```

**RL Training Export**: For (state, action, reward, next_state) transitions, the Orchestrator trace provides the high-level decision trace (dispatch → check → reason). Each Sub-Agent trace provides the low-level execution trace (tool call → observe → reason → tool call). These can be exported separately or merged by aligning on dispatch/result steps.

**Replay**: Orchestrator and Sub-Agent traces can be replayed independently. To replay from a specific point:
- Orchestrator: `graph.invoke(None, {thread_id: "rca-042", checkpoint_id: "..."})`
- Sub-Agent: `subgraph.invoke(None, {thread_id: "task-001-database", checkpoint_id: "..."})`

---

## Message Passing: ToolMessage Protocol

Sub-Agent results are passed back to the Orchestrator as **ToolMessage content** through the `check_tasks` tool (results included inline for completed tasks). The Orchestrator never sees Sub-Agent internal state directly.

### Flow

```
1. Orchestrator calls dispatch_agent("database", "Verify H1: check pool")
   → ToolMessage: {task_id: "task-001", status: "running"}

2. Sub-Agent executes asynchronously (ReAct loop with its own tools)
   → Final AIMessage: '{"pool_size": 100, "active": 100, "waiting": 45}'

3. TaskManager extracts result:
   result_state = await subgraph.ainvoke(input, config)
   final_msg = result_state["messages"][-1]  # Last AIMessage
   managed.result = parse_agent_output(final_msg.content)

4. Orchestrator calls check_tasks()
   → completed: [{task_id: "task-001", result: {pool_size: 100, active: 100, waiting: 45}}]

5. Orchestrator LLM reasons about the result
```

### Result Extraction

```python
def _extract_final_result(self, managed: ManagedTask) -> dict:
    """Extract the Sub-Agent's final output from its completed state.

    The Sub-Agent's last AIMessage contains the result as JSON.
    For Phase 1 (exploration): raw data dict
    For Phase 3 (verification): VerificationResult structure
    """
    # Get the final state from the completed subgraph
    config = managed.subgraph_config
    state = self._agents[managed.agent_id].get_state(config)
    messages = state.values.get("messages", [])

    # Find the last AIMessage (the agent's final response)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            try:
                return json.loads(msg.content)
            except json.JSONDecodeError:
                return {"raw_response": msg.content}

    return {"error": "No final response from agent"}
```

### What the Orchestrator Sees

The Orchestrator only interacts with Sub-Agent data through ToolMessages:

| Tool | ToolMessage Content |
|------|-------------------|
| `dispatch_agent` | `{task_id, agent_id, status: "running"}` |
| `check_tasks` | `{running: [...], completed: [{..., result: ...}], failed: [...]}` |
| `inject_instruction` | `"Instruction injected into task-001"` |
| `abort_task` | `"Aborted task-001: reason"` |

The Orchestrator then writes results into DiagnosticNotebook (collected_data, exploration_history, etc.) based on its own reasoning — this is part of the Orchestrator's ReAct loop logic, not automatic state mapping.

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
    challenge_strength: Literal["weak", "moderate", "strong"]  # Domain-specific, not reused elsewhere
    # "strong" challenge forces verdict downgrade (confirmed → partial)
```

This is **also a Sub-Agent task** — dispatched asynchronously like any other. The Orchestrator waits for both the verification result and the adversarial review before making the final verdict.

### Feature Gate Access in Code

Feature gates are accessed from the Orchestrator's runtime config. Since the Orchestrator is a `create_react_agent`, feature gates influence the **system prompt** and **tool behavior**, not graph routing:

```python
# Feature gates are injected into the system prompt template
# prompts/orchestrator_system.j2
{% if feature_gates.adversarial_review %}
After reaching a verdict, always request a counter-argument before confirming.
{% endif %}

{% if feature_gates.min_verifications_before_confirm > 1 %}
Before confirming a root cause, ensure you have verified at least
{{ feature_gates.min_verifications_before_confirm }} alternative hypotheses.
{% endif %}

# Feature gates can also influence tool behavior at runtime
# (e.g., dispatch_agent could auto-dispatch an adversarial agent after verification)
```

Since the Orchestrator is a `create_react_agent`, all decision-making happens within its ReAct loop — there is no explicit routing logic or `Command(goto=...)`. The system prompt (Jinja2 template) configures the LLM's diagnostic approach, and feature gates are injected as prompt conditions. Switching scenarios requires **only new config + prompt files**, zero code changes.

Prompt templates are external files — switching scenarios requires **only new config + prompt files**, zero code changes.

---

## TaskManager & Orchestrator Tools

The TaskManager is the bridge between the Orchestrator's tools and the asynchronously executing Sub-Agent subgraphs. It manages the lifecycle of all dispatched tasks and provides the backing implementation for the Orchestrator's monitoring and intervention tools.

### ManagedTask

```python
@dataclass
class ManagedTask:
    """A single Sub-Agent execution managed by TaskManager."""
    task_id: str
    agent_id: str
    instruction: str
    hypothesis_id: Optional[str] = None

    status: AgentRunStatus = AgentRunStatus.RUNNING
    current_step: int = 0
    max_steps: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Result (available when completed)
    result: Optional[dict] = None

    # Error (available when failed)
    error_summary: Optional[str] = None
    last_steps: Optional[list[dict]] = None

    # Internal
    asyncio_task: Optional[asyncio.Task] = field(default=None, repr=False)
    events_buffer: list[dict] = field(default_factory=list)  # Recent events for summary
    subgraph_config: Optional[dict] = field(default=None, repr=False)  # For update_state
    pending_instructions: list[str] = field(default_factory=list)  # Queued by inject()

    # Trajectory registry (for hierarchical trace reconstruction)
    parent_thread_id: Optional[str] = None    # Orchestrator's thread_id
    parent_dispatch_step: Optional[int] = None  # Orchestrator step at dispatch time
```

### TaskManager

```python
class TaskManager:
    """Manages asynchronously executing Sub-Agent subgraphs.

    Provides the backing implementation for Orchestrator tools:
    - dispatch_agent → submit()
    - check_tasks   → get_all_status() (includes results for completed tasks)
    - inject_instruction → inject()
    - abort_task    → abort()

    Also forwards stream events to WebSocket for frontend display.
    """

    def __init__(self, agents: dict[str, CompiledGraph], config: ScenarioConfig,
                 summary_model: Optional[BaseChatModel] = None,
                 websockets: Optional[set[WebSocket]] = None):
        self._agents = agents                    # agent_id → compiled subgraph
        self._config = config                    # For max_steps, timeout, etc.
        self._tasks: dict[str, ManagedTask] = {}
        self._summary_model = summary_model      # Lightweight model for progress summaries
        self._websockets = websockets or set()   # Connected frontend clients
        self._task_counter = 0

    async def submit(self, agent_id: str, instruction: str,
                     hypothesis_id: Optional[str] = None) -> ManagedTask:
        """Launch a Sub-Agent subgraph as a background asyncio.Task."""
        self._task_counter += 1
        task_id = f"task-{self._task_counter:03d}"

        agent_config = self._config.agents[agent_id]
        subgraph = self._agents[agent_id]

        # Unique thread for each task execution
        subgraph_config = {
            "configurable": {"thread_id": f"{task_id}-{agent_id}"},
            "recursion_limit": agent_config.execution.max_steps * 2,  # ReAct loop ≈ 2x steps
        }
        # Checkpointer: Sub-Agents share the same checkpointer backend (from system.yaml)
        # but each task gets a unique thread_id, ensuring isolated checkpoint chains.
        # The checkpointer instance is passed when compiling each Sub-Agent subgraph:
        #   subgraph = builder.compile(checkpointer=shared_checkpointer)

        managed = ManagedTask(
            task_id=task_id,
            agent_id=agent_id,
            instruction=instruction,
            hypothesis_id=hypothesis_id,
            max_steps=agent_config.execution.max_steps,
            started_at=datetime.now().isoformat(),
            subgraph_config=subgraph_config,
        )

        # Launch as background task with streaming
        managed.asyncio_task = asyncio.create_task(
            self._execute_agent(managed, subgraph, subgraph_config)
        )
        self._tasks[task_id] = managed
        return managed

    async def _execute_agent(self, managed: ManagedTask,
                              subgraph: CompiledGraph, config: dict):
        """Execute a subgraph, streaming events to WebSocket and tracking progress."""
        input_data = {"messages": [HumanMessage(content=managed.instruction)]}

        try:
            async for namespace, mode, data in subgraph.astream(
                input_data, config,
                stream_mode=["updates", "custom"],
                subgraphs=True,
            ):
                # Track step progress
                if is_tool_call_event(data):
                    managed.current_step += 1

                # Buffer recent events (for summary generation)
                managed.events_buffer.append(data)
                if len(managed.events_buffer) > 20:
                    managed.events_buffer = managed.events_buffer[-20:]

                # Forward to WebSocket for frontend
                await self._forward_to_websocket(managed.agent_id, namespace, mode, data)

            # Completed successfully
            managed.status = AgentRunStatus.COMPLETED
            managed.result = self._extract_final_result(managed)
            managed.completed_at = datetime.now().isoformat()
            managed.duration_seconds = self._calc_duration(managed)

        except asyncio.CancelledError:
            managed.status = AgentRunStatus.FAILED
            managed.error_summary = "Aborted by Orchestrator"
            managed.completed_at = datetime.now().isoformat()
            managed.duration_seconds = self._calc_duration(managed)

        except Exception as e:
            managed.status = AgentRunStatus.FAILED
            managed.error_summary = str(e)
            managed.last_steps = managed.events_buffer[-5:]
            managed.completed_at = datetime.now().isoformat()
            managed.duration_seconds = self._calc_duration(managed)

    async def get_all_status(self, wait_seconds: int = 10) -> dict:
        """Return status snapshot of all tasks, including full results for completed tasks.

        If all tasks are still running, waits up to wait_seconds for any to complete
        before returning. This prevents LLM from polling in a tight loop.
        """
        # Wait for at least one task to complete (if all are running)
        if wait_seconds > 0 and all(
            t.status == AgentRunStatus.RUNNING for t in self._tasks.values()
        ):
            await self._wait_for_any_completion(timeout=wait_seconds)

        result = {"running": [], "completed": [], "failed": []}
        for task_id, task in self._tasks.items():
            entry = {
                "task_id": task_id,
                "agent_id": task.agent_id,
                "hypothesis_id": task.hypothesis_id,
            }
            if task.status == AgentRunStatus.RUNNING:
                entry["step"] = task.current_step
                entry["max_steps"] = task.max_steps
                entry["summary"] = self._summarize(task) if self._summary_model else None
                result["running"].append(entry)
            elif task.status == AgentRunStatus.COMPLETED:
                entry["duration_seconds"] = task.duration_seconds
                entry["result"] = task.result  # Full result inline
                result["completed"].append(entry)
            elif task.status == AgentRunStatus.FAILED:
                entry["error_summary"] = task.error_summary
                entry["last_steps"] = task.last_steps
                result["failed"].append(entry)
        return result

    def inject(self, task_id: str, instruction: str):
        """Queue an instruction for a running Sub-Agent.

        The instruction is stored in a per-task queue. The Sub-Agent's
        pre_model_hook checks this queue before every LLM call and injects
        pending instructions into the LLM input as HumanMessages.

        This is safe because pre_model_hook runs synchronously within the
        ReAct loop — no race conditions with the streaming subgraph.
        """
        task = self._tasks.get(task_id)
        if not task or task.status != AgentRunStatus.RUNNING:
            raise ToolException(f"Task {task_id} is not running")
        if not hasattr(task, 'pending_instructions'):
            task.pending_instructions = []
        task.pending_instructions.append(instruction)

    async def abort(self, task_id: str, reason: str):
        """Abort a running Sub-Agent by cancelling its asyncio.Task."""
        task = self._tasks.get(task_id)
        if not task or task.status != AgentRunStatus.RUNNING:
            raise ToolException(f"Task {task_id} is not running")
        task.asyncio_task.cancel()
        # CancelledError handler in _execute_agent sets status to FAILED

    async def _forward_to_websocket(self, agent_id: str, namespace: tuple,
                                      mode: str, data: dict):
        """Forward stream events to all connected WebSocket clients."""
        event = {
            "agent_path": [agent_id] + list(namespace),
            "mode": mode,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        disconnected = set()
        for ws in self._websockets:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.add(ws)
        self._websockets -= disconnected

    def _summarize(self, task: ManagedTask) -> str:
        """Use lightweight model to summarize agent's recent activity."""
        if not task.events_buffer:
            return f"Starting {task.agent_id}..."
        events_text = format_events_for_summary(task.events_buffer[-10:])
        summary = self._summary_model.invoke(
            f"Summarize this agent's current activity in 1-2 sentences:\n{events_text}"
        )
        return summary.content
```

### Instruction Injection: pre_model_hook

Sub-Agents consume Orchestrator instructions via a `pre_model_hook` that checks the TaskManager's per-task instruction queue before every LLM call. This is safe because the hook runs synchronously within the Sub-Agent's ReAct loop.

```python
def build_instruction_hook(task_manager: TaskManager, task_id: str):
    """Build a pre_model_hook that checks for pending Orchestrator instructions."""

    def hook(state):
        instructions = task_manager.consume_instructions(task_id)
        if instructions:
            injected = [
                HumanMessage(content=f"[Orchestrator] {msg}")
                for msg in instructions
            ]
            return {"llm_input_messages": state["messages"] + injected}
        return {"llm_input_messages": state["messages"]}

    return hook

# Attach when creating the Sub-Agent for a task
agent = create_react_agent(
    model=model,
    tools=tools,
    pre_model_hook=build_instruction_hook(task_manager, task_id),
)
```

### Orchestrator Tools

The Orchestrator has six tools for task management and hypothesis management:

```python
@tool
async def dispatch_agent(
    agent_id: str,
    task: str,
    task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
    hypothesis_id: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Dispatch a task to a Sub-Agent. The agent starts executing immediately
    in the background and this tool returns right away with a task_id.

    Use check_tasks() to monitor progress and collect results when completed.
    Multiple agents can be dispatched concurrently — call this tool multiple times.

    Args:
        agent_id: Which Sub-Agent to dispatch (e.g., "infrastructure", "database", "logs")
        task: Natural language instruction for the agent. MUST include relevant prior
            findings, specific signals to investigate, and which hypothesis is being tested.
        task_type: Investigation approach:
            - "scout": Initial recon — discover anomalies, map topology
            - "verify": Test a specific hypothesis with targeted evidence
            - "deep_analyze": Focused deep dive into specific data source/service
        hypothesis_id: Optional hypothesis being investigated (for tracing)
    """
    managed = await task_manager.submit(agent_id, task, task_type, hypothesis_id)

    # Transparently update Notebook: record the dispatch
    notebook = get_current_notebook()
    notebook.exploration_history.append(ExplorationStep(
        step_number=notebook.current_step,
        phase=notebook.current_phase,
        action=f"Dispatched {agent_id}: {task[:80]}",
        timestamp=datetime.now().isoformat(),
        content=f"Dispatched {agent_id} agent. Task: {task}",
        target_agents=[agent_id],
        agent_outcomes={agent_id: AgentOutcome(
            agent_id=agent_id, task_id=managed.task_id,
            status=AgentRunStatus.RUNNING,
        )},
    ))
    notebook.current_step += 1

    return Command(update={
        "notebook": notebook,
        "messages": [ToolMessage(
            content=json.dumps({"task_id": managed.task_id, "agent_id": agent_id, "status": "running"}),
            tool_call_id=tool_call_id,
        )],
    })


@tool
async def check_tasks(
    wait_seconds: int = 10,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Check the current status of all dispatched tasks.

    If all tasks are still running, waits internally up to wait_seconds
    before returning — so each call returns meaningful new information.

    Returns a dashboard with three sections:
    - running: Tasks still executing, with step progress and activity summary
    - completed: Tasks that have finished, with full results inline
    - failed: Tasks that encountered errors

    Args:
        wait_seconds: Max seconds to wait if all tasks are still running (default 10).
            Increase for long investigations, decrease for quick checks.
    """
    results = await task_manager.get_all_status(wait_seconds=wait_seconds)

    # Transparently update Notebook with completed/failed results
    notebook = get_current_notebook()
    for completed in results["completed"]:
        notebook.collected_data[completed["agent_id"]] = completed.get("result")
        notebook.exploration_history.append(ExplorationStep(
            step_number=notebook.current_step,
            phase=notebook.current_phase,
            action=f"Result from {completed['agent_id']}",
            timestamp=datetime.now().isoformat(),
            content=str(completed.get("result", "")),
            agent_outcomes={completed["agent_id"]: AgentOutcome(
                agent_id=completed["agent_id"],
                task_id=completed["task_id"],
                status=AgentRunStatus.COMPLETED,
                duration_seconds=completed.get("duration_seconds"),
            )},
        ))
        notebook.current_step += 1

    for failed in results["failed"]:
        notebook.exploration_history.append(ExplorationStep(
            step_number=notebook.current_step,
            phase=notebook.current_phase,
            action=f"Failed: {failed['agent_id']}",
            timestamp=datetime.now().isoformat(),
            content=f"Agent {failed['agent_id']} failed: {failed.get('error_summary', 'unknown')}",
            agent_outcomes={failed["agent_id"]: AgentOutcome(
                agent_id=failed["agent_id"],
                task_id=failed["task_id"],
                status=AgentRunStatus.FAILED,
                error=failed.get("error_summary"),
            )},
        ))
        notebook.current_step += 1

    return Command(update={
        "notebook": notebook,
        "messages": [ToolMessage(content=json.dumps(results), tool_call_id=tool_call_id)],
    })


@tool
def update_hypothesis(
    id: str,
    description: str,
    status: Literal["formed", "investigating", "confirmed", "rejected", "refined", "inconclusive"],
    evidence_summary: Optional[str] = None,
    parent_id: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Create or update a hypothesis in the DiagnosticNotebook.

    Call this every round to maintain your hypothesis board:
    - Create new hypotheses as they emerge from evidence
    - Update status as verification progresses
    - Record evidence summaries for traceability
    - Use parent_id when refining a hypothesis (creates child hypothesis)

    Args:
        id: Hypothesis ID (e.g., "H1", "H2"). Reuse ID to update existing.
        description: What the hypothesis claims (e.g., "DB connection pool exhaustion")
        status: Current status in the lifecycle
        evidence_summary: Key evidence supporting this status change
        parent_id: If this is a refinement, the parent hypothesis ID
    """
    notebook = get_current_notebook()
    now = datetime.now().isoformat()

    if id in notebook.hypotheses:
        h = notebook.hypotheses[id]
        h.description = description
        h.status = HypothesisStatus(status)
        if evidence_summary:
            h.evidence.append(evidence_summary)
        h.last_updated = now
    else:
        notebook.hypotheses[id] = Hypothesis(
            id=id, description=description,
            status=HypothesisStatus(status),
            evidence=[evidence_summary] if evidence_summary else [],
            counter_evidence=[],
            created_at=now, last_updated=now,
        )
    if parent_id:
        notebook.hypotheses[id].parent_id = parent_id

    return Command(update={
        "notebook": notebook,
        "messages": [ToolMessage(
            content=f"Hypothesis {id} updated: {status} — {description}",
            tool_call_id=tool_call_id,
        )],
    })


@tool
def remove_hypothesis(
    id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Remove a hypothesis from the board. Use when a hypothesis is irrelevant
    or has been merged into another.

    Args:
        id: Hypothesis ID to remove
    """
    notebook = get_current_notebook()
    if id in notebook.hypotheses:
        del notebook.hypotheses[id]
    return Command(update={
        "notebook": notebook,
        "messages": [ToolMessage(content=f"Hypothesis {id} removed", tool_call_id=tool_call_id)],
    })


@tool
def inject_instruction(task_id: str, instruction: str) -> str:
    """Inject a new instruction into a running Sub-Agent.

    The instruction is queued and will be injected into the agent's LLM input
    on its next reasoning step (via pre_model_hook). The agent sees it as a
    HumanMessage from [Orchestrator].

    Use when:
    - Agent seems stuck on a wrong path (from check_tasks summary)
    - New information from another agent should redirect this agent's focus
    - You want to narrow or expand the agent's investigation scope

    Args:
        task_id: The task to instruct
        instruction: The instruction text
    """
    task_manager.inject(task_id, instruction)
    return f"Instruction queued for {task_id}. Agent will see it on next reasoning step."


@tool
def abort_task(task_id: str, reason: str) -> str:
    """Abort a running task.

    Use when:
    - Agent is clearly stuck or in a loop (visible from check_tasks step count)
    - Another agent's results have made this task unnecessary
    - Timeout approaching

    Args:
        task_id: The task to abort
        reason: Why the task is being aborted (recorded in trajectory)
    """
    await task_manager.abort(task_id, reason)
    return f"Aborted {task_id}: {reason}"
```

### Example Flow

```
Orchestrator (Phase 3: Verification)
  │
  ├─ dispatch_agent("database", "Verify H1: connection pool exhaustion")
  │   → {task_id: "task-001", status: "running"}
  ├─ dispatch_agent("infrastructure", "Verify H1: check resource limits")
  │   → {task_id: "task-002", status: "running"}
  │
  ├─ ... Orchestrator continues reasoning (ReAct loop) ...
  │
  ├─ check_tasks()
  │   → running: [
  │       {task_id: "task-001", agent: "database",
  │        step: 3, max_steps: 30, summary: "Querying slow query log"},
  │       {task_id: "task-002", agent: "infrastructure",
  │        step: 5, max_steps: 20, summary: "Checking memory, disk I/O collected"},
  │     ]
  │   → completed: []
  │   Decision: Both progressing normally, continue waiting.
  │
  ├─ check_tasks()
  │   → running: [
  │       {task_id: "task-001", agent: "database",
  │        step: 8, max_steps: 30, summary: "Retrying connection to replica, 3 failed attempts"},
  │     ]
  │   → completed: [
  │       {task_id: "task-002", agent: "infrastructure", duration: 12.3,
  │        result: {cpu: 0.45, memory: 0.6, disk_io: {...}, network: {...}}},
  │     ]
  │   Decision: Infrastructure done. Database struggling with replica.
  │
  ├─ inject_instruction("task-001", "Skip replica, focus on primary pool metrics")
  │   → "Instruction injected into task-001"
  │
  ├─ check_tasks()
  │   → completed: [
  │       {task_id: "task-001", agent: "database", duration: 18.0,
  │        result: {pool_size: 100, active: 100, waiting: 45}},
  │     ]
  │
  └─ All results collected. Orchestrator reasons about verdict...
```

---

## Frontend Conversation Reconstruction

Since messages are minimal, the frontend reconstructs conversation from Notebook:

```typescript
function rebuildConversation(messages: Message[], notebook: DiagnosticNotebook): Message[] {
  const conversation = [...messages];

  for (const step of notebook.exploration_history) {
    // All phases: render phase tag + natural language content
    // Structured fields (target_agents, verdict) add context; content carries the narrative
    if (step.phase === "exploration") {
      const agents = step.target_agents?.join(", ") || "unknown";
      conversation.push({
        role: "assistant",
        content: `[Phase 1: Exploration] Dispatched to: ${agents}\n\n${step.content}`,
      });
    } else if (step.phase === "generation") {
      conversation.push({
        role: "assistant",
        content: `[Phase 2: Hypothesis Generation]\n\n${step.content}`,
      });
    } else if (step.phase === "verification") {
      conversation.push({
        role: "assistant",
        content: `[Phase 3] Verify ${step.target_hypothesis_id}\n` +
          `${step.content}\n` +              // Natural language report
          `Verdict: ${step.verdict}`,
      });
    } else if (step.phase === "confirmation") {
      conversation.push({
        role: "assistant",
        content: `[Phase 4] ${step.content}`,  // Natural language conclusion
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
    system: "prompts/orchestrator_system.j2"
    hypothesis_generation: "prompts/hypothesis_generation.j2"
    verification_task: "prompts/verify_hypothesis.j2"
    adversarial_review: "prompts/adversarial_review.j2"

  tools:
    - dispatch_agent          # Async dispatch to Sub-Agent (non-blocking, with task_type)
    - check_tasks             # Monitor all task status + collect results (with wait)
    - update_hypothesis       # LLM explicitly manages hypothesis board
    - remove_hypothesis       # LLM removes irrelevant hypotheses
    - inject_instruction      # Inject instruction into running task
    - abort_task              # Abort running task

  task_manager:
    summary_model: "gpt-4o-mini"         # Lightweight model for progress summaries
    max_events_for_summary: 10           # Recent events to include in summary
    stream_to_frontend: true             # Forward events to WebSocket

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
| VerificationResult (verdict + report) | Cleanly separates structured verdict from natural language report; two-field structure replaces former three-block design |
| Two-layer context compression | Sub-Agent (pre_model_hook) + Orchestrator (phase summaries); prompt optimization only, checkpoints retain full data |
| Compression ref in state | `compression_refs` field in state; `from_id` / `to_id` enables O(1) drill-down to fine-grained history |
| recall_history tool | Post-compression Agents can query their own pre-compression messages; retrieves raw data, tool params, and reasoning from checkpoint chain |
| Unified retrieval interface | Intra-task recall and cross-task knowledge share the same conceptual interface (natural language query → relevant data) |
| Prompt templates (Jinja2) | Scenario switching with zero code changes |
