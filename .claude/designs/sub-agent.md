# Design: Sub-Agent

**Status**: DRAFT
**Last Updated**: 2026-03-07

---

> **Code conventions**: Same as [orchestrator.md](orchestrator.md#code-conventions-in-this-document). Data structures (dataclass, TypedDict) are **normative**; function logic is **illustrative**.

## Overview

Sub-Agents are **independently compiled subgraphs**, created via `create_react_agent`. They are launched asynchronously by the **TaskManager** as `asyncio.Task`s — they are NOT nodes in the Root StateGraph. Sub-Agents receive tasks from the Orchestrator (via `dispatch_agent` tool), execute tool calls, and return results (via `check_tasks` tool, which includes results inline for completed tasks).

**Key constraint**: In hypothesis-driven RCA, Sub-Agents act as **data collectors** — they return raw data and structured verification results, not independent reasoning.

---

## State Schema

Each Sub-Agent has its own State Schema. Since Sub-Agents are independently compiled subgraphs (not nodes in the Root graph), there is **no automatic state mapping** between parent and child. Communication happens exclusively through the Orchestrator's tools (`dispatch_agent` sends instructions, `check_tasks` retrieves results inline for completed tasks).

```python
# Sub-Agent state (fully independent, not mapped to Root graph)
class SubAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Agent's own message history (dedup by ID, supports RemoveMessage)
    scratchpad: list[str]                     # Private working notes
    observations: list[str]                   # Private observations
    tool_call_count: int = 0                  # Tracked for progress reporting
    compression_refs: list[CompressionRef] = []  # Compression tracking (see orchestrator.md)
```

> **⚠️ LangGraph Verified**: `add_messages` (from `langgraph.graph.message`) is the standard reducer for message lists. It handles deduplication by message ID, supports `RemoveMessage` for deletion, and is consistent with `create_react_agent`'s internal state. Do NOT use `operator.add` — it would cause duplicate messages.

Since Sub-Agents are invoked by TaskManager (not as graph nodes), their state is fully private. The Orchestrator receives results through `check_tasks` tool (completed results inline) — it never sees the Sub-Agent's internal state directly.

---

## Implementation via create_react_agent

> **⚠️ API Note**: At design time, `create_react_agent` is in `langgraph.prebuilt`. Verify the current import path at implementation time, as the API may have been reorganized.

> **Implementation Note — state_schema**: `state_schema` is intentionally **NOT** passed to `create_react_agent`. Same reason as the Orchestrator: the framework's default `AgentState` includes `remaining_steps` which is required by `create_react_agent`. See [orchestrator.md](orchestrator.md#message-management-mode-2-minimal-messages--notebook) for the full rationale.

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

def create_sub_agent(agent_id: str, config: AgentConfig, tool_registry: ToolRegistry):
    model = ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
    )

    tools = [tool_registry.get_tool(name).create_with_config(**config.tool_settings.get(name, {}))
             for name in config.tools]

    system_prompt = load_prompt_template(config.prompt_template, agent_id=agent_id)

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        name=agent_id,
        state_schema=SubAgentState,
        interrupt_before=config.interrupt_before or [],
        pre_model_hook=build_compression_hook(config),  # Context compression
    )
```

> **Context compression**: The `pre_model_hook` compresses message history when token count reaches 80% of the model's context limit. Full history remains in state for trajectory export. See [orchestrator.md](orchestrator.md#context-compression-intra-task-memory) for the compression data structures and configuration.

### ReAct Execution Loop

`create_react_agent` automatically implements the ReAct cycle:

```
Input (messages)
    ↓
[Agent Node: LLM Reasoning]
    ├─ Output: ToolCall[] → [Tools Node] → Loop back
    └─ Output: Final answer → Return to parent
```

Each iteration continues until:
- Agent returns final answer
- `recursion_limit` reached (see note below)
- Interrupt point triggered

> **⚠️ LangGraph Verification**: `create_react_agent` does NOT have a native `max_steps` parameter. Step limiting is done via `recursion_limit` in the invocation config (e.g., `graph.invoke(input, {"recursion_limit": 40})`). The default is 10000. The YAML config's `execution.max_steps` must be translated to `recursion_limit` by our framework at invocation time. Note: `remaining_steps` is tracked in the agent state and when < 2, the agent returns a final message instead of raising `GraphRecursionError`.

> **Implementation Note — Step Limiting**: `_build_budget_hook(max_steps)` injects urgency messages into the conversation when the step budget is running low. This is the current step-limiting mechanism — it nudges the agent to wrap up rather than hard-terminating via `recursion_limit`. The budget hook runs as a `pre_model_hook` and prepends a system message when remaining steps fall below a threshold.

---

## Configuration

Sub-Agent configuration is defined **inline in the scenario file** (`scenarios/<name>/scenario.yaml`). See [system-design-overview.md](system-design-overview.md#configuration-system) for the full config system.

### Agent Config Example

```yaml
# scenarios/rca_hypothesis/scenario.yaml (agents section)
agents:
  infrastructure:
    model: "gpt-4o-mini"
    temperature: 0.2
    prompt: "prompts/agents/infrastructure.j2"    # Relative to scenario dir
    tools: [check_cpu, check_memory, check_disk, check_network]
    execution:
      max_steps: 20
      timeout: 120
      retry:
        max_attempts: 3
        initial_interval: 1.0
    compression:
      compression_model: "gpt-4o-mini"

  database:
    model: "gpt-4"
    temperature: 0.1
    prompt: "prompts/agents/database.j2"
    tools: [get_db_metrics, analyze_slow_queries, check_connections, check_locks]
    tool_settings:
      analyze_slow_queries:
        threshold_ms: 500
    execution:
      max_steps: 30
      timeout: 300
      interrupt_before: ["tools"]
      retry:
        max_attempts: 3
        initial_interval: 1.0
    compression:
      compression_model: "gpt-4o-mini"
```

### Model Selection Guidelines

| Agent Type | Model | Temperature | Rationale | Typical task_type |
|-----------|-------|-------------|-----------|-------------------|
| Infrastructure | gpt-4o-mini | 0.2 | Metric collection, lightweight | scout, verify |
| Logs | gpt-4o-mini | 0.3 | Pattern matching, moderate | scout, deep_analyze |
| Database | gpt-4 | 0.1 | SQL accuracy critical | verify, deep_analyze |
| Analyzer | gpt-4 | 0.5 | Creative reasoning | deep_analyze, verify |

---

## Prompt Management

System prompts are Jinja2 templates (`.j2`), stored in the scenario's `prompts/agents/` directory. Each Sub-Agent has a **base prompt** configured per agent_id, plus a **task_type overlay** that adjusts behavior based on the task type dispatched by the Orchestrator.

### Task Types

The Orchestrator dispatches tasks with a `task_type` parameter that determines the Sub-Agent's investigation approach and output format:

| Task Type | Purpose | Output Focus | Typical Budget |
|-----------|---------|-------------|----------------|
| **scout** | Initial reconnaissance — discover anomalies, map topology | What's wrong and where | Higher (broad exploration) |
| **verify** | Test a specific hypothesis with targeted evidence | Verdict (supported/contradicted/inconclusive) + evidence | Medium (focused queries) |
| **deep_analyze** | Focused deep dive into specific data source or service | Precise quantitative data + causal chain | Medium-High (detailed) |

### Base Agent Prompt Template

```jinja2
{# prompts/agents/{{ agent_id }}.j2 — shared base for all task types #}

You are a {{ agent_id }} diagnostics specialist (Agent: {{ agent_id }}).

<data_sources>
{{ data_source_description }}
</data_sources>

<tools>
Available tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
</tools>

<tool_discipline>
You have a budget of {{ tool_call_budget }} tool calls for this task. Use them wisely.

1. **THINK FIRST**: Before any data query, reason about what you're looking for and why.
   Choose which 1-2 tools to call next with specific parameters.
2. **CALL INCREMENTALLY**: Make 1-3 tool calls per round, then analyze results before
   calling more. Do NOT batch all queries at once.
3. **ANALYZE BETWEEN ROUNDS**: After receiving results, assess what you learned,
   identify gaps, and decide next steps based on evidence.
4. **COMPARE BASELINES**: When checking for anomalies, compare against normal/baseline
   data to confirm deviations are real, not normal behavior.
</tool_discipline>

<output>
Your response will be sent to the Orchestrator. Be PRECISE and DATA-DENSE:
- Use exact identifiers (service names, hosts, metrics) — no vague descriptions
- Include specific values and timestamps
- Omit reasoning process and hedging — only report findings
- Do NOT echo raw tool outputs — extract and summarize key data points
</output>

{% block task_specific %}{% endblock %}
```

### Scout Task Overlay

```jinja2
{# prompts/task_types/scout.j2 — extends base agent prompt #}
{% extends agent_base_prompt %}

{% block task_specific %}
<mission>
Perform initial reconnaissance for the incident investigation:
- Enumerate available data sources and their contents
- Identify anomalies, error patterns, and timing relationships
- Map involved components and their dependencies
- Report what data is available for deeper investigation

Focus on WHAT anomalies exist and WHERE — deep analysis comes later.
</mission>

<anomaly_definition>
A true anomaly must be significantly different from baseline behavior.
Cross-check every suspected anomaly against normal/baseline data.
Discard anything present in both periods — that is normal behavior.
</anomaly_definition>
{% endblock %}
```

### Verify Task Overlay

```jinja2
{# prompts/task_types/verify.j2 — extends base agent prompt #}
{% extends agent_base_prompt %}

{% block task_specific %}
<mission>
Test the assigned hypothesis with targeted evidence gathering.

Verdict options:
- **SUPPORTED**: Evidence satisfies temporal, spatial, and causal checks
- **CONTRADICTED**: Strong contradicting evidence or broken causal chain
- **INCONCLUSIVE**: Insufficient or ambiguous evidence

Your report MUST include:
- Verdict with one-sentence justification
- Supporting evidence (what confirms the hypothesis)
- Contradicting evidence (what argues against it — actively look for this)
- Key data points with exact values
</mission>

<critical_evaluation>
- **Causation ≠ correlation**: Temporal co-occurrence does NOT prove causation.
  Require the actual propagation mechanism.
- **Symptoms ≠ causes**: "Service X has high error rate" is a symptom. Ask WHY.
- **Cross-validate**: Confirm findings across multiple sources. One source alone
  cannot confirm a hypothesis.
</critical_evaluation>
{% endblock %}
```

### Deep Analyze Task Overlay

```jinja2
{# prompts/task_types/deep_analyze.j2 — extends base agent prompt #}
{% extends agent_base_prompt %}

{% block task_specific %}
<mission>
Perform a focused, in-depth investigation of the specific data source,
time range, or component you've been assigned.

- Extract precise timing, correlation, and causation information
- Surface patterns that high-level analysis might miss
- Quantify anomalies with exact data points and baseline comparisons
- Quantify everything: "CPU 85% abnormal vs 12% normal", not "CPU was high"
</mission>
{% endblock %}
```

### Compression Prompt Template

When Sub-Agent message history is compressed (via `pre_model_hook`), this prompt guides the compression into a structured 8-section summary (see `prompts/compression/sub_agent_compress.j2`):

1. **Task Assignment** — original instruction, target hypothesis ID
2. **Key Findings** — confirmed anomalies with exact values, timestamps, service names
3. **Affected Services** — per-service anomaly (abnormal vs normal values)
4. **Service Call Chains** — trace-backed propagation paths
5. **Errors and Anomaly Patterns** — error types, log signatures, temporal correlations
6. **Hypothesis Assessment** — per-hypothesis verdict
7. **Open Questions** — unresolved items, suggested follow-ups
8. **Investigation Progress** — tools called, data sources covered vs unchecked

### Context Briefing Rule

> **Critical**: Sub-Agents run in isolation — they ONLY see what the Orchestrator writes in the `task` instruction. The Orchestrator's system prompt enforces that every dispatch instruction must include relevant prior findings, specific signals to investigate, and which hypothesis is being tested. See [orchestrator.md](orchestrator.md#system-prompt-guidelines-not-enforced-by-graph) for the context briefing rules.

### Configuration

```yaml
# scenarios/rca_hypothesis/scenario.yaml (agents section)
agents:
  infrastructure:
    model: "gpt-4o-mini"
    temperature: 0.2
    prompt: "prompts/agents/infrastructure.j2"
    task_type_prompts:                          # Task type overlays
      scout: "prompts/task_types/scout.j2"
      verify: "prompts/task_types/verify.j2"
      deep_analyze: "prompts/task_types/deep_analyze.j2"
    tools: [check_cpu, check_memory, check_disk, check_network]
    execution:
      tool_call_budget: 20                      # Per-task tool call budget
      timeout: 120
      retry:
        max_attempts: 3
        initial_interval: 1.0
    compression:
      prompt: "prompts/compression/sub_agent_compress.j2"
      compression_model: "gpt-4o-mini"
```

Loading:
```python
from jinja2 import Template

def load_prompt_template(path: str, **context) -> str:
    template = Template(Path(path).read_text())
    return template.render(**context)
```

---

## Tool Registry

Tools are registered centrally and bound to agents via config:

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, func: Callable, config_schema: dict):
        self._tools[name] = ToolDefinition(name, func, config_schema)

    def get_tool(self, name: str) -> ToolDefinition:
        return self._tools[name]

class ToolDefinition:
    def create_with_config(self, **config) -> Tool:
        """Create a LangChain Tool instance with bound config parameters."""
        def tool_with_config(**args):
            return self.func(config=config, **args)
        return Tool(name=self.name, func=tool_with_config, description=self.description)
```

---

## Communication with Orchestrator

### Data Collection (Phase 1)

```python
# Orchestrator dispatches via tool
# dispatch_agent("infrastructure", "Scan infrastructure metrics: CPU, memory, disk, network")

# Sub-Agent returns a diagnostic report (natural language with inline data)
AIMessage(content="""Scanned infrastructure metrics for initial assessment.

CPU usage at 85% — elevated, primarily from postgres (45%) and java service (28%).
Memory at 40% — within normal range, no swap activity.
Disk I/O: read 120 MB/s, write 80 MB/s — moderate, no saturation.
Network: latency 2ms avg, no packet loss, bandwidth utilization 30%.

Notable: CPU spike correlates with the incident start time (14:30 UTC).
inode usage at 92% on /var/log — not critical yet but worth monitoring.

Raw: cpu=0.85, memory=0.40, disk_read=120MB/s, disk_write=80MB/s, net_latency=2ms""")
```

### Hypothesis Verification (Phase 3)

```python
# Orchestrator dispatches via tool
# dispatch_agent("database", "Verify H1: Database connection pool exhaustion.
#   Focus on pool_size, active connections, wait queue, slow queries.")

# Sub-Agent returns VerificationResult as JSON with verdict + natural language report
AIMessage(content=json.dumps({
    "verdict": "confirmed",
    "report": """Investigated database connection pool status for H1 (pool exhaustion).

Connection pool is at capacity: 100/100 active connections, 45 requests in wait queue.
Slow query analysis found 3 queries exceeding 2s, the worst being a JOIN on orders table (4.2s).
Lock wait time averages 320ms across active transactions.

Supporting: Pool completely saturated, significant wait queue, slow queries holding connections.
Contradicting: Primary CPU at 42% (not a CPU bottleneck), replica lag only 0.8s.
Unexpected: Connection timeout errors spiked from 0 to 12 in the last 15 minutes.

Key conclusion: Connection pool exhaustion is the primary bottleneck.
Raw: pool_size=100, active=100, waiting=45, slow_queries_count=3, primary_cpu=42%, replica_lag=0.8s"""
}))
```

---

## Orchestrator Intervention

> **Design Decision**: Sub-Agents run asynchronously as `asyncio.Task`s managed by the TaskManager. The Orchestrator monitors them via `check_tasks` tool and intervenes via `inject_instruction` or `abort_task` when necessary — see [orchestrator.md](orchestrator.md#taskmanager--orchestrator-tools).
>
> The `interrupt_before` mechanism is reserved for **safety-critical agents** (e.g., database agents executing destructive queries) where external review is mandatory before tool execution.

### Safety Interrupts (Optional, Per-Agent)

For agents that execute potentially dangerous operations (e.g., database DDL, production config changes), `interrupt_before: ["tools"]` pauses the agent before every tool call for **human operator approval**.

> **Design Decision**: Safety interrupts are a **Layer 4 (manual)** concern — they require human judgment, not LLM automation. The Orchestrator is not involved in the approval flow. Instead, the external operator (via frontend Debug Panel or REST API) reviews pending tool calls and resumes execution.

Workflow:
1. Sub-Agent's `create_react_agent` is compiled with `interrupt_before=["tools"]`
2. Agent pauses before each tool call → checkpoint saved
3. Operator reviews pending tool call via `GET /api/tasks/{thread_id}/state`
4. Operator approves → `POST /api/tasks/{thread_id}/resume` with `Command(resume=True)`
5. Or operator rejects → `POST /api/tasks/{thread_id}/resume` with modified instructions

Configuration:
```yaml
agents:
  database:
    execution:
      interrupt_before: ["tools"]   # Requires human approval for all tool calls
```

### Orchestrator-Driven Intervention

For non-safety scenarios, the Orchestrator uses tools to intervene:

```python
# Orchestrator detects agent is stuck (via check_tasks tool)
# → Injects new instruction
inject_instruction("task-001", "Skip replica, focus on primary pool metrics")

# Orchestrator determines agent's task is no longer needed
# → Aborts agent
abort_task("task-002", "H1 already confirmed by database agent")
```

These tools are backed by the `TaskManager` which queues instructions via a per-task instruction queue (consumed by Sub-Agent's `pre_model_hook`), and uses `asyncio.Task.cancel()` for abort.

---

## Agent Pool

> **Design Decision**: All Sub-Agents are `create_react_agent` instances differing only in config (tools, prompts, model, temperature). The graph topology is static — compiled once at startup from YAML config. Adding/removing agents requires restarting the service and recompiling the graph.
>
> Dynamic runtime addition/removal was considered but deemed unnecessary: the agent pool is determined by the diagnostic scenario's config, not runtime conditions.

> **Implementation Note**: The current implementation uses a single "worker" agent configuration with `task_type_prompts` for scout/verify/deep_analyze, NOT a per-agent-id pool. `AgentPool.get_worker(task_type)` lazily creates one compiled subgraph per `task_type`. The `agent_id` passed to `dispatch_agent` is used for naming and logging but all agents share the same worker config. Per-agent-id configuration (different models, tools, prompts per agent_id) is a future enhancement.

Agent pool is built at startup from config:

```python
class AgentPool:
    """Collection of independently compiled Sub-Agent subgraphs.

    Sub-Agents are NOT added as graph nodes. They are compiled independently
    and managed by the TaskManager, which launches them as asyncio.Tasks.
    """

    def __init__(self, scenario_config: ScenarioConfig, tool_registry: ToolRegistry):
        self.agents: dict[str, CompiledGraph] = {}
        for agent_id, agent_config in scenario_config.agents.items():
            self.agents[agent_id] = create_sub_agent(
                agent_id, agent_config, tool_registry
            )

    def get_agent(self, agent_id: str) -> CompiledGraph:
        """Get a compiled sub-agent subgraph by ID."""
        return self.agents[agent_id]
```

> **Integration with AgentSystemBuilder**: `AgentPool` is created during `AgentSystemBuilder.build()`. The builder loads scenario config, creates the pool of compiled Sub-Agent subgraphs, and passes them to the `TaskManager`. See [generic-state-wrapper.md](generic-state-wrapper.md#agentsystembuilder) for the full build process.

The Orchestrator graph is built separately — it contains only the Orchestrator `create_react_agent` as its sole node. Sub-Agents are passed to the `TaskManager` for async invocation.

---

## Error Handling

Errors are handled at four layers — see [system-design-overview.md](system-design-overview.md#error-handling-layers) for the full model.

Sub-Agent level (Layer 1 + 2):

```python
# Layer 1: LLM self-correction via handle_tool_errors=True (default in create_react_agent)
# Tool errors are returned as ToolMessage, LLM reasons about them and tries alternatives

# Layer 2: API-level retry via LangChain's built-in model retry
# ChatOpenAI(max_retries=3) handles transient API errors (429, 5xx, timeouts)
# For task-level retry, the TaskManager wraps subgraph execution:
#   try: await subgraph.ainvoke(...)
#   except: retry with exponential backoff (configured per agent)
```

> **⚠️ Note**: LangGraph's `RetryPolicy` only applies to nodes within a `StateGraph` (via `add_node(retry_policy=...)`). Since Sub-Agents are independently compiled subgraphs launched as `asyncio.Task`s (not nodes in the Root graph), `RetryPolicy` cannot be applied to them directly. Retry logic is implemented at two levels:
> 1. **API layer**: `ChatOpenAI(max_retries=...)` handles transient HTTP errors
> 2. **Task layer**: TaskManager's `_execute_agent` wraps execution with retry + exponential backoff

Error types and handling:
- Tool error → Layer 1: returned as ToolMessage for LLM self-correction
- Rate limit / timeout / server error → Layer 2: RetryPolicy automatic retry
- Fatal (all retries exhausted) → Layer 3: bubble up to Orchestrator via `check_tasks` with error summary + last steps

---

## Related Documents

- [System Architecture](system-design-overview.md) — Overall system design
- [Orchestrator](orchestrator.md) — Orchestrator design and hypothesis flow
- [Generic State Wrapper](generic-state-wrapper.md) — SDK framework

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| create_react_agent | Pre-built ReAct loop, verify import path at implementation time |
| Independent State Schema | Full isolation; Sub-Agents are compiled independently, not graph nodes. No state mapping — communication via tools only |
| Independent Sub-Agent checkpointer | Each Sub-Agent has its own thread_id via TaskManager; enables trajectory export and replay per agent |
| Config-driven tools/prompts | New agents via YAML only, no code changes |
| Data-only constraint (RCA) | All reasoning centralized in Orchestrator for traceability |
| pre_model_hook compression | **✅ Verified: `llm_input_messages` return key preserves full state in checkpoints. Triggered at 80% context limit** |
| Static agent pool | Graph compiled once at startup. No runtime hot-reload — simplicity over dynamism |
| Pull-based monitoring | Orchestrator uses `check_tasks` tool in ReAct loop; sub-agents run as asyncio.Tasks via TaskManager |
| recursion_limit for step control | **⚠️ No native `max_steps` param — must translate to `recursion_limit` in config** |
