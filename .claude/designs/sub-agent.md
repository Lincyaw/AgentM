# Design: Sub-Agent

**Status**: DRAFT
**Last Updated**: 2026-03-07

---

> **Code conventions**: Same as [orchestrator.md](orchestrator.md#code-conventions-in-this-document). Data structures (dataclass, TypedDict) are **normative**; function logic is **illustrative**.

## Overview

Sub-Agents are **independently compiled subgraphs**, created via `create_react_agent`. They are launched asynchronously by the **TaskManager** as `asyncio.Task`s — they are NOT nodes in the Root StateGraph. Sub-Agents receive tasks from the Orchestrator (via `dispatch_agent` tool), execute tool calls, and return results (via `get_result` tool).

**Key constraint**: In hypothesis-driven RCA, Sub-Agents act as **data collectors** — they return raw data and structured verification results, not independent reasoning.

---

## State Schema

Each Sub-Agent has its own State Schema. Since Sub-Agents are independently compiled subgraphs (not nodes in the Root graph), there is **no automatic state mapping** between parent and child. Communication happens exclusively through the Orchestrator's tools (`dispatch_agent` sends instructions, `get_result` retrieves results).

```python
# Sub-Agent state (fully independent, not mapped to Root graph)
class SubAgentState(TypedDict):
    messages: Annotated[list, operator.add]   # Agent's own message history
    scratchpad: list[str]                     # Private working notes
    observations: list[str]                   # Private observations
    tool_call_count: int = 0                  # Tracked for progress reporting
```

Since Sub-Agents are invoked by TaskManager (not as graph nodes), their state is fully private. The Orchestrator receives results through `get_result` tool — it never sees the Sub-Agent's internal state directly.

---

## Implementation via create_react_agent

> **⚠️ API Migration Note**: `create_react_agent` has been deprecated in recent LangGraph versions and moved to `langchain.agents` as `create_agent`. The import path should be updated to `from langchain.agents import create_agent`. Core functionality (state_schema, pre_model_hook, interrupt_before) remains the same. Verify the exact API at implementation time.

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

| Agent Type | Model | Temperature | Rationale |
|-----------|-------|-------------|-----------|
| Infrastructure | gpt-4o-mini | 0.2 | Metric collection, lightweight |
| Logs | gpt-4o-mini | 0.3 | Pattern matching, moderate |
| Database | gpt-4 | 0.1 | SQL accuracy critical |
| Analyzer | gpt-4 | 0.5 | Creative reasoning |

---

## Prompt Management

System prompts are Jinja2 templates (`.j2`), stored in the scenario's `prompts/agents/` directory:

```jinja2
{# scenarios/rca_hypothesis/prompts/agents/infrastructure.j2 #}

You are an infrastructure diagnostics specialist (Agent: {{ agent_id }}).

Your role: Collect system metrics. Return RAW DATA ONLY, no reasoning.

Available tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

Output format (JSON):
{
  "cpu_usage": 0.85,
  "memory_usage": 0.4,
  "disk_io": { ... },
  "network": { ... }
}

IMPORTANT: Do NOT include reasoning like "I think CPU is high because..."
Only return metric values.
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

For agents that execute potentially dangerous operations, `interrupt_before: ["tools"]` can be configured to require external approval:

```python
# In config: interrupt_before: ["tools"]
# → Agent pauses before executing any tool call
# → External runner (not Orchestrator node) reviews and resumes

# External runner reviews pending tool calls
state = graph.get_state(config, subgraphs=True)
# ... inspect sub-agent's pending tool calls ...
graph.invoke(Command(resume=True), config)
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

These tools are backed by the `TaskManager` which calls `subgraph.aupdate_state()` for injection and `asyncio.Task.cancel()` for abort.

---

## Agent Pool

> **Design Decision**: All Sub-Agents are `create_react_agent` instances differing only in config (tools, prompts, model, temperature). The graph topology is static — compiled once at startup from YAML config. Adding/removing agents requires restarting the service and recompiling the graph.
>
> Dynamic runtime addition/removal was considered but deemed unnecessary: the agent pool is determined by the diagnostic scenario's config, not runtime conditions.

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

The Orchestrator graph is built separately — it contains only the Orchestrator `create_react_agent` as its sole node. Sub-Agents are passed to the `TaskManager` for async invocation.

---

## Error Handling

Errors are handled at four layers — see [system-design-overview.md](system-design-overview.md#error-handling-layers) for the full model.

Sub-Agent level (Layer 1 + 2):

```python
# Layer 1: LLM self-correction via handle_tool_errors=True (default in create_react_agent)
# Tool errors are returned as ToolMessage, LLM reasons about them and tries alternatives

# Layer 2: Transient errors handled via RetryPolicy
workflow.add_node(
    "sub_agent",
    sub_agent,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
)
```

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
| create_react_agent | Pre-built ReAct loop, works out of the box. **⚠️ Deprecated — use `create_agent` from `langchain.agents`** |
| Independent State Schema | Full isolation; Sub-Agents are compiled independently, not graph nodes. No state mapping — communication via tools only |
| Independent Sub-Agent checkpointer | Each Sub-Agent has its own thread_id via TaskManager; enables `update_state()` for injection |
| Config-driven tools/prompts | New agents via YAML only, no code changes |
| Data-only constraint (RCA) | All reasoning centralized in Orchestrator for traceability |
| pre_model_hook compression | **✅ Verified: `llm_input_messages` return key preserves full state in checkpoints. Triggered at 80% context limit** |
| Static agent pool | Graph compiled once at startup. No runtime hot-reload — simplicity over dynamism |
| Pull-based monitoring | Orchestrator uses `check_tasks` tool in ReAct loop; sub-agents run as asyncio.Tasks via TaskManager |
| recursion_limit for step control | **⚠️ No native `max_steps` param — must translate to `recursion_limit` in config** |
