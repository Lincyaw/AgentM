# Design: Sub-Agent

**Status**: DRAFT
**Last Updated**: 2026-03-07

---

## Overview

Sub-Agents are **Subgraph nodes** in the Root StateGraph. Each Sub-Agent is an independently compiled LangGraph, created via `create_react_agent`. Sub-Agents receive tasks from the Orchestrator, execute tool calls, and return results to the Root graph.

**Key constraint**: In hypothesis-driven RCA, Sub-Agents act as **data collectors** — they return raw data and structured verification results, not independent reasoning.

---

## State Schema Isolation

Each Sub-Agent has its own State Schema. Only fields with the **same name** as the parent are automatically mapped.

```python
# Root graph state
class ExecutorState(TypedDict):
    messages: Annotated[list, operator.add]   # Shared
    notebook: DiagnosticNotebook              # Orchestrator-only

# Sub-Agent state (private)
class SubAgentState(TypedDict):
    messages: Annotated[list, operator.add]   # Mapped to/from Root
    scratchpad: list[str]                     # Private — invisible to Root
    observations: list[str]                   # Private
    tool_call_count: int = 0                  # Private
```

Rules:
- Same-name fields auto-map between parent and child
- Child-only fields are **invisible** to the parent
- Child's return is filtered to only include parent-matching fields

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

> **Context compression**: The `pre_model_hook` compresses message history when tool call count exceeds the configured threshold. Full history remains in state for trajectory export. See [orchestrator.md](orchestrator.md#context-compression-intra-task-memory) for the compression data structures and configuration.

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

### Agent Config (`agents/*.yaml`)

```yaml
# agents/infrastructure.yaml
agent:
  name: "infrastructure"
  description: "Collect infrastructure metrics (CPU, memory, disk, network)"

  model: "gpt-4o-mini"
  temperature: 0.2

  tools:
    - check_cpu
    - check_memory
    - check_disk
    - check_network

  tool_settings:
    check_cpu:
      interval: "5m"
      granularity: "1m"
    check_memory:
      include_gc_stats: true

  prompt:
    template: "templates/agents/infrastructure_system.txt"

  execution:
    max_steps: 20
    timeout: 120
    interrupt_before: []  # No interrupts for infra checks
```

```yaml
# agents/database.yaml
agent:
  name: "database"
  description: "Check database performance, queries, connections"

  model: "gpt-4"
  temperature: 0.1  # Lowest — SQL must be accurate

  tools:
    - get_db_metrics
    - analyze_slow_queries
    - check_connections
    - check_locks
    - explain_query

  tool_settings:
    analyze_slow_queries:
      threshold_ms: 1000
      top_n: 5

  prompt:
    template: "templates/agents/db_system.txt"

  execution:
    max_steps: 30
    timeout: 300
    interrupt_before: ["tools"]  # Orchestrator reviews DB queries
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

System prompts are loaded from external template files (Jinja2):

```
# templates/agents/infrastructure_system.txt

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
# Orchestrator → Sub-Agent (via messages)
HumanMessage(content=json.dumps({
    "task": "Quick scan infrastructure",
    "depth": "overview",
}))

# Sub-Agent → Orchestrator (raw data only)
AIMessage(content=json.dumps({
    "cpu": 0.85,
    "memory": 0.4,
    "disk_io": {"read": 120, "write": 80},
}))
```

### Hypothesis Verification (Phase 3)

```python
# Orchestrator → Sub-Agent
HumanMessage(content=json.dumps({
    "task": "Verify H1: Database connection pool exhaustion",
    "focus_areas": ["pool_size", "active_connections", "wait_time"],
    "depth": "detail",
}))

# Sub-Agent → Orchestrator (three-block VerificationResult)
AIMessage(content=json.dumps({
    "investigation_data": {
        "pool_size": 100,
        "active_connections": 100,
        "waiting_connections": 45,
    },
    "reasoning": {
        "supporting_reasons": ["Pool full (100/100)", "45 waiting"],
        "rejecting_reasons": ["CPU acceptable"],
        "neutral_observations": ["Network normal"],
        "key_findings": ["Pool config insufficient"],
    },
    "verdict": "confirmed",
}))
```

---

## Orchestrator Intervention

> **Design Decision**: Sub-Agents run **uninterrupted** by default. The Orchestrator monitors them via a pull-based `check_agents` tool and intervenes only when necessary — see [orchestrator.md](orchestrator.md#monitoring--intervention).
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
# Orchestrator detects agent is stuck (via check_agents tool)
# → Injects new instruction
inject_instruction("database", "Skip replica, focus on primary pool metrics")

# Orchestrator determines agent's task is no longer needed
# → Aborts agent
abort_agent("infrastructure", "H1 already confirmed by database agent")
```

These tools are backed by the `ExecutionRunner` which calls `graph.update_state()` on the sub-agent's namespace externally.

---

## Agent Pool

> **Design Decision**: All Sub-Agents are `create_react_agent` instances differing only in config (tools, prompts, model, temperature). The graph topology is static — compiled once at startup from YAML config. Adding/removing agents requires restarting the service and recompiling the graph.
>
> Dynamic runtime addition/removal was considered but deemed unnecessary: the agent pool is determined by the diagnostic scenario's config, not runtime conditions.

Agent pool is built at startup from config:

```python
class AgentPool:
    def __init__(self, config_dir: str, tool_registry: ToolRegistry):
        self.agents = {}
        for config_file in Path(config_dir).glob("*.yaml"):
            agent_config = load_agent_config(config_file)
            self.agents[agent_config.name] = create_sub_agent(
                agent_config.name, agent_config, tool_registry
            )

    def build_graph(self, orchestrator_node) -> CompiledGraph:
        """Build and compile the complete graph with all agents."""
        builder = StateGraph(ExecutorState)
        builder.add_node("orchestrator", orchestrator_node)

        for agent_id, agent in self.agents.items():
            builder.add_node(agent_id, agent)
            builder.add_edge(agent_id, "orchestrator")

        builder.add_edge(START, "orchestrator")
        return builder.compile(checkpointer=checkpointer)
```

---

## Error Handling

```python
# create_react_agent handles tool errors via handle_tool_errors=True
# For transient errors, use RetryPolicy:

workflow.add_node(
    "sub_agent",
    sub_agent,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
)
```

Error types:
- Rate limit → exponential backoff
- Timeout → simple retry
- Tool error → returned as ToolMessage for LLM self-correction
- Fatal → bubble up to Orchestrator for decision

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
| Independent State Schema | Full isolation; avoids state pollution. **✅ Verified: `state_schema` param supported, same-name field auto-mapping works** |
| Same-name field mapping | Automatic, no explicit mapping code needed. **✅ Verified** |
| No Sub-Agent checkpointer | Root manages all checkpoints; simpler architecture |
| Config-driven tools/prompts | New agents via YAML only, no code changes |
| Data-only constraint (RCA) | All reasoning centralized in Orchestrator for traceability |
| pre_model_hook compression | **✅ Verified: `llm_input_messages` return key preserves full state in checkpoints** |
| Static agent pool | Graph compiled once at startup. No runtime hot-reload — simplicity over dynamism |
| Pull-based monitoring | Orchestrator uses `check_agents` tool, sub-agents run uninterrupted |
| recursion_limit for step control | **⚠️ No native `max_steps` param — must translate to `recursion_limit` in config** |
