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
- `max_steps` limit reached
- Interrupt point triggered

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

## Interrupt Points

Sub-Agents can be paused at configured points for Orchestrator review:

```python
# In config: interrupt_before: ["tools"]
# → Agent pauses before executing any tool call

# Orchestrator reviews pending tool calls
state = graph.get_state(config)
pending_tools = state.values.get("pending_tool_calls")

# Approve, modify, or reject
graph.update_state(config, {"pending_tool_calls": filtered_calls})
graph.invoke(Command(resume=True), config)
```

---

## Dynamic Agent Pool

Support runtime addition/removal of Sub-Agents via config hot-reload:

```python
class DynamicSubAgentPool:
    def add_agent(self, agent_id: str, agent_config: dict):
        agent = create_sub_agent(agent_id, AgentConfig(**agent_config), self.tool_registry)
        self.builder.add_node(agent_id, agent)
        self.builder.add_edge(agent_id, "orchestrator")
        self._rebuild_graph()

    def remove_agent(self, agent_id: str):
        del self.agents[agent_id]
        self._rebuild_graph()
```

Triggered by file watcher (watchdog) on config directory changes.

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
| create_react_agent | Pre-built ReAct loop, works out of the box |
| Independent State Schema | Full isolation; avoids state pollution |
| Same-name field mapping | Automatic, no explicit mapping code needed |
| No Sub-Agent checkpointer | Root manages all checkpoints; simpler architecture |
| Config-driven tools/prompts | New agents via YAML only, no code changes |
| Data-only constraint (RCA) | All reasoning centralized in Orchestrator for traceability |
