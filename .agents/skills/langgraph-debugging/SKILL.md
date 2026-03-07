---
name: langgraph-debugging
description: "INVOKE THIS SKILL when debugging LangGraph applications. Covers LangGraph Studio, LangSmith tracing, graph visualization (Mermaid/PNG/ASCII), XRay mode, debug stream, step-by-step inspection, and state introspection."
---

<overview>
LangGraph provides multiple debugging and visualization tools:

- **Graph visualization**: `draw_mermaid()`, `draw_png()`, `draw_ascii()` for static structure
- **XRay mode**: Expand subgraphs to see internal structure
- **Debug stream**: Real-time comprehensive execution events
- **State inspection**: `get_state()`, `get_state_history()` for post-hoc analysis
- **LangGraph Studio**: Visual IDE for prototyping and debugging
- **LangSmith**: Production tracing and observability platform
</overview>

<debugging-tool-selection>

| Tool | When to Use | Level |
|------|------------|-------|
| `draw_mermaid()` | Verify graph structure at design time | Static |
| XRay mode | Inspect subgraph internals | Static |
| `stream(mode="debug")` | Trace execution in real-time | Runtime |
| `get_state(config)` | Inspect current state at any point | Runtime |
| `get_state_history(config)` | Walk full execution history | Post-hoc |
| LangGraph Studio | Visual prototyping and interactive debugging | IDE |
| LangSmith | Production tracing, evaluation, monitoring | Production |

</debugging-tool-selection>

---

## Graph Visualization

Visualize graph structure to verify node connections and routing logic.

<ex-draw-mermaid>
<python>
Generate a Mermaid diagram of the graph structure.
```python
from langgraph.graph import StateGraph, START, END

# After building the graph
graph = builder.compile()

# Generate Mermaid markdown
mermaid_str = graph.get_graph().draw_mermaid()
print(mermaid_str)
# Output:
# %%{init: {'flowchart': {'curve': 'linear'}}}%%
# graph TD;
#   __start__([__start__]):::first
#   supervisor(supervisor)
#   researcher(researcher)
#   coder(coder)
#   __end__([__end__]):::last
#   __start__ --> supervisor
#   supervisor -.-> researcher
#   supervisor -.-> coder
#   supervisor -.-> __end__
#   researcher --> supervisor
#   coder --> supervisor

# Save to file for rendering
with open("graph_structure.md", "w") as f:
    f.write(f"```mermaid\n{mermaid_str}\n```")

# Generate PNG image (requires graphviz)
png_data = graph.get_graph().draw_png()
with open("graph_structure.png", "wb") as f:
    f.write(png_data)

# ASCII representation (no dependencies needed)
ascii_art = graph.get_graph().draw_ascii()
print(ascii_art)
```
</python>
</ex-draw-mermaid>

---

## XRay Mode — Inspect Subgraphs

Expand subgraph nodes to see their internal structure.

<ex-xray>
<python>
Use XRay mode to visualize nested subgraph internals.
```python
# Default: Subgraphs appear as single nodes
print(graph.get_graph().draw_mermaid())
# Shows: supervisor -> research_team -> eng_team (opaque boxes)

# XRay level 1: Expand one level of subgraphs
print(graph.get_graph(xray=1).draw_mermaid())
# Shows: supervisor -> [research_lead -> web_searcher -> paper_reader] -> [eng_lead -> coder -> tester]

# XRay level True: Expand ALL levels recursively
print(graph.get_graph(xray=True).draw_mermaid())
# Shows full hierarchy down to leaf nodes

# XRay level 2: Expand up to 2 levels deep
print(graph.get_graph(xray=2).draw_mermaid())

# Save XRay visualization as PNG
png = graph.get_graph(xray=True).draw_png()
with open("graph_xray.png", "wb") as f:
    f.write(png)
```
</python>
</ex-xray>

---

## Debug Stream Mode

Get maximum execution detail in real-time.

<ex-debug-stream>
<python>
Use debug stream mode for comprehensive execution visibility.
```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "debug-1"}}

for event in graph.stream(
    {"messages": [HumanMessage("Debug this workflow")]},
    config,
    stream_mode="debug",
):
    event_type = event.get("type")
    payload = event.get("payload", {})

    if event_type == "checkpoint":
        step = event.get("step", "?")
        print(f"\n{'='*50}")
        print(f"CHECKPOINT at step {step}")
        print(f"  Values: {payload.get('values', {})}")

    elif event_type == "task":
        name = payload.get("name", "unknown")
        status = "ERROR" if payload.get("error") else "OK"
        print(f"  TASK '{name}': {status}")
        if payload.get("result"):
            print(f"    Result: {payload['result']}")
        if payload.get("error"):
            print(f"    Error: {payload['error']}")
```
</python>
</ex-debug-stream>

---

## Step-by-Step State Inspection

Inspect graph state at any point during or after execution.

<ex-state-inspection>
<python>
Inspect current state and execution history for debugging.
```python
config = {"configurable": {"thread_id": "debug-session"}}

# Run the graph
result = graph.invoke({"messages": ["Process this"]}, config)

# Inspect current state
state = graph.get_state(config)
print(f"Current values: {state.values}")
print(f"Next nodes to execute: {state.next}")  # Empty if complete
print(f"Metadata: {state.metadata}")

# Check for errors in tasks
for task in (state.tasks or []):
    print(f"  Task: {task.name}")
    if hasattr(task, "error") and task.error:
        print(f"  ERROR: {task.error}")

# Walk execution history
print("\n--- Execution History ---")
for snapshot in graph.get_state_history(config):
    step = snapshot.metadata.get("step", "?")
    source = snapshot.metadata.get("source", "?")
    next_nodes = snapshot.next
    print(f"Step {step} ({source}): next={next_nodes}")
    print(f"  Messages: {len(snapshot.values.get('messages', []))}")
```
</python>
</ex-state-inspection>

---

## Interactive Debugging with update_state

Modify state and resume for interactive debugging sessions.

<ex-interactive-debug>
<python>
Modify state at any checkpoint and resume for interactive debugging.
```python
config = {"configurable": {"thread_id": "interactive-debug"}}

# Run until a specific point (using interrupt_before)
graph_with_breakpoints = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["critical_node"],  # Breakpoint before this node
)

# Execute — pauses before 'critical_node'
result = graph_with_breakpoints.invoke({"messages": ["Start"]}, config)
print(f"Paused before: {graph_with_breakpoints.get_state(config).next}")

# Inspect state at the breakpoint
state = graph_with_breakpoints.get_state(config)
print(f"State at breakpoint: {state.values}")

# Option 1: Continue as-is
result = graph_with_breakpoints.invoke(None, config)

# Option 2: Modify state and continue
graph_with_breakpoints.update_state(config, {
    "messages": state.values["messages"] + ["[Debug] Injected message"],
})
result = graph_with_breakpoints.invoke(None, config)

# Option 3: Skip the node — jump to a different one
from langgraph.types import Command
result = graph_with_breakpoints.invoke(
    Command(goto="alternative_node"),
    config,
)
```
</python>
</ex-interactive-debug>

---

## LangGraph Studio

Visual IDE for prototyping and debugging agent workflows.

<langgraph-studio-setup>

### Setup

1. **Install LangGraph CLI**: `pip install langgraph-cli`
2. **Create `langgraph.json`** in your project root:

```json
{
  "dependencies": ["."],
  "graphs": {
    "my_agent": "./src/agent.py:graph"
  },
  "env": ".env"
}
```

3. **Launch dev server**:

```bash
langgraph dev
# Opens LangGraph Studio at http://localhost:8123
# Studio URL will be printed in console output
```

### Features

- **Visual graph editor**: See graph structure and connections
- **Interactive execution**: Step through nodes, inspect state at each step
- **State editor**: Modify state between steps
- **Thread management**: Switch between conversation threads
- **Streaming view**: Watch LLM tokens and events in real-time

</langgraph-studio-setup>

---

## LangSmith Tracing

Production-grade observability for agent execution.

<ex-langsmith-tracing>
<python>
Configure LangSmith tracing for production monitoring.
```python
import os

# Set environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."  # Your LangSmith API key
os.environ["LANGCHAIN_PROJECT"] = "agentm-production"

# All graph executions are now automatically traced
# View traces at: https://smith.langchain.com

# Add custom tags and metadata for filtering
result = graph.invoke(
    {"messages": ["Process this"]},
    config={
        "configurable": {"thread_id": "t1"},
        "tags": ["production", "v2.0"],
        "metadata": {
            "user_id": "alice",
            "request_source": "api",
        },
    },
)

# Tags and metadata appear in LangSmith for filtering and grouping
```
</python>
</ex-langsmith-tracing>

<langsmith-features>

### LangSmith Key Features

| Feature | Description |
|---------|-------------|
| **Trace viewer** | Visual execution tree with timing and token usage |
| **Feedback** | Add human feedback to traces for evaluation |
| **Datasets** | Create evaluation datasets from traces |
| **Monitoring** | Real-time dashboards for latency, errors, costs |
| **Evaluation** | Run agents against datasets and score results |
| **Annotation** | Human annotators review and label agent outputs |

</langsmith-features>

---

## Debugging Patterns

<debugging-patterns>

### Common Debugging Workflows

**1. "Why did the agent choose this path?"**
```python
# Use debug stream to see conditional edge decisions
for event in graph.stream(input, config, stream_mode="debug"):
    if event.get("type") == "task":
        print(f"Executed: {event['payload']['name']}")
```

**2. "What was the state before the error?"**
```python
# Inspect state at the failure point
state = graph.get_state(config)
print(state.values)  # State when error occurred
print(state.tasks)   # Which task errored
```

**3. "I want to re-run from step 5 with different input"**
```python
# Fork from a specific checkpoint
history = list(graph.get_state_history(config))
step_5 = next(s for s in history if s.metadata.get("step") == 5)
graph.update_state(step_5.config, {"messages": ["new input"]})
result = graph.invoke(None, step_5.config)
```

**4. "Which sub-agent is producing bad output?"**
```python
# Stream with subgraphs to identify the culprit
for ns, data in graph.stream(input, config, stream_mode="updates", subgraphs=True):
    agent = " > ".join(ns) if ns else "main"
    print(f"[{agent}] {data}")
```

**5. "Is my graph structure correct?"**
```python
# Visualize with XRay
print(graph.get_graph(xray=True).draw_mermaid())
```

</debugging-patterns>

---

## Common Fixes

<fix-no-graphviz>
<python>
draw_png requires graphviz system package.
```bash
# If draw_png fails with "graphviz not found"
# macOS
brew install graphviz

# Ubuntu
sudo apt-get install graphviz

# Or use draw_mermaid (no dependencies) or draw_ascii
mermaid = graph.get_graph().draw_mermaid()
ascii_art = graph.get_graph().draw_ascii()
```
</python>
</fix-no-graphviz>

<fix-xray-subgraph>
<python>
XRay only expands compiled subgraphs added as nodes.
```python
# WRONG: Function node won't expand with XRay
def my_node(state):
    subgraph.invoke(state)  # Invisible to XRay

# CORRECT: Add compiled subgraph directly as a node
builder.add_node("sub_agent", compiled_subgraph)  # XRay can expand this
```
</python>
</fix-xray-subgraph>

<fix-debug-stream-volume>
<python>
Debug mode produces high volume — filter by event type.
```python
# Filter debug events to reduce noise
for event in graph.stream(input, config, stream_mode="debug"):
    # Only show task completions, skip checkpoints
    if event.get("type") == "task" and event.get("payload", {}).get("result"):
        print(f"Completed: {event['payload']['name']}")
```
</python>
</fix-debug-stream-volume>

<boundaries>
### What You Should NOT Do

- Use `draw_png` without installing graphviz system package — use `draw_mermaid` instead
- Rely only on `print()` debugging — use structured streaming and state inspection
- Forget `subgraphs=True` when debugging multi-agent issues — sub-agent events are hidden by default
- Use debug stream mode in production — high overhead, use "updates" or "tasks" instead
- Expect XRay to show function-internal subgraphs — only compiled subgraphs added as nodes are visible
</boundaries>

---

<deepwiki-tips>

### Need More Details?

Use DeepWiki MCP (`mcp__deepwiki__ask_question`) with `repoName: "langchain-ai/langgraph"` to query:

- **"How to set up langgraph.json for LangGraph Studio?"** — Studio configuration
- **"How does the debug stream mode format its events internally?"** — debug event schema
- **"How to use get_graph with xray parameter for nested subgraph visualization?"** — XRay internals
- **"How to integrate LangSmith tracing with LangGraph checkpointing?"** — unified observability
- **"How to use interrupt_before and interrupt_after for breakpoint debugging?"** — programmatic breakpoints
- **"How does RemoteGraph work with LangGraph Studio?"** — remote debugging

</deepwiki-tips>
