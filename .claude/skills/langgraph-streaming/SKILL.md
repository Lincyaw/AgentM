---
name: langgraph-streaming
description: "INVOKE THIS SKILL when implementing streaming in LangGraph — LLM tokens, state updates, custom events, subgraph streaming. Covers all stream modes, StreamWriter, nested streaming, and async patterns."
---

<overview>
LangGraph supports multiple streaming modes for real-time output:

- **values**: Full state after each step
- **updates**: State deltas per node
- **messages**: LLM token-by-token streaming
- **custom**: User-defined events via StreamWriter
- **debug**: Comprehensive debug information
- **checkpoints**: Checkpoint creation events
- **tasks**: Task start/finish events

Multiple modes can be combined in a single `stream()` call.
</overview>

<stream-mode-selection>

| Mode | What It Yields | Best For |
|------|---------------|----------|
| `"values"` | Complete state after each step | Monitoring full state transitions |
| `"updates"` | `{node_name: {updates}}` per step | Tracking incremental changes |
| `"messages"` | `(AIMessageChunk, metadata)` tuples | Chat UIs, token-by-token display |
| `"custom"` | User-defined data via StreamWriter | Progress bars, status updates |
| `"debug"` | Comprehensive debug events | Development, troubleshooting |
| `"checkpoints"` | Checkpoint creation events | Monitoring persistence |
| `"tasks"` | Task start/finish with results | Execution timeline tracking |

</stream-mode-selection>

---

## Basic Streaming

<ex-stream-values>
<python>
Stream complete state after each step.
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    input: str
    step: int
    result: str

def step_1(state: State) -> dict:
    return {"step": 1, "result": "Step 1 done"}

def step_2(state: State) -> dict:
    return {"step": 2, "result": "Step 2 done"}

graph = (
    StateGraph(State)
    .add_node("step_1", step_1)
    .add_node("step_2", step_2)
    .add_edge(START, "step_1")
    .add_edge("step_1", "step_2")
    .add_edge("step_2", END)
    .compile()
)

# Stream full state at each step
for state in graph.stream({"input": "test"}, stream_mode="values"):
    print(f"Step {state.get('step', 0)}: {state.get('result', 'start')}")
```
</python>
</ex-stream-values>

<ex-stream-updates>
<python>
Stream only the incremental updates from each node.
```python
# Stream deltas — see which node produced what updates
for update in graph.stream({"input": "test"}, stream_mode="updates"):
    for node_name, node_output in update.items():
        print(f"Node '{node_name}' produced: {node_output}")
# Output:
# Node 'step_1' produced: {'step': 1, 'result': 'Step 1 done'}
# Node 'step_2' produced: {'step': 2, 'result': 'Step 2 done'}
```
</python>
</ex-stream-updates>

---

## LLM Token Streaming

Stream individual tokens from LLM calls within nodes.

<ex-stream-messages>
<python>
Stream LLM tokens in real-time for chat UI display.
```python
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

# Assuming graph has an LLM-calling node
config = {"configurable": {"thread_id": "chat-1"}}

for chunk in graph.stream(
    {"messages": [HumanMessage("Explain quantum computing")]},
    config,
    stream_mode="messages",
):
    token, metadata = chunk
    # metadata includes: langgraph_step, langgraph_node, langgraph_triggers
    if hasattr(token, "content") and token.content:
        print(token.content, end="", flush=True)
    # Check which node produced this token
    # print(f"\n[from {metadata.get('langgraph_node')}]")

# Async version
async for chunk in graph.astream(
    {"messages": [HumanMessage("Explain quantum computing")]},
    config,
    stream_mode="messages",
):
    token, metadata = chunk
    if hasattr(token, "content") and token.content:
        print(token.content, end="", flush=True)
```
</python>
</ex-stream-messages>

---

## Custom Event Streaming

Emit custom progress data from within nodes using `StreamWriter` / `get_stream_writer()`.

<ex-stream-custom>
<python>
Emit custom progress updates from within nodes.
```python
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    data: str
    result: str

def processing_node(state: State) -> dict:
    writer = get_stream_writer()

    writer({"type": "progress", "step": "loading", "percent": 0})
    data = load_data(state["data"])

    writer({"type": "progress", "step": "processing", "percent": 50})
    result = process_data(data)

    writer({"type": "progress", "step": "complete", "percent": 100})
    return {"result": result}

graph = (
    StateGraph(State)
    .add_node("process", processing_node)
    .add_edge(START, "process")
    .add_edge("process", END)
    .compile()
)

# Consume custom events
for event in graph.stream({"data": "input.csv"}, stream_mode="custom"):
    if event.get("type") == "progress":
        print(f"[{event['step']}] {event['percent']}%")
```
</python>
</ex-stream-custom>

---

## Multiple Stream Modes

Combine multiple modes in a single `stream()` call to get different event types simultaneously.

<ex-multiple-modes>
<python>
Combine multiple stream modes to get both state updates and custom events.
```python
# Request multiple modes — output becomes (mode, data) tuples
for mode, data in graph.stream(
    {"messages": [HumanMessage("Hello")]},
    stream_mode=["updates", "custom", "messages"],
):
    if mode == "updates":
        for node, output in data.items():
            print(f"[UPDATE] {node}: {output}")
    elif mode == "custom":
        print(f"[CUSTOM] {data}")
    elif mode == "messages":
        token, metadata = data
        if hasattr(token, "content") and token.content:
            print(f"[TOKEN] {token.content}", end="")
```
</python>
</ex-multiple-modes>

---

## Streaming from Subgraphs

Monitor sub-agent execution by streaming their internal events.

<ex-subgraph-streaming>
<python>
Stream events from inside subgraphs using subgraphs=True.
```python
# Enable subgraph streaming — events include namespace path
for namespace, data in graph.stream(
    {"messages": [HumanMessage("Research and code this")]},
    stream_mode="updates",
    subgraphs=True,
):
    if namespace:
        # namespace is a tuple like ("researcher",) or ("eng_team", "coder")
        agent_path = " > ".join(namespace)
        print(f"[Sub-agent: {agent_path}]")
    else:
        print("[Main graph]")
    print(f"  Data: {data}")

# Combine subgraphs with multiple modes
for namespace, mode, data in graph.stream(
    {"messages": [HumanMessage("Complex task")]},
    stream_mode=["updates", "messages"],
    subgraphs=True,
):
    agent = " > ".join(namespace) if namespace else "main"
    if mode == "messages":
        token, meta = data
        if hasattr(token, "content") and token.content:
            print(f"[{agent}] {token.content}", end="")
    elif mode == "updates":
        print(f"\n[{agent}] Update: {data}")
```
</python>
</ex-subgraph-streaming>

<subgraph-streaming-key-points>

### Key Points for Subgraph Streaming

- Set `subgraphs=True` in `stream()` to receive events from all levels
- Events are prefixed with a **namespace tuple** identifying the subgraph path
- Empty namespace `()` means the event is from the top-level graph
- Nested subgraphs produce nested namespaces: `("team_a", "researcher")`
- Works with all stream modes

</subgraph-streaming-key-points>

---

## Debug Stream Mode

Get maximum information for development and troubleshooting.

<ex-stream-debug>
<python>
Use debug mode for comprehensive execution visibility.
```python
for event in graph.stream(
    {"messages": [HumanMessage("Debug this")]},
    stream_mode="debug",
):
    event_type = event.get("type")

    if event_type == "checkpoint":
        print(f"[CHECKPOINT] Step {event['step']}")
        print(f"  Values: {event['values']}")

    elif event_type == "task":
        task = event.get("payload", {})
        print(f"[TASK] {task.get('name')}")
        if task.get('error'):
            print(f"  ERROR: {task['error']}")
        elif task.get('result'):
            print(f"  Result: {task['result']}")
```
</python>
</ex-stream-debug>

---

## Async Streaming

<ex-async-stream>
<python>
Async streaming patterns for web servers and async applications.
```python
import asyncio
from langchain_core.messages import HumanMessage

async def stream_agent_response(user_input: str):
    """Async generator for streaming agent responses."""
    config = {"configurable": {"thread_id": "async-session"}}

    async for chunk in graph.astream(
        {"messages": [HumanMessage(user_input)]},
        config,
        stream_mode="messages",
    ):
        token, metadata = chunk
        if hasattr(token, "content") and token.content:
            yield token.content

# Usage in a web framework (e.g., FastAPI)
# @app.get("/stream")
# async def stream_endpoint(query: str):
#     return StreamingResponse(
#         stream_agent_response(query),
#         media_type="text/event-stream",
#     )

# Async streaming with subgraphs
async for namespace, data in graph.astream(
    {"messages": [HumanMessage("Hello")]},
    config,
    stream_mode="updates",
    subgraphs=True,
):
    agent = " > ".join(namespace) if namespace else "main"
    print(f"[{agent}] {data}")
```
</python>
</ex-async-stream>

---

## print_mode

Use `print_mode` to automatically print streamed output to console without affecting the stream output.

<ex-print-mode>
<python>
Automatically print streamed data while still processing programmatically.
```python
# print_mode outputs to console; stream_mode controls what you iterate over
for update in graph.stream(
    {"messages": [HumanMessage("Hello")]},
    stream_mode="updates",
    print_mode="messages",  # Tokens auto-printed to console
):
    # 'update' contains state updates (stream_mode)
    # LLM tokens are printed to stdout automatically (print_mode)
    pass
```
</python>
</ex-print-mode>

---

## Common Fixes

<fix-stream-mode-tuple>
<python>
When using multiple modes, output is (mode, data) tuples.
```python
# WRONG: Single mode — data is yielded directly
for data in graph.stream(input, stream_mode="updates"):
    print(data)  # dict

# WRONG: Multiple modes — must unpack tuple
for data in graph.stream(input, stream_mode=["updates", "custom"]):
    print(data)  # This is a tuple, not the data!

# CORRECT: Unpack the (mode, data) tuple
for mode, data in graph.stream(input, stream_mode=["updates", "custom"]):
    print(f"Mode: {mode}, Data: {data}")
```
</python>
</fix-stream-mode-tuple>

<fix-subgraph-namespace-tuple>
<python>
With subgraphs=True, output is prefixed with namespace tuple.
```python
# WRONG: Forgetting to account for namespace
for data in graph.stream(input, stream_mode="updates", subgraphs=True):
    print(data)  # This is (namespace, data), not just data!

# CORRECT: Unpack namespace
for namespace, data in graph.stream(input, stream_mode="updates", subgraphs=True):
    print(f"From: {namespace}, Data: {data}")

# With multiple modes AND subgraphs — triple unpack
for namespace, mode, data in graph.stream(
    input, stream_mode=["updates", "messages"], subgraphs=True
):
    print(f"From: {namespace}, Mode: {mode}")
```
</python>
</fix-subgraph-namespace-tuple>

<fix-messages-mode-metadata>
<python>
Messages mode yields (token, metadata) tuples, not just tokens.
```python
# WRONG: Trying to print chunk directly
for chunk in graph.stream(input, stream_mode="messages"):
    print(chunk.content)  # AttributeError! chunk is a tuple

# CORRECT: Unpack token and metadata
for token, metadata in graph.stream(input, stream_mode="messages"):
    if hasattr(token, "content") and token.content:
        print(token.content, end="")
```
</python>
</fix-messages-mode-metadata>

<boundaries>
### What You Should NOT Do

- Assume stream output format without checking mode — each mode has different structure
- Use `stream_mode="messages"` without an LLM node — no tokens to stream
- Forget `subgraphs=True` when you need to monitor sub-agent execution
- Mix up `stream_mode` (what you iterate) and `print_mode` (what auto-prints to console)
- Block the async event loop with sync operations inside `astream` handlers
</boundaries>

---

<deepwiki-tips>

### Need More Details?

Use DeepWiki MCP (`mcp__deepwiki__ask_question`) with `repoName: "langchain-ai/langgraph"` to query:

- **"How does StreamMessagesHandler capture LLM tokens for the messages stream mode?"** — internal token capture mechanism
- **"How does get_stream_writer() work internally in the Pregel engine?"** — custom event emission internals
- **"How does the subgraphs parameter affect stream output namespacing?"** — namespace resolution
- **"How to use astream_events for fine-grained event control?"** — alternative streaming API
- **"How does print_mode differ from stream_mode?"** — output routing details
- **"How to stream from RemoteGraph deployments?"** — streaming in LangGraph Cloud

</deepwiki-tips>
