---
name: langgraph-trajectory
description: "INVOKE THIS SKILL when recording, exporting, or replaying LangGraph execution trajectories. Covers hierarchical traces, trajectory export for RL training, checkpoint history, replay from failure, and serialization formats."
---

<overview>
LangGraph execution trajectories are captured through its checkpointing and streaming systems:

- **Checkpoints**: State snapshots at every super-step, forming a linked list of execution history
- **Streaming events**: Real-time events (updates, tasks, debug) during execution
- **Hierarchical traces**: Subgraph events are namespaced, preserving parent-child relationships
- **Export**: Trajectories can be serialized for analysis, replay, or RL training

**Two complementary approaches:**
- **Post-hoc**: Use `get_state_history()` after execution to walk the checkpoint chain
- **Real-time**: Use `stream(stream_mode="debug", subgraphs=True)` during execution
</overview>

<trajectory-capture-methods>

| Method | When | What You Get | Use Case |
|--------|------|-------------|----------|
| `get_state_history(config)` | After execution | All checkpoints with metadata | Post-hoc analysis, replay |
| `stream(mode="debug")` | During execution | Checkpoint + task events | Real-time monitoring |
| `stream(mode="tasks")` | During execution | Task start/finish events | Execution timeline |
| `stream(subgraphs=True)` | During execution | Namespaced sub-agent events | Hierarchical tracing |
| Custom StreamWriter | During execution | User-defined trajectory data | RL-specific annotations |

</trajectory-capture-methods>

---

## Capturing Hierarchical Traces

Record execution traces that preserve the parent-child relationship between orchestrator and sub-agents.

<ex-hierarchical-trace>
<python>
Capture hierarchical traces from a multi-agent execution.
```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

@dataclass
class TraceEvent:
    timestamp: str
    agent_path: list[str]   # e.g. ["orchestrator", "researcher"]
    node: str
    event_type: str          # "update", "task_start", "task_end", "custom"
    data: dict = field(default_factory=dict)

@dataclass
class ExecutionTrace:
    thread_id: str
    events: list[TraceEvent] = field(default_factory=list)

    def add_event(self, agent_path: list[str], node: str, event_type: str, data: dict):
        self.events.append(TraceEvent(
            timestamp=datetime.utcnow().isoformat(),
            agent_path=agent_path,
            node=node,
            event_type=event_type,
            data=data,
        ))

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self.events], indent=2)

# Capture trace during streaming execution
trace = ExecutionTrace(thread_id="task-1")
config = {"configurable": {"thread_id": "task-1"}}

for namespace, mode, data in graph.stream(
    {"messages": ["Build a REST API"]},
    config,
    stream_mode=["updates", "custom"],
    subgraphs=True,
):
    agent_path = list(namespace) if namespace else ["main"]

    if mode == "updates":
        for node_name, node_output in data.items():
            trace.add_event(agent_path, node_name, "update", node_output)
    elif mode == "custom":
        trace.add_event(agent_path, "custom", "custom", data)

# Export trace
with open("traces/task-1.json", "w") as f:
    f.write(trace.to_json())
```
</python>
</ex-hierarchical-trace>

---

## Checkpoint-Based Trajectory Export

Walk the full checkpoint history after execution to reconstruct the trajectory.

<ex-checkpoint-trajectory>
<python>
Export the complete execution trajectory from checkpoint history.
```python
import json
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "session-1"}}

# Run the graph
result = graph.invoke({"messages": ["Process this task"]}, config)

# Walk checkpoint history (most recent first)
history = list(graph.get_state_history(config))

trajectory = []
for snapshot in reversed(history):  # Chronological order
    step_data = {
        "step": snapshot.metadata.get("step", -1),
        "source": snapshot.metadata.get("source", "unknown"),  # "input", "loop", "update"
        "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
        "parent_checkpoint_id": (
            snapshot.parent_config["configurable"]["checkpoint_id"]
            if snapshot.parent_config else None
        ),
        "next_nodes": list(snapshot.next),
        "values": snapshot.values,
        "tasks": [
            {
                "name": task.name,
                "error": task.error if hasattr(task, "error") else None,
            }
            for task in (snapshot.tasks or [])
        ],
    }
    trajectory.append(step_data)

# Export as JSON
with open("trajectories/session-1.json", "w") as f:
    json.dump(trajectory, f, indent=2, default=str)

print(f"Exported {len(trajectory)} steps")
```
</python>
</ex-checkpoint-trajectory>

---

## Sub-Agent Trajectory Isolation

Export trajectories for individual sub-agents separately.

<ex-subagent-trace-isolation>
<python>
Separate trajectories by agent for individual analysis.
```python
from collections import defaultdict

# During streaming, separate events by agent
agent_traces: dict[str, list] = defaultdict(list)

for namespace, data in graph.stream(
    {"messages": ["Complex multi-agent task"]},
    config,
    stream_mode="updates",
    subgraphs=True,
):
    agent_name = namespace[-1] if namespace else "orchestrator"

    for node_name, node_output in data.items():
        agent_traces[agent_name].append({
            "full_path": list(namespace),
            "node": node_name,
            "output": node_output,
        })

# Export per-agent trajectories
for agent_name, events in agent_traces.items():
    with open(f"trajectories/{agent_name}.json", "w") as f:
        json.dump(events, f, indent=2, default=str)
    print(f"Agent '{agent_name}': {len(events)} events")
```
</python>
</ex-subagent-trace-isolation>

---

## Trajectory Export for RL Training

Structure trajectories in formats suitable for reinforcement learning training.

<ex-rl-trajectory>
<python>
Export trajectories in RL-compatible format (state, action, reward, next_state).
```python
from dataclasses import dataclass, asdict
import json

@dataclass
class RLTransition:
    """Single (s, a, r, s') transition for RL training."""
    state: dict           # Current state snapshot
    action: dict          # Node execution (the "action" taken)
    reward: float         # Reward signal (custom or heuristic)
    next_state: dict      # State after action
    agent: str            # Which agent took this action
    step: int             # Step number in trajectory
    metadata: dict        # Additional context (node name, timing, etc.)

def extract_rl_transitions(graph, config) -> list[RLTransition]:
    """Convert checkpoint history to RL transitions."""
    history = list(graph.get_state_history(config))
    history.reverse()  # Chronological order

    transitions = []
    for i in range(len(history) - 1):
        current = history[i]
        next_snap = history[i + 1]

        # Skip input checkpoints (no action taken)
        if current.metadata.get("source") == "input":
            continue

        # Determine which node executed (the "action")
        tasks = current.tasks or []
        action_info = {
            "nodes_executed": [t.name for t in tasks],
            "errors": [t.error for t in tasks if hasattr(t, "error") and t.error],
        }

        # Compute reward (customize based on your objective)
        reward = compute_reward(current.values, next_snap.values, action_info)

        transitions.append(RLTransition(
            state=current.values,
            action=action_info,
            reward=reward,
            next_state=next_snap.values,
            agent=tasks[0].name if tasks else "unknown",
            step=current.metadata.get("step", i),
            metadata={
                "checkpoint_id": current.config["configurable"]["checkpoint_id"],
                "source": current.metadata.get("source"),
            },
        ))

    return transitions

def compute_reward(state: dict, next_state: dict, action: dict) -> float:
    """Custom reward function — adapt to your training objective."""
    # Example: penalize errors, reward progress
    if action["errors"]:
        return -1.0
    # Reward based on task completion
    messages = next_state.get("messages", [])
    if messages and "completed" in str(messages[-1]).lower():
        return 1.0
    return 0.1  # Small positive reward for progress

# Export for training
config = {"configurable": {"thread_id": "training-run-1"}}
result = graph.invoke({"messages": ["Solve this problem"]}, config)

transitions = extract_rl_transitions(graph, config)

# Save as JSONL (one transition per line — standard RL format)
with open("training_data/run-1.jsonl", "w") as f:
    for t in transitions:
        f.write(json.dumps(asdict(t), default=str) + "\n")

print(f"Exported {len(transitions)} transitions for RL training")
```
</python>
</ex-rl-trajectory>

---

## Real-Time Trajectory Annotation

Use `StreamWriter` to add custom annotations during execution.

<ex-trajectory-annotation>
<python>
Annotate trajectory with custom events from within nodes.
```python
from langgraph.config import get_stream_writer

def annotated_agent(state):
    writer = get_stream_writer()

    # Emit trajectory annotations
    writer({
        "trace_type": "decision",
        "agent": "researcher",
        "reasoning": "User query mentions 'papers', routing to arxiv search",
        "confidence": 0.85,
    })

    result = search_arxiv(state["messages"][-1])

    writer({
        "trace_type": "action_result",
        "agent": "researcher",
        "tool": "arxiv_search",
        "result_count": len(result),
        "quality_score": assess_quality(result),
    })

    return {"messages": [f"Found {len(result)} relevant papers"]}

# Capture annotations during streaming
trajectory_annotations = []
for mode, data in graph.stream(
    input_data,
    config,
    stream_mode=["updates", "custom"],
):
    if mode == "custom" and isinstance(data, dict) and "trace_type" in data:
        trajectory_annotations.append(data)
```
</python>
</ex-trajectory-annotation>

---

## Replay from Failure

Resume execution from the exact point of failure, preserving all prior state.

<ex-replay-from-failure>
<python>
Detect failure in checkpoint history and replay from that point.
```python
config = {"configurable": {"thread_id": "task-with-error"}}

# First attempt — may fail
try:
    result = graph.invoke({"messages": ["Process data"]}, config)
except Exception as e:
    print(f"Execution failed: {e}")

# Inspect the state after failure
state = graph.get_state(config)
print(f"Failed at node(s): {state.next}")
print(f"Current values: {state.values}")

# Check if there were task errors
for task in (state.tasks or []):
    if hasattr(task, "error") and task.error:
        print(f"Task '{task.name}' error: {task.error}")

# Option 1: Retry from failure point (same state)
result = graph.invoke(None, config)  # None = resume from last checkpoint

# Option 2: Fix state before retrying
graph.update_state(config, {
    "messages": state.values["messages"] + ["[System] Retrying with corrected parameters"],
})
result = graph.invoke(None, config)

# Option 3: Fork from a known-good checkpoint
history = list(graph.get_state_history(config))
# Find the last successful checkpoint
good_checkpoint = next(
    s for s in history
    if not any(hasattr(t, "error") and t.error for t in (s.tasks or []))
)
result = graph.invoke(None, good_checkpoint.config)
```
</python>
</ex-replay-from-failure>

---

## Debug Replay from Specific Point

For development: replay from any point in the execution history.

<ex-debug-replay>
<python>
Fork execution from a specific step for debugging.
```python
config = {"configurable": {"thread_id": "debug-session"}}

# Run the full execution
result = graph.invoke({"messages": ["Complex task"]}, config)

# Browse all checkpoints
history = list(graph.get_state_history(config))
for i, snapshot in enumerate(history):
    print(f"[{i}] Step {snapshot.metadata.get('step')}: "
          f"next={snapshot.next}, "
          f"source={snapshot.metadata.get('source')}")

# Fork from step 3 (for example) and try a different approach
target = history[3]  # Pick the checkpoint to replay from
fork_config = graph.update_state(
    target.config,
    {"messages": ["[Debug] Trying alternative approach..."]},
)

# Execute from the forked state — creates a new branch
debug_result = graph.invoke(None, fork_config)

# The original execution history is preserved;
# the fork creates a new checkpoint chain
```
</python>
</ex-debug-replay>

---

## Common Fixes

<fix-history-order>
<python>
get_state_history returns most recent first — reverse for chronological.
```python
# get_state_history returns newest-first
history = list(graph.get_state_history(config))

# WRONG: First element is the LAST checkpoint
first = history[0]  # This is the most recent!

# CORRECT: Reverse for chronological order
chronological = list(reversed(history))
first_step = chronological[0]  # This is the initial state
```
</python>
</fix-history-order>

<fix-resume-none>
<python>
Use None (not empty dict) to resume from the latest checkpoint.
```python
# WRONG: Empty dict treated as new input
graph.invoke({}, config)

# CORRECT: None means "resume from checkpoint"
graph.invoke(None, config)
```
</python>
</fix-resume-none>

<boundaries>
### What You Should NOT Do

- Assume `get_state_history` returns chronological order — it's newest-first
- Use `{}` to resume from checkpoint — use `None`
- Export raw checkpoint objects directly — they may contain non-serializable data, extract `.values` and `.metadata`
- Forget to handle task errors when building RL transitions — errors need special reward signals
- Replay from a checkpoint without understanding that the next nodes will re-execute — code before the checkpoint is not re-run, only the pending nodes
</boundaries>

---

<deepwiki-tips>

### Need More Details?

Use DeepWiki MCP (`mcp__deepwiki__ask_question`) with `repoName: "langchain-ai/langgraph"` to query:

- **"How does get_state_history track parent_config for checkpoint chains?"** — understanding checkpoint linked lists
- **"How does checkpoint metadata track source (input/loop/update/fork)?"** — checkpoint provenance
- **"How to use update_state with Overwrite to replace reducer-managed fields?"** — state modification for replay
- **"How does the tasks field in StateSnapshot work?"** — understanding task results and errors
- **"How to access subgraph checkpoints separately from parent graph?"** — isolated sub-agent history
- **"How does fork via update_state create a new checkpoint branch?"** — branching execution

</deepwiki-tips>
