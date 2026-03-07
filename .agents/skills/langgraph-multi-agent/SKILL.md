---
name: langgraph-multi-agent
description: "INVOKE THIS SKILL when building multi-agent systems with LangGraph. Covers supervisor pattern, swarm handoffs, hierarchical teams, state isolation, agent-to-agent messaging, Command.PARENT, and create_react_agent as sub-agent."
---

<overview>
LangGraph supports multiple multi-agent orchestration patterns:

- **Supervisor**: A central orchestrator delegates tasks to specialized sub-agents
- **Swarm / Handoff**: Agents transfer control to each other peer-to-peer
- **Hierarchical**: Nested supervisors managing sub-teams
- **State isolation**: Each agent maintains its own state via subgraph scoping
- **Message passing**: Agents communicate through shared state channels or Command updates
</overview>

<pattern-selection>

| Pattern | When to Use | Complexity |
|---------|------------|------------|
| **Supervisor** | Central coordinator decides which agent to call | Medium |
| **Swarm / Handoff** | Agents autonomously transfer to the right specialist | Medium |
| **Hierarchical** | Large systems with sub-teams (supervisor of supervisors) | High |
| **Fan-out workers** | Same task template, different inputs in parallel | Low |

</pattern-selection>

---

## Supervisor Pattern

The supervisor is a node that decides which sub-agent to call next based on the current state. Sub-agents are either subgraphs or `create_react_agent` instances added as nodes.

<ex-supervisor-basic>
<python>
Build a supervisor that routes tasks to specialized sub-agents.
```python
from typing import Literal, Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage

class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str

def supervisor(state: SupervisorState) -> Command[Literal["researcher", "coder", "__end__"]]:
    """Central orchestrator that decides which agent to call next."""
    # In practice, use an LLM to decide the next agent
    last_message = state["messages"][-1]
    if "research" in last_message.content.lower():
        return Command(goto="researcher")
    elif "code" in last_message.content.lower():
        return Command(goto="coder")
    return Command(goto=END)

def researcher(state: SupervisorState) -> dict:
    """Research specialist sub-agent."""
    return {"messages": [SystemMessage(content="[Researcher] Found relevant papers...")]}

def coder(state: SupervisorState) -> dict:
    """Coding specialist sub-agent."""
    return {"messages": [SystemMessage(content="[Coder] Implementation complete...")]}

graph = (
    StateGraph(SupervisorState)
    .add_node("supervisor", supervisor)
    .add_node("researcher", researcher)
    .add_node("coder", coder)
    .add_edge(START, "supervisor")
    .add_edge("researcher", "supervisor")  # Return to supervisor after each agent
    .add_edge("coder", "supervisor")
    .compile()
)

result = graph.invoke({"messages": [HumanMessage(content="Research LLM papers")]})
```
</python>
</ex-supervisor-basic>

### Supervisor with LLM Decision-Making

<ex-supervisor-llm>
<python>
Use an LLM to dynamically decide which sub-agent to invoke.
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from typing import Literal

model = ChatOpenAI(model="gpt-4o")

# Define specialized sub-agents with their own tools
research_agent = create_react_agent(
    model,
    tools=[search_tool, arxiv_tool],
    prompt="You are a research specialist. Find relevant information.",
    name="researcher",
)

coding_agent = create_react_agent(
    model,
    tools=[code_executor, file_writer],
    prompt="You are a coding specialist. Write and test code.",
    name="coder",
)

# Supervisor uses LLM to route
def supervisor(state: MessagesState) -> Command[Literal["researcher", "coder", "__end__"]]:
    response = model.invoke([
        SystemMessage(content="""You are a team supervisor. Based on the conversation,
        decide which agent should act next. Respond with ONLY one of:
        - "researcher" if information gathering is needed
        - "coder" if code writing is needed
        - "FINISH" if the task is complete"""),
        *state["messages"],
    ])
    next_agent = response.content.strip().lower()
    if next_agent == "finish":
        return Command(goto=END)
    return Command(goto=next_agent)

graph = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor)
    .add_node("researcher", research_agent)  # Compiled agent as node
    .add_node("coder", coding_agent)
    .add_edge(START, "supervisor")
    .add_edge("researcher", "supervisor")
    .add_edge("coder", "supervisor")
    .compile()
)
```
</python>
</ex-supervisor-llm>

---

## Swarm / Agent Handoff Pattern

Agents transfer control directly to each other using `Command.PARENT` or handoff tools — no central supervisor required.

<ex-handoff-with-command-parent>
<python>
Sub-agent hands off to another agent in the parent graph using Command.PARENT.
```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated

@tool(return_direct=True)
def transfer_to_billing(
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the conversation to the billing specialist."""
    return Command(
        goto="billing_agent",
        update={"messages": [
            ToolMessage(
                content="Transferring to billing specialist...",
                name="transfer_to_billing",
                tool_call_id=tool_call_id,
            )
        ]},
        graph=Command.PARENT,  # Jump to parent graph's node
    )

@tool(return_direct=True)
def transfer_to_support(
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the conversation to technical support."""
    return Command(
        goto="support_agent",
        update={"messages": [
            ToolMessage(
                content="Transferring to technical support...",
                name="transfer_to_support",
                tool_call_id=tool_call_id,
            )
        ]},
        graph=Command.PARENT,
    )

# Each agent has its own tools + handoff tools
triage_agent = create_react_agent(
    model,
    tools=[transfer_to_billing, transfer_to_support],
    prompt="You are a triage agent. Route customers to the right specialist.",
    name="triage",
)

billing_agent = create_react_agent(
    model,
    tools=[lookup_invoice, process_refund, transfer_to_support],
    prompt="You are a billing specialist.",
    name="billing",
)

support_agent = create_react_agent(
    model,
    tools=[search_docs, create_ticket, transfer_to_billing],
    prompt="You are a technical support specialist.",
    name="support",
)

# Parent graph wires agents together
graph = (
    StateGraph(MessagesState)
    .add_node("triage_agent", triage_agent, destinations=("billing_agent", "support_agent"))
    .add_node("billing_agent", billing_agent, destinations=("support_agent",))
    .add_node("support_agent", support_agent, destinations=("billing_agent",))
    .add_edge(START, "triage_agent")
    .compile()
)
```
</python>
</ex-handoff-with-command-parent>

<handoff-key-concepts>

### Key Concepts for Handoffs

- **`Command.PARENT`**: Tells LangGraph to apply the `goto` and `update` to the **parent** graph, not the current subgraph
- **`destinations` parameter**: When adding a subgraph node, declare which parent nodes it can hand off to via `destinations=("node_a", "node_b")`
- **Handoff tools**: Tools that return `Command(graph=Command.PARENT, goto=...)` to transfer control
- **`InjectedToolCallId`**: Automatically injects the current tool_call_id so you can create proper `ToolMessage` responses

</handoff-key-concepts>

---

## Hierarchical Teams (Supervisor of Supervisors)

Nest supervisors to create multi-level agent hierarchies. Each level is a compiled subgraph.

<ex-hierarchical>
<python>
Build a hierarchical system where a top-level supervisor delegates to team leads.
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal, Annotated, TypedDict
import operator

class TeamState(TypedDict):
    messages: Annotated[list, operator.add]
    results: Annotated[list[str], operator.add]

# --- Research Team (Level 2) ---
def research_lead(state: TeamState) -> Command[Literal["web_searcher", "paper_reader", "__end__"]]:
    # Decide which researcher to use
    return Command(goto="web_searcher")

def web_searcher(state: TeamState) -> dict:
    return {"results": ["Web search results..."]}

def paper_reader(state: TeamState) -> dict:
    return {"results": ["Paper analysis..."]}

research_team = (
    StateGraph(TeamState)
    .add_node("research_lead", research_lead)
    .add_node("web_searcher", web_searcher)
    .add_node("paper_reader", paper_reader)
    .add_edge(START, "research_lead")
    .add_edge("web_searcher", "research_lead")
    .add_edge("paper_reader", "research_lead")
    .compile(name="research_team")
)

# --- Engineering Team (Level 2) ---
def eng_lead(state: TeamState) -> Command[Literal["coder", "tester", "__end__"]]:
    return Command(goto="coder")

def coder(state: TeamState) -> dict:
    return {"results": ["Code implementation..."]}

def tester(state: TeamState) -> dict:
    return {"results": ["Test results: all passed"]}

eng_team = (
    StateGraph(TeamState)
    .add_node("eng_lead", eng_lead)
    .add_node("coder", coder)
    .add_node("tester", tester)
    .add_edge(START, "eng_lead")
    .add_edge("coder", "eng_lead")
    .add_edge("tester", "eng_lead")
    .compile(name="eng_team")
)

# --- Top-Level Supervisor (Level 1) ---
def top_supervisor(state: TeamState) -> Command[Literal["research_team", "eng_team", "__end__"]]:
    if not state.get("results"):
        return Command(goto="research_team")
    return Command(goto="eng_team")

top_graph = (
    StateGraph(TeamState)
    .add_node("top_supervisor", top_supervisor)
    .add_node("research_team", research_team)  # Subgraph as node
    .add_node("eng_team", eng_team)            # Subgraph as node
    .add_edge(START, "top_supervisor")
    .add_edge("research_team", "top_supervisor")
    .add_edge("eng_team", "top_supervisor")
    .compile()
)
```
</python>
</ex-hierarchical>

---

## State Isolation Between Agents

Each sub-agent (subgraph) has its own private state. The parent graph only sees what the subgraph returns.

<state-isolation-strategies>

| Strategy | How | Use Case |
|----------|-----|----------|
| **Subgraph with different schema** | Subgraph uses its own `TypedDict` | Agent needs private scratchpad |
| **Subgraph checkpointer scoping** | `checkpointer=True/False/None` | Control persistence per agent |
| **Shared fields via overlap** | Parent and child share field names | Pass specific data between levels |

</state-isolation-strategies>

<ex-state-isolation>
<python>
Subgraph with private state — only shared fields pass between parent and child.
```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator

# Parent state — only sees messages and final_result
class ParentState(TypedDict):
    messages: Annotated[list, operator.add]
    final_result: str

# Child state — has private scratchpad fields
class ChildState(TypedDict):
    messages: Annotated[list, operator.add]  # Shared with parent (same key name)
    scratchpad: list[str]                     # Private to child
    intermediate: str                          # Private to child

def child_step_1(state: ChildState) -> dict:
    return {"scratchpad": ["step 1 notes"], "intermediate": "partial"}

def child_step_2(state: ChildState) -> dict:
    # Use private scratchpad, return result via shared field
    notes = state["scratchpad"]
    return {"messages": [f"Child result based on {len(notes)} notes"]}

child_graph = (
    StateGraph(ChildState)
    .add_node("step_1", child_step_1)
    .add_node("step_2", child_step_2)
    .add_edge(START, "step_1")
    .add_edge("step_1", "step_2")
    .add_edge("step_2", END)
    .compile()
)

# Parent graph — child's scratchpad/intermediate are invisible
parent_graph = (
    StateGraph(ParentState)
    .add_node("child_agent", child_graph)  # Subgraph as node
    .add_edge(START, "child_agent")
    .add_edge("child_agent", END)
    .compile()
)
```
</python>
</ex-state-isolation>

<state-overlap-rules>

### State Overlap Rules

- Fields with the **same key name** in parent and child schemas are automatically mapped
- Fields that exist **only in the child** schema are private to the child
- The child's return value is filtered to only include fields present in the parent schema
- Use this to create private scratchpads, intermediate results, or agent-specific context

</state-overlap-rules>

---

## Message Passing Between Agents

<message-passing-patterns>

| Pattern | Mechanism | When |
|---------|-----------|------|
| **Through shared state** | Both agents read/write `messages` field | Sequential agents |
| **Via Command update** | `Command(update={...}, goto="agent")` | Supervisor routing |
| **Via Send** | `Send("agent", {"messages": [...]})` | Fan-out to parallel agents |
| **Via Command.PARENT** | `Command(graph=Command.PARENT, update={...})` | Subgraph to parent |

</message-passing-patterns>

<ex-message-passing-send>
<python>
Fan out different tasks to multiple agents in parallel using Send.
```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator

class CoordinatorState(TypedDict):
    task: str
    agent_results: Annotated[list[dict], operator.add]

def coordinator(state: CoordinatorState):
    """Dispatch subtasks to specialized agents in parallel."""
    return [
        Send("analysis_agent", {"subtask": "Analyze requirements", "source": state["task"]}),
        Send("design_agent", {"subtask": "Create architecture", "source": state["task"]}),
        Send("risk_agent", {"subtask": "Assess risks", "source": state["task"]}),
    ]

def analysis_agent(state: dict) -> dict:
    return {"agent_results": [{"agent": "analysis", "output": f"Analysis of: {state['source']}"}]}

def design_agent(state: dict) -> dict:
    return {"agent_results": [{"agent": "design", "output": f"Design for: {state['source']}"}]}

def risk_agent(state: dict) -> dict:
    return {"agent_results": [{"agent": "risk", "output": f"Risks in: {state['source']}"}]}

def synthesizer(state: CoordinatorState) -> dict:
    summary = "; ".join(r["output"] for r in state["agent_results"])
    return {"task": f"Synthesized: {summary}"}

graph = (
    StateGraph(CoordinatorState)
    .add_node("analysis_agent", analysis_agent)
    .add_node("design_agent", design_agent)
    .add_node("risk_agent", risk_agent)
    .add_node("synthesizer", synthesizer)
    .add_conditional_edges(START, coordinator, ["analysis_agent", "design_agent", "risk_agent"])
    .add_edge("analysis_agent", "synthesizer")
    .add_edge("design_agent", "synthesizer")
    .add_edge("risk_agent", "synthesizer")
    .add_edge("synthesizer", END)
    .compile()
)
```
</python>
</ex-message-passing-send>

---

## Supervisor Monitoring and Intervention

The supervisor can inspect sub-agent state and intervene mid-execution.

<ex-supervisor-monitoring>
<python>
Supervisor monitors sub-agent progress via streaming and can interrupt.
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command, interrupt

checkpointer = InMemorySaver()

# Sub-agent that checks for supervisor intervention at key points
def worker_agent(state: MessagesState) -> dict:
    # Do some work...
    result = do_expensive_computation(state)

    # Check point: allow supervisor to review before continuing
    review = interrupt({
        "agent": "worker",
        "status": "checkpoint",
        "partial_result": result,
        "message": "Intermediate result ready for review",
    })

    if review.get("action") == "abort":
        return {"messages": ["[Worker] Task aborted by supervisor"]}
    elif review.get("action") == "redirect":
        return {"messages": [f"[Worker] Redirected: {review['instruction']}"]}

    return {"messages": [f"[Worker] Completed: {result}"]}

graph = (
    StateGraph(MessagesState)
    .add_node("worker", worker_agent)
    .add_edge(START, "worker")
    .add_edge("worker", END)
    .compile(checkpointer=checkpointer)
)

config = {"configurable": {"thread_id": "task-1"}}

# Start the worker — it pauses at the interrupt
result = graph.invoke({"messages": ["Process this data"]}, config)
# result["__interrupt__"] shows the worker's checkpoint data

# Supervisor reviews and decides to continue, abort, or redirect
# Option 1: Continue
result = graph.invoke(Command(resume={"action": "continue"}), config)

# Option 2: Abort
result = graph.invoke(Command(resume={"action": "abort"}), config)

# Option 3: Redirect with new instructions
result = graph.invoke(
    Command(resume={"action": "redirect", "instruction": "Focus on section 3 only"}),
    config,
)
```
</python>
</ex-supervisor-monitoring>

<ex-supervisor-state-injection>
<python>
Supervisor injects new instructions into a running sub-agent via update_state.
```python
config = {"configurable": {"thread_id": "task-1"}}

# Check current state of the sub-agent
state = graph.get_state(config)
print(f"Next node: {state.next}")
print(f"Current messages: {state.values['messages']}")

# Inject new instructions before resuming
graph.update_state(
    config,
    {"messages": ["[Supervisor] New priority: focus on error handling"]},
)

# Resume execution with the injected instructions
result = graph.invoke(None, config)
```
</python>
</ex-supervisor-state-injection>

---

## create_react_agent as Sub-Agent

`create_react_agent` creates a prebuilt ReAct agent that can be used standalone or as a node in a larger graph.

<create-react-agent-key-params>

| Parameter | Type | Purpose |
|-----------|------|---------|
| `model` | `str \| BaseChatModel \| Callable` | LLM to use (e.g. `"openai:gpt-4o"`) |
| `tools` | `list[BaseTool] \| ToolNode` | Tools available to this agent |
| `prompt` | `str \| SystemMessage \| Callable \| Runnable` | System prompt |
| `name` | `str` | Agent name (used in graph visualization) |
| `state_schema` | `TypedDict \| BaseModel` | Custom state (must include `messages`) |
| `checkpointer` | `BaseCheckpointSaver` | Persistence |
| `interrupt_before` | `list[str]` | Pause before these nodes (`"agent"`, `"tools"`) |
| `interrupt_after` | `list[str]` | Pause after these nodes |
| `response_format` | `type` | Structured output schema |
| `pre_model_hook` | `Callable \| Runnable` | Run before LLM call (trim messages, inject context) |
| `post_model_hook` | `Callable \| Runnable` | Run after LLM call (validate, approve) |

</create-react-agent-key-params>

<ex-create-react-agent-sub>
<python>
Use create_react_agent instances as sub-agents in a supervisor graph.
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# Each agent has its own tools and prompt
researcher = create_react_agent(
    model,
    tools=[web_search, arxiv_search],
    prompt="You are a research assistant. Find and summarize information.",
    name="researcher",
)

analyst = create_react_agent(
    model,
    tools=[data_loader, chart_generator],
    prompt="You are a data analyst. Analyze data and create visualizations.",
    name="analyst",
)

writer = create_react_agent(
    model,
    tools=[text_editor, grammar_check],
    prompt="You are a technical writer. Write clear documentation.",
    name="writer",
)

# Wire into supervisor graph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from typing import Literal

def supervisor(state: MessagesState) -> Command[Literal["researcher", "analyst", "writer", "__end__"]]:
    # Use LLM to decide next agent (simplified here)
    return Command(goto="researcher")

graph = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor)
    .add_node("researcher", researcher)
    .add_node("analyst", analyst)
    .add_node("writer", writer)
    .add_edge(START, "supervisor")
    .add_edge("researcher", "supervisor")
    .add_edge("analyst", "supervisor")
    .add_edge("writer", "supervisor")
    .compile()
)
```
</python>
</ex-create-react-agent-sub>

<ex-prompt-patterns>
<python>
Different ways to configure agent prompts.
```python
from langchain_core.messages import SystemMessage

# 1. Simple string — becomes SystemMessage prepended to messages
agent1 = create_react_agent(model, tools, prompt="You are a helpful assistant.")

# 2. SystemMessage — same effect, more explicit
agent2 = create_react_agent(model, tools, prompt=SystemMessage(content="You are a helpful assistant."))

# 3. Callable — dynamic prompt based on state
def dynamic_prompt(state):
    user_name = state.get("user_name", "User")
    return [
        SystemMessage(content=f"You are helping {user_name}. Be concise."),
        *state["messages"],
    ]
agent3 = create_react_agent(model, tools, prompt=dynamic_prompt)

# 4. With custom state — access additional fields in prompt
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    remaining_steps: int
    context: str  # Custom field

def context_prompt(state: AgentState):
    return [
        SystemMessage(content=f"Context: {state['context']}"),
        *state["messages"],
    ]
agent4 = create_react_agent(model, tools, prompt=context_prompt, state_schema=AgentState)
```
</python>
</ex-prompt-patterns>

---

## Common Fixes

<fix-subgraph-destinations>
<python>
Declare destinations when using Command.PARENT in subgraph nodes.
```python
# WRONG: Parent graph doesn't know about handoff targets
graph.add_node("agent_a", agent_a_subgraph)

# CORRECT: Declare valid handoff destinations
graph.add_node("agent_a", agent_a_subgraph, destinations=("agent_b", "agent_c"))
```
</python>
</fix-subgraph-destinations>

<fix-reducer-for-multi-agent>
<python>
Always use a reducer on fields that multiple agents write to.
```python
# WRONG: Last agent's result overwrites previous
class State(TypedDict):
    results: list  # No reducer!

# CORRECT: Accumulate results from all agents
class State(TypedDict):
    results: Annotated[list, operator.add]
```
</python>
</fix-reducer-for-multi-agent>

<fix-supervisor-loop>
<python>
Ensure supervisor has a path to END to avoid infinite loops.
```python
# WRONG: Supervisor always routes to an agent, never ends
def supervisor(state) -> Command[Literal["agent_a", "agent_b"]]:
    return Command(goto="agent_a")

# CORRECT: Include END as a possible destination
def supervisor(state) -> Command[Literal["agent_a", "agent_b", "__end__"]]:
    if task_complete(state):
        return Command(goto=END)
    return Command(goto="agent_a")
```
</python>
</fix-supervisor-loop>

<boundaries>
### What You Should NOT Do

- Build multi-agent without a clear routing strategy — define how agents are selected
- Use the same stateful subgraph (`checkpointer=True`) in parallel — namespace conflicts
- Forget `destinations` param when using `Command.PARENT` handoffs — parent won't know where to route
- Skip reducers on shared accumulator fields — last agent's write overwrites all previous
- Create circular handoffs without a termination condition — infinite loops
</boundaries>

---

<deepwiki-tips>

### Need More Details?

Use DeepWiki MCP (`mcp__deepwiki__ask_question`) with `repoName: "langchain-ai/langgraph"` to query:

- **"How does create_react_agent handle tool calling loops internally?"** — understand the internal ReAct cycle
- **"How to use Send with Command for parallel agent dispatch?"** — advanced fan-out patterns
- **"How does checkpoint_ns work for subgraph namespace isolation?"** — deep dive into namespace scoping
- **"How to stream events from subgraphs with subgraphs=True?"** — monitoring sub-agent execution in real-time
- **"How does ToolNode handle parallel tool calls?"** — parallel tool execution within an agent
- **"How to use pre_model_hook and post_model_hook in create_react_agent?"** — advanced agent lifecycle hooks

</deepwiki-tips>
