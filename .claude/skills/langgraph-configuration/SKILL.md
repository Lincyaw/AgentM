---
name: langgraph-configuration
description: "INVOKE THIS SKILL when configuring LangGraph agents at runtime — tools, prompts, models, and context via config files. Covers RunnableConfig, context_schema, Runtime, dynamic model/tool selection, and config-driven agent setup."
---

<overview>
LangGraph provides multiple layers of runtime configuration:

- **RunnableConfig**: Pass `thread_id`, tags, metadata, and custom configurable values at invocation time
- **context_schema / Runtime**: Type-safe runtime context for dynamic model/tool/prompt selection
- **Prompt configuration**: String, SystemMessage, Callable, or Runnable prompts per agent
- **Tool configuration**: Per-agent tool lists, dynamic tool selection
- **Store**: Cross-thread persistent configuration (user preferences, settings)
</overview>

<config-layers>

| Layer | Scope | Persistence | Use Case |
|-------|-------|-------------|----------|
| `RunnableConfig` | Per invocation | No | thread_id, tags, metadata |
| `context_schema` / `Runtime` | Per invocation | No | User context, model selection |
| `prompt` parameter | Per agent (compile time) | Yes | Agent personality, system instructions |
| `Store` | Cross-thread | Yes | User preferences, long-term settings |
| Config file (custom) | Application-level | Yes | Tool registry, prompt templates |

</config-layers>

---

## RunnableConfig

Pass configuration at every `invoke()` or `stream()` call.

<ex-runnable-config>
<python>
Pass runtime configuration including thread_id, metadata, and custom values.
```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "configurable": {
        "thread_id": "session-123",       # Required for persistence
        "user_id": "user-456",            # Custom configurable value
        "model_name": "gpt-4o",           # Custom: select model at runtime
    },
    "tags": ["production", "team-alpha"],  # For tracing/filtering
    "metadata": {"source": "api", "version": "2.0"},  # For observability
}

result = graph.invoke({"messages": ["Hello"]}, config)

# Access config inside a node
def my_node(state, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    user_id = config["configurable"].get("user_id", "anonymous")
    return {"response": f"Hello {user_id}"}
```
</python>
</ex-runnable-config>

---

## context_schema and Runtime

Type-safe runtime context — the recommended approach for passing structured configuration to nodes.

<ex-context-schema>
<python>
Define a typed context schema and access it via Runtime in nodes.
```python
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing import Annotated, TypedDict
import operator

@dataclass
class AgentContext:
    user_id: str
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    language: str = "en"

class State(TypedDict):
    messages: Annotated[list, operator.add]
    response: str

def agent_node(state: State, runtime: Runtime[AgentContext]) -> dict:
    """Node that uses typed runtime context."""
    user_id = runtime.context.user_id
    model_name = runtime.context.model_name
    lang = runtime.context.language

    # Use context to configure behavior
    if lang == "zh":
        system_msg = f"You are helping user {user_id}. Respond in Chinese."
    else:
        system_msg = f"You are helping user {user_id}. Respond in English."

    return {"response": f"Configured for {model_name}, user {user_id}"}

graph = (
    StateGraph(State, context_schema=AgentContext)
    .add_node("agent", agent_node)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile()
)

# Pass context at invocation time
result = graph.invoke(
    {"messages": ["Hello"]},
    config={
        "configurable": {"thread_id": "t1"},
    },
    context=AgentContext(user_id="alice", model_name="gpt-4o", language="zh"),
)
```
</python>
</ex-context-schema>

---

## Dynamic Model Selection

Select different models per invocation or per agent based on runtime context.

<ex-dynamic-model>
<python>
Dynamically select models based on runtime context.
```python
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime

@dataclass
class ModelContext:
    model_name: str = "gpt-4o-mini"

# Pre-instantiate models (avoid creating new instances per request)
MODELS = {
    "gpt-4o": ChatOpenAI(model="gpt-4o"),
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
    "gpt-4.1": ChatOpenAI(model="gpt-4.1"),
}

def select_model(state, runtime: Runtime[ModelContext]):
    """Dynamically select model based on runtime context."""
    model_name = runtime.context.model_name
    model = MODELS.get(model_name, MODELS["gpt-4o-mini"])
    return model.bind_tools(tools)

agent = create_react_agent(
    model=select_model,  # Callable for dynamic selection
    tools=tools,
    context_schema=ModelContext,
)

# Use fast model for simple queries
result = agent.invoke(
    {"messages": ["What's 2+2?"]},
    context=ModelContext(model_name="gpt-4o-mini"),
)

# Use powerful model for complex queries
result = agent.invoke(
    {"messages": ["Analyze this codebase architecture..."]},
    context=ModelContext(model_name="gpt-4o"),
)
```
</python>
</ex-dynamic-model>

---

## Config-File-Driven Agent Setup

Build a configuration-driven system where agent tools, prompts, and models are defined in config files.

<ex-config-file-driven>
<python>
Load agent configuration from YAML/JSON files for declarative setup.
```python
import yaml
from dataclasses import dataclass, field
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI

# --- config/agents.yaml ---
# orchestrator:
#   model: gpt-4o
#   prompt: "You are a team lead coordinating agents."
#   tools: [assign_task, check_status, terminate_agent]
#
# researcher:
#   model: gpt-4o-mini
#   prompt: "You research topics thoroughly."
#   tools: [web_search, arxiv_search]
#
# coder:
#   model: gpt-4o
#   prompt: "You write clean Python code."
#   tools: [code_executor, file_writer, linter]

@dataclass
class AgentConfig:
    model: str
    prompt: str
    tools: list[str]
    interrupt_before: list[str] = field(default_factory=list)

def load_agent_configs(config_path: str) -> dict[str, AgentConfig]:
    """Load agent configurations from YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return {
        name: AgentConfig(**config)
        for name, config in raw.items()
    }

# Tool registry — maps tool names to implementations
TOOL_REGISTRY: dict[str, callable] = {
    "web_search": web_search_tool,
    "arxiv_search": arxiv_search_tool,
    "code_executor": code_executor_tool,
    "file_writer": file_writer_tool,
    "linter": linter_tool,
    "assign_task": assign_task_tool,
    "check_status": check_status_tool,
    "terminate_agent": terminate_agent_tool,
}

def build_agent_from_config(name: str, config: AgentConfig):
    """Build a create_react_agent from config."""
    model = ChatOpenAI(model=config.model)
    tools = [TOOL_REGISTRY[t] for t in config.tools]

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=config.prompt,
        name=name,
        interrupt_before=config.interrupt_before or None,
    )

def build_graph_from_configs(config_path: str):
    """Build complete multi-agent graph from config file."""
    configs = load_agent_configs(config_path)

    builder = StateGraph(MessagesState)

    # Create each agent from config
    agents = {}
    for name, config in configs.items():
        agent = build_agent_from_config(name, config)
        agents[name] = agent
        builder.add_node(name, agent)

    # Wire supervisor pattern (orchestrator routes to others)
    builder.add_edge(START, "orchestrator")
    for name in configs:
        if name != "orchestrator":
            builder.add_edge(name, "orchestrator")

    return builder.compile()

# Usage
graph = build_graph_from_configs("config/agents.yaml")
```
</python>
</ex-config-file-driven>

---

## Per-Agent Prompt Configuration

<prompt-config-patterns>

| Prompt Type | Syntax | Best For |
|------------|--------|----------|
| Static string | `prompt="You are..."` | Fixed personality |
| SystemMessage | `prompt=SystemMessage(...)` | Explicit message type |
| Callable | `prompt=lambda state: [...]` | Dynamic based on state |
| Runnable | `prompt=ChatPromptTemplate(...)` | Template with variables |

</prompt-config-patterns>

<ex-prompt-from-config>
<python>
Load prompts from external files or templates.
```python
from pathlib import Path
from langchain_core.messages import SystemMessage

def load_prompt(prompt_path: str, **kwargs) -> str:
    """Load prompt template from file and format with kwargs."""
    template = Path(prompt_path).read_text()
    return template.format(**kwargs)

# Prompt template file: prompts/researcher.txt
# "You are a research assistant for {project_name}.
#  Focus area: {focus_area}.
#  Output format: {output_format}."

researcher = create_react_agent(
    model,
    tools=research_tools,
    prompt=load_prompt(
        "prompts/researcher.txt",
        project_name="AgentM",
        focus_area="LLM architectures",
        output_format="bullet points",
    ),
)

# Dynamic prompt that reads from config at runtime
def configurable_prompt(state):
    config = state.get("agent_config", {})
    system = config.get("system_prompt", "You are a helpful assistant.")
    constraints = config.get("constraints", "")
    return [
        SystemMessage(content=f"{system}\n\nConstraints: {constraints}"),
        *state["messages"],
    ]

agent = create_react_agent(model, tools, prompt=configurable_prompt)
```
</python>
</ex-prompt-from-config>

---

## Runtime Store for Persistent Configuration

<ex-store-config>
<python>
Use Store to persist agent configuration across threads.
```python
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

store = InMemoryStore()

# Save agent configurations to store (done once, persists)
store.put(("config", "agents"), "researcher", {
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "max_results": 10,
})
store.put(("config", "agents"), "coder", {
    "model": "gpt-4o",
    "temperature": 0.0,
    "style": "concise",
})

# Node reads config from store at runtime
def configurable_agent(state, runtime: Runtime):
    agent_name = state.get("current_agent", "researcher")
    config = runtime.store.get(("config", "agents"), agent_name)

    if config:
        model_name = config.value["model"]
        temp = config.value["temperature"]
        # Use config to customize behavior...

    return {"result": f"Running {agent_name} with {model_name}"}

graph = builder.compile(checkpointer=checkpointer, store=store)
```
</python>
</ex-store-config>

---

## Common Fixes

<fix-config-not-in-node>
<python>
Access config via function parameter, not global scope.
```python
# WRONG: Config not available
def my_node(state):
    thread_id = config["configurable"]["thread_id"]  # NameError!

# CORRECT: Request config as parameter
from langchain_core.runnables import RunnableConfig

def my_node(state, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]

# CORRECT: Use Runtime for typed context
from langgraph.runtime import Runtime

def my_node(state, runtime: Runtime[MyContext]):
    user_id = runtime.context.user_id
```
</python>
</fix-config-not-in-node>

<fix-context-vs-config>
<python>
Use context_schema for structured data, RunnableConfig for infrastructure values.
```python
# context_schema — for business logic (user preferences, feature flags)
@dataclass
class AppContext:
    user_id: str
    premium_user: bool = False

# RunnableConfig — for infrastructure (thread_id, tracing)
config = {"configurable": {"thread_id": "t1"}, "tags": ["prod"]}

# Pass both
result = graph.invoke(
    {"messages": ["Hello"]},
    config=config,
    context=AppContext(user_id="alice", premium_user=True),
)
```
</python>
</fix-context-vs-config>

<boundaries>
### What You Should NOT Do

- Hardcode model names in agent nodes — use config or context_schema for flexibility
- Access store directly in nodes — use `runtime.store` via Runtime parameter
- Put infrastructure concerns (thread_id) in context_schema — use RunnableConfig for those
- Create model instances inside nodes — pre-instantiate and select via config
- Modify config during execution — config is read-only per invocation
</boundaries>

---

<deepwiki-tips>

### Need More Details?

Use DeepWiki MCP (`mcp__deepwiki__ask_question`) with `repoName: "langchain-ai/langgraph"` to query:

- **"How does context_schema replace the deprecated config_schema?"** — migration guide
- **"How does Runtime inject store, config, and context into nodes?"** — Runtime internals
- **"How to use InjectedStore in tool functions?"** — tools accessing persistent config
- **"How does create_react_agent handle the model parameter when it's a callable?"** — dynamic model selection internals
- **"How to use pre_model_hook for message trimming based on config?"** — context window management

</deepwiki-tips>
