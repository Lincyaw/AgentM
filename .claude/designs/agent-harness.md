# Design: Agent Harness SDK

**Status**: PROPOSAL
**Created**: 2026-03-26

---

## 1. Background & Motivation

### 1.1 What Prompted This

A reverse-engineering analysis of Claude Code's multi-agent architecture (see [reference doc](../../docs/references/claude-code-agent-team-architecture.md)) revealed a key architectural insight:

> **The SDK is a Harness — it controls agent conversation loops, manages lifecycle, and routes messages. The LLM is a stateless function being called.**

Claude Code implements this with no graph framework at all — just Node.js async loops and file-based message passing. This forced us to re-examine whether LangGraph is the right foundation for AgentM's SDK layer.

### 1.2 The LangGraph Question

Analysis of how AgentM actually uses LangGraph:

| What LangGraph provides | How AgentM uses it |
|------------------------|-------------------|
| StateGraph (nodes, edges) | Orchestrator: 3-node graph. Worker: 4-node graph. |
| Multi-agent subgraph composition | **Not used.** Workers are independent `asyncio.Task`s, not graph nodes. |
| State mapping between parent/child graphs | **Not used.** Communication via tools only. |
| `create_react_agent` | **Not used in current node-based architecture.** Custom graph builders instead. |
| Checkpointing | Used for state persistence and recovery. ✅ |
| Streaming (`astream`) | Used for event capture in `TaskManager._execute_agent`. ✅ |
| `pre_model_hook` | **Not used directly.** Custom `NodePipeline.before()` replaces it. |
| TypedDict state schema | Used, but adds rigidity (compile-time fixed). |

**Conclusion**: AgentM uses LangGraph primarily as a **single-agent loop engine** (StateGraph with 3-4 nodes) and for **checkpointing + streaming**. The multi-agent orchestration is entirely custom code (`TaskManager`, `AgentPool`, orchestrator tools).

### 1.3 Current Architecture's Implicit Assumptions

Several design decisions that should be scenario-level are currently embedded in the SDK core:

| Assumption | Where it lives | Should be |
|-----------|---------------|-----------|
| Workers are data-only (no reasoning) | `sub-agent.md` design principle | RCA scenario choice |
| Orchestrator controls all reasoning | `orchestrator.md` core design | RCA scenario choice |
| Communication is unidirectional (Orchestrator → Worker) | `TaskManager` API (inject only) | SDK should support bidirectional |
| Agent pool is static (compiled at startup) | `AgentPool.__init__()` compiles all agents | SDK should allow dynamic spawn |
| State schema is TypedDict (LangGraph requirement) | `state_schema` parameter everywhere | SDK should be schema-flexible |
| Phase definitions must exist | `ReasoningStrategy.phase_definitions()` | Some scenarios have no phases |

### 1.4 Design Goal

Build a minimal SDK (the "Harness") that provides:

1. **Agent loop execution** — run a single agent's conversation loop
2. **Agent lifecycle management** — spawn, monitor, abort agents
3. **Inter-agent communication** — bidirectional message passing
4. **Middleware pipeline** — extensible hooks around the agent loop
5. **Optional checkpointing** — persist and recover agent state

Scenarios (RCA, trajectory analysis, general purpose) build on top of this SDK using its primitives. LangGraph becomes an optional backend for the agent loop, not a structural dependency.

---

## 2. Design Philosophy

### 2.1 SDK = Harness

The Harness (SDK) is the actual running process. It:
- Maintains conversation history per agent
- Calls the LLM API with accumulated messages
- Intercepts tool calls and executes them locally
- Injects external messages (from other agents, from users) into the conversation
- Decides when to stop the loop

The LLM is **not a process**. It is a stateless function: `f(messages, tools) → response`. Every "agent" is just an independent conversation loop managed by the Harness.

### 2.2 SDK vs Scenario Separation

The SDK provides **capabilities**. Scenarios choose which capabilities to use.

| SDK provides | Scenarios decide |
|-------------|-----------------|
| Agent loop execution | What the agent's prompt says |
| Tool dispatch mechanism | Which tools to give the agent |
| Bidirectional message channel | Whether workers send back or not |
| Dynamic agent spawning | Whether pool is static or dynamic |
| Middleware hooks | Which middleware to wire (compression, budget, trajectory...) |
| Checkpoint store interface | Whether to checkpoint and how |

The SDK never mandates:
- That workers can't reason (scenario choice)
- That communication must be unidirectional (scenario choice)
- That phases must exist (scenario choice)
- That state must be a TypedDict (scenario choice)

### 2.3 Message Injection as First-Class Primitive

Any external source should be able to inject a message into any agent's conversation loop:
- Orchestrator injects instructions into worker (current `inject_instruction`)
- Worker reports findings back to orchestrator (new capability)
- External system pushes alerts into running agent (future)
- User intervenes mid-execution (future)

This is implemented as a simple inbox queue per agent, drained before each LLM call.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      AgentRuntime (Harness)                    │
│                                                                │
│  Lifecycle: spawn / abort / wait / status                      │
│  Messaging: send(to, message) — routed via agent inbox queues  │
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ AgentLoop A       │  │ AgentLoop B       │   ...            │
│  │ (orchestrator)    │  │ (worker)          │                   │
│  │                   │  │                   │                   │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                   │
│  │ │ Middleware[]  │ │  │ │ Middleware[]  │ │                   │
│  │ └──────────────┘ │  │ └──────────────┘ │                   │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                   │
│  │ │ inbox: Queue  │ │  │ │ inbox: Queue  │ │                   │
│  │ └──────────────┘ │  │ └──────────────┘ │                   │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                   │
│  │ │ Backend:     │ │  │ │ Backend:     │ │                   │
│  │ │ Simple | LG  │ │  │ │ Simple | LG  │ │                   │
│  │ └──────────────┘ │  │ └──────────────┘ │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                │
│  ┌──────────────────────────────────────┐                     │
│  │ CheckpointStore (optional)            │                     │
│  │ impl: Memory | SQLite | PostgreSQL    │                     │
│  └──────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
                    LLM API (remote, stateless)
```

### Comparison with Current Architecture

```
Current:                              Proposed:

AgentSystemBuilder                    AgentRuntime
  ├─ AgentPool (static, compiled)       ├─ spawn() (dynamic, on-demand)
  ├─ TaskManager                        ├─ send() / wait() / abort()
  │    ├─ submit()                      └─ (TaskManager subsumed)
  │    ├─ get_all_status()
  │    ├─ inject()                    AgentLoop (Protocol)
  │    └─ abort()                       ├─ SimpleAgentLoop (new)
  ├─ create_node_orchestrator()         └─ LangGraphAgentLoop (wraps current)
  └─ build_worker_subgraph()
                                      ReasoningStrategy (unchanged)
ReasoningStrategy                       └─ Scenarios build on AgentRuntime
  └─ Scenario-specific logic              using tools that call runtime methods
```

---

## 4. Core Interfaces

> **Design constraint**: Every method here was stress-tested against 11 scenarios
> (structured output, human-in-the-loop, cascading abort, rate limiting,
> intermediate results, agent restart, etc.) — see the "Verified not needed"
> table at the end of this section.

### 4.1 AgentLoop — Single Agent Conversation Loop

```python
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class AgentLoop(Protocol):
    """Protocol for a single agent's conversation loop.

    An AgentLoop encapsulates the cycle:
        receive input → call LLM → execute tools → repeat → return result

    Implementations may use LangGraph, a simple while-loop, or any other
    mechanism. The SDK does not care — it only interacts via this interface.

    stream() is the primary method; run() is a convenience wrapper that
    iterates the stream and returns the final result.
    """

    async def run(
        self, input: str, *, config: RunConfig | None = None
    ) -> AgentResult:
        """Run the agent loop to completion.

        Default implementation iterates stream() and extracts the result
        from the final "complete" event. Implementations may override for
        efficiency.
        """
        ...

    async def stream(
        self, input: str, *, config: RunConfig | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding events as they occur.

        This is the primary execution method. The runtime always uses
        stream() internally to capture events for EventHandler forwarding.

        The last event MUST be type="complete" with data={"result": AgentResult}.

        Yields AgentEvent instances for: llm_start, llm_end, tool_start,
        tool_end, inject, error, complete.
        """
        ...

    def inject(self, message: str) -> None:
        """Inject a message into the agent's inbox.

        The message will be consumed before the next LLM call.
        Can be called concurrently while the agent loop is running
        (safe under asyncio single-thread model).
        """
        ...
```

**Why `stream()` is primary**: The runtime needs events for `EventHandler` forwarding (trajectory recording, WebSocket streaming, debug console). If `run()` were primary, events would be lost. Making `stream()` primary means `run()` is trivially derived:

```python
async def run(self, input, *, config=None):
    result = None
    async for event in self.stream(input, config=config):
        if event.type == "complete":
            result = event.data.get("result")
    return result
```

### 4.2 AgentRuntime — The Harness

```python
class AgentRuntime:
    """Manages multiple AgentLoops: lifecycle, messaging, coordination.

    This is the Harness. It replaces TaskManager + AgentPool with a
    unified interface for dynamic agent management.
    """

    def __init__(
        self,
        *,
        checkpoint_store: CheckpointStore | None = None,
        event_handler: EventHandler | None = None,
    ) -> None: ...

    # --- Lifecycle ---

    async def spawn(
        self,
        agent_id: str,
        *,
        loop: AgentLoop,
        input: str,
        parent_id: str | None = None,
        config: RunConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentHandle:
        """Spawn a new agent and start its loop as an asyncio.Task.

        Args:
            agent_id: Unique identifier for this agent instance.
            loop: The AgentLoop implementation to run.
            input: Initial instruction for the agent.
            parent_id: If set, this agent is a child. When the parent
                terminates (completed/failed/aborted), all running
                children are automatically aborted.
            config: Execution config (timeout, max_steps, thread_id).
            metadata: Arbitrary metadata (scenario-specific, e.g. hypothesis_id).

        Returns:
            AgentHandle for monitoring and interaction.

        Internally, the runtime iterates loop.stream() to capture events
        and forward them to the EventHandler.
        """
        ...

    async def abort(self, agent_id: str, reason: str) -> bool:
        """Abort a running agent. Returns False if already stopped.

        Cascades: also aborts all running children (agents whose
        parent_id == agent_id), recursively.
        """
        ...

    async def wait(
        self, agent_id: str, *, timeout: float | None = None
    ) -> AgentResult:
        """Block until an agent completes. Raises TimeoutError if exceeded."""
        ...

    async def wait_any(
        self,
        agent_ids: list[str] | None = None,
        *,
        timeout: float | None = None,
    ) -> list[str]:
        """Wait for any agent to complete. Returns newly completed agent_ids.

        Args:
            agent_ids: If given, only watch these agents. None = all.

        Uses asyncio.Event (not polling) for precise wake-up.
        """
        ...

    # --- Communication ---

    async def send(self, to: str, message: str) -> None:
        """Send a message to a running agent's inbox.

        Calls agent.loop.inject(message) internally.
        Raises ValueError if agent_id not found or agent not running.
        """
        ...

    # --- Status ---

    def get_status(self) -> dict[str, AgentInfo]:
        """Snapshot of all agents: status, duration, step count, result."""
        ...

    def get_result(self, agent_id: str) -> AgentResult | None:
        """Get result of a completed agent, or None if still running."""
        ...

    def get_running_ids(self) -> list[str]:
        """List agent_ids of currently running agents."""
        ...
```

### 4.3 Data Types

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RunConfig:
    """Per-run execution configuration."""
    max_steps: int | None = None
    timeout: float | None = None
    thread_id: str | None = None       # for checkpointing
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Outcome of an agent loop execution."""
    agent_id: str
    status: AgentStatus
    output: Any = None                  # final response (str, dict, Pydantic model)
    error: str | None = None
    duration_seconds: float | None = None
    steps: int = 0                      # total LLM call rounds
    tool_calls: int = 0                 # total tool invocations
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEvent:
    """Streaming event from an agent loop."""
    type: str                           # llm_start, llm_end, tool_start, tool_end,
                                        # inject, complete, error
    agent_id: str
    data: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    timestamp: str = ""


@dataclass
class AgentInfo:
    """Runtime status snapshot of an agent."""
    agent_id: str
    status: AgentStatus
    parent_id: str | None = None
    current_step: int = 0
    started_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result: AgentResult | None = None


class AgentHandle:
    """Reference to a running agent within the runtime.

    Convenience wrapper — all operations delegate to AgentRuntime.
    """

    def __init__(self, runtime: AgentRuntime, agent_id: str) -> None:
        self._runtime = runtime
        self.agent_id = agent_id

    @property
    def status(self) -> AgentStatus: ...

    @property
    def result(self) -> AgentResult | None: ...

    async def wait(self, *, timeout: float | None = None) -> AgentResult:
        return await self._runtime.wait(self.agent_id, timeout=timeout)

    async def send(self, message: str) -> None:
        await self._runtime.send(self.agent_id, message)

    async def abort(self, reason: str) -> bool:
        return await self._runtime.abort(self.agent_id, reason)
```

### 4.4 Middleware

```python
from collections.abc import Awaitable, Callable


class Middleware(Protocol):
    """Hook into the agent loop at defined points.

    All methods have default pass-through behavior. Implementations
    override only the hooks they need.

    Design: 3 hooks total.
    - on_llm_start / on_llm_end: chain in order (output of one → input of next)
    - on_tool_call: wrapping pattern (call call_next to proceed, or return
      directly to short-circuit with a cached/intercepted result)
    """

    async def on_llm_start(
        self, messages: list[Any], ctx: LoopContext
    ) -> list[Any]:
        """Called before each LLM invocation.

        May modify, filter, or append to the message list.
        Return the (possibly modified) messages.
        """
        return messages

    async def on_llm_end(
        self, response: Any, ctx: LoopContext
    ) -> Any:
        """Called after each LLM invocation.

        May inspect or modify the response.
        """
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        call_next: Callable[[str, dict], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        """Wrap a single tool execution.

        Call ``await call_next(tool_name, tool_args)`` to proceed to the
        next middleware (or the actual tool). Return the result string.

        Capabilities:
        - Pass through:     ``return await call_next(name, args)``
        - Short-circuit:    ``return cached_result`` (skip execution)
        - Modify args:      ``return await call_next(name, new_args)``
        - Transform output: ``r = await call_next(name, args); return r[:1000]``
        - Record before/after: wrap the call_next with logging
        """
        return await call_next(tool_name, tool_args)


@dataclass
class LoopContext:
    """Read-only context available to middleware during a loop iteration."""
    agent_id: str
    step: int
    max_steps: int | None
    tool_call_count: int
    metadata: dict[str, Any]
```

**Why wrapping instead of start/end hooks for tools:**

The original design had `on_tool_start` + `on_tool_end`. This breaks for `DedupMiddleware` — it needs to return a cached result without calling the actual tool, but skipping the tool leaves no `ToolMessage` in the conversation. The wrapping pattern (`call_next`) handles all cases uniformly:

```python
# DedupMiddleware — short-circuit with cache
async def on_tool_call(self, name, args, call_next, ctx):
    if (key := self._key(name, args)) in self._cache:
        return self._cache[key]           # skip tool, return cached
    result = await call_next(name, args)
    self._cache[key] = result
    return result

# TrajectoryMiddleware — record before + after
async def on_tool_call(self, name, args, call_next, ctx):
    await self._record("tool_start", name=name, args=args)
    result = await call_next(name, args)
    await self._record("tool_end", name=name, result=result)
    return result

# ToolOutputMiddleware — truncate long results
async def on_tool_call(self, name, args, call_next, ctx):
    result = await call_next(name, args)
    return result[:self._limit] + "…" if len(result) > self._limit else result
```

**Middleware composition**: `on_llm_start/end` chain sequentially (output → input). `on_tool_call` nests (each middleware's `call_next` invokes the next middleware):

```python
def _compose_tool_middleware(middlewares, actual_call):
    """Build a nested call chain for on_tool_call."""
    chain = actual_call
    for mw in reversed(middlewares):
        prev = chain
        chain = lambda name, args, _mw=mw, _prev=prev: (
            _mw.on_tool_call(name, args, _prev, ctx)
        )
    return chain
```

### 4.5 Verified Not Needed

These capabilities were evaluated and determined to NOT require interface changes:

| Scenario | How it works without interface changes |
|----------|---------------------------------------|
| **Structured output** | `SimpleAgentLoop(output_schema=MyModel)` constructor param. After ReAct loop ends, an extra LLM call via `model.with_structured_output(schema)` produces a Pydantic model as `AgentResult.output`. Protocol unchanged — `output: Any` accommodates it. |
| **Shared state between agents** | Tools + closures pattern. Scenario creates a shared object (e.g. `ServiceProfileStore`), wraps it in tools, injects those tools into both orchestrator and worker loops. SDK doesn't know about it. See Section 6.4. |
| **Human-in-the-loop** | Agent calls a blocking `ask_user` tool that `await`s user input. Or use `LangGraphAgentLoop` with `interrupt_before`. Loop protocol unchanged. |
| **Rate limiting** | `AgentRuntime` uses an internal `asyncio.Semaphore` in `spawn()`. No new API. |
| **Agent accesses runtime** | Tools capture `runtime` reference via closure (Section 6.1). Loop and Middleware don't need runtime reference. |
| **Intermediate results** | Worker calls `report_finding` tool → `runtime.send(orchestrator_id, msg)`. Already supported. |
| **Agent restart** | Caller creates a new `AgentLoop` via factory and calls `runtime.spawn()` again. No "restart" primitive needed. |
| **Event filtering** | `EventHandler.on_event()` receives all events; caller filters by `event.agent_id`. No `subscribe()` needed. |
| **Concurrent tool calls** | `SimpleAgentLoop` executes tool calls sequentially. Parallel tool execution is a loop implementation detail, not a protocol concern. |
| **Multiple LLM providers** | Loop constructor takes `model: ChatModel` (Protocol: `ainvoke(messages) -> response`). Any LangChain model or custom wrapper satisfies it. |
| **Agent-to-agent peer messaging** | Workers use `runtime.send(other_worker_id, msg)` via a tool. Runtime already supports any-to-any messaging. No scenario uses it yet. |
| **Graceful stop (vs abort)** | Inject a "please wrap up" message via `runtime.send()`. The agent decides to finish. Abort is for forced termination. |

### 4.5 CheckpointStore

```python
class CheckpointStore(Protocol):
    """Persist and recover agent conversation state.

    Implementations: MemoryCheckpointStore, SQLiteCheckpointStore,
    or LangGraphCheckpointAdapter (wraps LangGraph's BaseCheckpointSaver).
    """

    async def save(
        self, agent_id: str, state: dict[str, Any]
    ) -> str:
        """Save a state snapshot. Returns checkpoint_id."""
        ...

    async def load(
        self, agent_id: str, *, checkpoint_id: str | None = None
    ) -> dict[str, Any] | None:
        """Load state. None checkpoint_id = latest. Returns None if not found."""
        ...

    async def list_checkpoints(
        self, agent_id: str
    ) -> list[dict[str, Any]]:
        """List available checkpoints for an agent (newest first)."""
        ...
```

### 4.6 EventHandler

```python
class EventHandler(Protocol):
    """Receive streaming events from all agents in the runtime.

    Implementations: TrajectoryCollector adapter, WebSocket broadcaster,
    debug console renderer.
    """

    async def on_event(self, event: AgentEvent) -> None:
        """Called for every event emitted by any agent in the runtime."""
        ...
```

---

## 5. Agent Loop Implementations

### 5.1 SimpleAgentLoop

A lightweight ReAct loop with no external framework dependency.

```python
class SimpleAgentLoop:
    """Pure-Python agent loop. No LangGraph dependency.

    Implements the ReAct cycle: LLM → tool calls → LLM → ... → final answer.
    Middleware hooks fire at each stage. Inbox is drained before each LLM call.

    stream() is the primary method. run() delegates to it.
    """

    def __init__(
        self,
        *,
        model: ChatModel,                      # Protocol: async ainvoke(messages) -> response
        tools: list[Tool],
        system_prompt: str,
        middleware: list[Middleware] | None = None,
        output_schema: type[BaseModel] | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self._model = model
        self._tools = {t.name: t for t in tools}
        self._system_prompt = system_prompt
        self._middleware = middleware or []
        self._output_schema = output_schema     # None = text output; set = structured
        self._checkpoint_store = checkpoint_store
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(self, input: str, *, config: RunConfig | None = None) -> AgentResult:
        """Convenience: iterate stream(), return final result."""
        result = None
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                result = event.data.get("result")
        return result

    async def stream(self, input: str, *, config: RunConfig | None = None) -> AsyncIterator[AgentEvent]:
        config = config or RunConfig()
        agent_id = config.metadata.get("agent_id", "")
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=input),
        ]
        step = 0
        tool_call_count = 0

        while config.max_steps is None or step < config.max_steps:
            # 1. Drain inbox
            while self._inbox:
                injected = self._inbox.pop(0)
                messages.append(HumanMessage(content=f"[Injected message]\n{injected}"))
                yield AgentEvent(type="inject", agent_id=agent_id, step=step,
                                 data={"message": injected})

            # 2. Middleware: on_llm_start
            ctx = LoopContext(
                agent_id=agent_id, step=step, max_steps=config.max_steps,
                tool_call_count=tool_call_count, metadata=config.metadata,
            )
            prepared = messages
            for mw in self._middleware:
                prepared = await mw.on_llm_start(prepared, ctx)

            yield AgentEvent(type="llm_start", agent_id=agent_id, step=step)

            # 3. Call LLM
            response = await self._model.ainvoke(prepared)

            # 4. Middleware: on_llm_end
            for mw in self._middleware:
                response = await mw.on_llm_end(response, ctx)

            messages.append(response)
            yield AgentEvent(type="llm_end", agent_id=agent_id, step=step,
                             data={"content": getattr(response, "content", "")})

            # 5. No tool calls → finalize
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                if self._output_schema:
                    # Structured output: extra LLM call to compress conversation
                    # into a Pydantic model (mirrors current collect_and_compress)
                    structured_model = self._model.with_structured_output(self._output_schema)
                    output = await structured_model.ainvoke(messages)
                else:
                    output = getattr(response, "content", str(response))

                result = AgentResult(
                    agent_id=agent_id, status=AgentStatus.COMPLETED,
                    output=output, steps=step + 1, tool_calls=tool_call_count,
                )
                yield AgentEvent(type="complete", agent_id=agent_id, step=step,
                                 data={"result": result})
                return

            # 6. Execute tools (through middleware wrapping chain)
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})

                # Build the call chain: mw_N( ... mw_1( actual_tool ) ... )
                async def _actual_call(n: str, a: dict) -> str:
                    return await self._tools[n].ainvoke(a)

                chain = _actual_call
                for mw in reversed(self._middleware):
                    prev = chain
                    chain = lambda n, a, _mw=mw, _prev=prev: (
                        _mw.on_tool_call(n, a, _prev, ctx)
                    )

                yield AgentEvent(type="tool_start", agent_id=agent_id, step=step,
                                 data={"tool": name, "args": args})
                result_str = await chain(name, args)
                tool_call_count += 1
                yield AgentEvent(type="tool_end", agent_id=agent_id, step=step,
                                 data={"tool": name, "result": result_str})

                messages.append(ToolMessage(
                    content=result_str, tool_call_id=tc.get("id", ""),
                ))

            step += 1

            # 7. Checkpoint (optional)
            if self._checkpoint_store:
                await self._checkpoint_store.save(
                    agent_id, {"messages": messages, "step": step},
                )

        # Max steps exhausted
        result = AgentResult(
            agent_id=agent_id, status=AgentStatus.FAILED,
            error=f"Max steps ({config.max_steps}) reached",
            steps=step, tool_calls=tool_call_count,
        )
        yield AgentEvent(type="complete", agent_id=agent_id, step=step,
                         data={"result": result})
```

**Notes:**
- Uses `langchain_core` message types for compatibility with existing tools and models. This is a data format dependency, not a framework dependency.
- `ChatModel` protocol: any object with `async ainvoke(messages) -> response`. LangChain's `BaseChatModel` satisfies this. So does a plain wrapper around the Anthropic/OpenAI SDK.
- `inject()` appends to a list. Single-process asyncio makes this safe without locks.

### 5.2 LangGraphAgentLoop

Wraps an existing LangGraph compiled graph (either `create_react_agent` or a custom `StateGraph`). Provides checkpointing and streaming from LangGraph while conforming to the `AgentLoop` protocol.

```python
class LangGraphAgentLoop:
    """Wraps a LangGraph CompiledGraph as an AgentLoop.

    Used when you need LangGraph-specific features:
    - Full checkpoint history with time-travel
    - Subgraph streaming
    - interrupt_before / interrupt_after
    - Complex multi-node graphs (orchestrator's 3-node decision graph)

    For simple ReAct agents, prefer SimpleAgentLoop.
    """

    def __init__(self, compiled_graph: Any) -> None:
        self._graph = compiled_graph
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(self, input: str, *, config: RunConfig | None = None) -> AgentResult:
        config = config or RunConfig()
        graph_config = {
            "configurable": {
                "thread_id": config.thread_id or str(uuid.uuid4()),
            },
        }
        if config.max_steps:
            graph_config["recursion_limit"] = config.max_steps * 2

        input_data = {"messages": [HumanMessage(content=input)]}
        result = await self._graph.ainvoke(input_data, graph_config)

        return AgentResult(
            agent_id=config.metadata.get("agent_id", ""),
            status=AgentStatus.COMPLETED,
            output=result,
        )

    async def stream(self, input: str, *, config: RunConfig | None = None) -> AsyncIterator[AgentEvent]:
        config = config or RunConfig()
        graph_config = {
            "configurable": {
                "thread_id": config.thread_id or str(uuid.uuid4()),
            },
        }
        input_data = {"messages": [HumanMessage(content=input)]}

        async for namespace, mode, data in self._graph.astream(
            input_data, graph_config,
            stream_mode=["updates", "custom"],
            subgraphs=True,
        ):
            yield AgentEvent(
                type=mode,
                agent_id=config.metadata.get("agent_id", ""),
                data=data if isinstance(data, dict) else {"raw": data},
            )
```

**When to use which:**

| Scenario | Recommended Loop | Reason |
|----------|-----------------|--------|
| Simple workers (data collection, verification) | `SimpleAgentLoop` | No graph overhead, direct middleware control |
| Complex orchestrator (multi-node decision routing) | `LangGraphAgentLoop` | Current orchestrator uses 3-node graph with `<decision>` routing |
| Agents needing full checkpoint history | `LangGraphAgentLoop` | LangGraph checkpoint is production-grade |
| New scenarios (prototyping) | `SimpleAgentLoop` | Faster iteration, no compile step |

---

## 6. Scenario Integration Pattern

Scenarios build on the SDK by:
1. Creating tools that wrap `AgentRuntime` methods
2. Building `AgentLoop` instances with scenario-specific prompts and tools
3. Using `ReasoningStrategy` for domain logic (unchanged)

### 6.1 How an Orchestrator's Tools Map to AgentRuntime

```python
# Scenario-level code (e.g. scenarios/rca/)
def create_orchestrator_tools(runtime: AgentRuntime, agent_factory, **kwargs):
    """Create tools that the orchestrator LLM can call."""

    @tool
    async def dispatch_agent(agent_id: str, instruction: str, task_type: str) -> str:
        """Dispatch a worker agent to investigate."""
        loop = agent_factory.create_worker(task_type=task_type)
        handle = await runtime.spawn(
            agent_id=f"{agent_id}-{uuid.uuid4().hex[:6]}",
            loop=loop,
            input=instruction,
            metadata={"task_type": task_type, **kwargs},
        )
        return f"Dispatched {handle.agent_id}"

    @tool
    async def check_tasks() -> dict:
        """Check status of all dispatched workers."""
        status = runtime.get_status()
        # ... format into the same structure Orchestrator expects ...
        return formatted_status

    @tool
    async def inject_instruction(agent_id: str, instruction: str) -> str:
        """Send a new instruction to a running worker."""
        await runtime.send(agent_id, instruction)
        return f"Instruction sent to {agent_id}"

    @tool
    async def abort_task(agent_id: str, reason: str) -> str:
        """Abort a running worker."""
        await runtime.abort(agent_id, reason)
        return f"Aborted {agent_id}: {reason}"

    return [dispatch_agent, check_tasks, inject_instruction, abort_task]
```

### 6.2 Worker-to-Orchestrator Reporting (New Capability)

For scenarios where workers need to proactively report findings:

```python
# Scenario provides this tool to worker agents
def create_worker_tools(runtime: AgentRuntime, orchestrator_id: str):

    @tool
    async def report_finding(finding: str) -> str:
        """Report a significant finding to the orchestrator."""
        await runtime.send(orchestrator_id, f"[Worker finding] {finding}")
        return "Finding reported"

    return [report_finding]
```

The orchestrator's loop drains its inbox and sees these as injected messages. No architectural change needed — just a new tool that calls `runtime.send()` in the reverse direction.

### 6.3 RCA Scenario Sketch

```python
class RCAScenario:
    def __init__(self, scenario_config: ScenarioConfig):
        self.config = scenario_config
        self.strategy = HypothesisDrivenStrategy()

    async def run(self, task_description: str) -> AgentResult:
        runtime = AgentRuntime()

        # Build orchestrator tools that reference the runtime
        orch_tools = create_orchestrator_tools(
            runtime, self._agent_factory(),
        )
        orch_tools += self.strategy.create_scenario_tools(...)

        # Build orchestrator loop (complex graph → use LangGraph backend)
        orch_loop = self._build_orchestrator_loop(orch_tools)

        # Spawn orchestrator — it will spawn workers via dispatch_agent tool
        handle = await runtime.spawn(
            "orchestrator",
            loop=orch_loop,
            input=task_description,
        )

        return await handle.wait()

    def _agent_factory(self):
        """Factory that creates worker AgentLoops on demand."""
        # Workers are simple ReAct loops — no need for LangGraph
        class Factory:
            def create_worker(self, task_type: str) -> AgentLoop:
                return SimpleAgentLoop(
                    model=create_chat_model(config.model, ...),
                    tools=tool_registry.get_tools(config.tools),
                    system_prompt=load_prompt(config.prompt, task_type=task_type),
                    middleware=[BudgetMiddleware(...), CompressionMiddleware(...)],
                    output_schema=get_answer_schema(task_type),  # structured output
                )
        return Factory()
```

### 6.4 Shared State Between Agents (Scenario Pattern)

Shared state (e.g. `ServiceProfileStore`) is a **scenario concern**, not an SDK concern. The SDK provides the building blocks — tools and closures — and the scenario wires them.

```python
# RCA scenario: shared service profile between orchestrator and workers

class RCAScenario:
    def __init__(self, config):
        # Shared state — lives at the scenario level
        self.profile_store = ServiceProfileStore()

    def _agent_factory(self):
        store = self.profile_store  # captured by closure

        class Factory:
            def create_worker(self, task_type: str) -> AgentLoop:
                return SimpleAgentLoop(
                    model=...,
                    tools=[
                        *tool_registry.get_tools(config.tools),
                        make_update_profile_tool(store),   # write shared state
                        make_query_profile_tool(store),    # read shared state
                    ],
                    output_schema=get_answer_schema(task_type),
                    ...
                )
        return Factory()

    def _orchestrator_tools(self, runtime):
        store = self.profile_store

        return [
            *create_orchestrator_tools(runtime, self._agent_factory()),
            make_query_profile_tool(store),  # orchestrator can also read
        ]
```

**Key design**: The `ServiceProfileStore` is just a Python object. Worker A writes to it via `update_service_profile` tool; Worker B reads from it via `query_service_profile` tool. The SDK has no knowledge of the store's existence — it just executes tool calls.

**Why this works for structured data sharing**:
- Tools can accept and return structured data (Pydantic models as args/return)
- The shared object's merge semantics (topology union, anomaly upgrade) are domain-specific — RCA-specific — not something a generic SDK should abstract
- If the deployment model changes to multi-process, only the store implementation changes (in-memory → Redis/SQLite). The tool interface and the SDK interface stay identical.

---

## 7. Middleware Migration

### Current → Proposed Mapping

| Current (`AgentMMiddleware`) | Proposed (`Middleware`) | Notes |
|------------------------------|----------------------|-------|
| `before_model(state) -> state` | `on_llm_start(messages, ctx) -> messages` | Messages instead of full state dict |
| `aafter_model(state) -> state` | `on_llm_end(response, ctx) -> response` | Direct response access |
| `awrap_tool_call(request, handler)` | `on_tool_call(name, args, call_next, ctx)` | Same wrapping pattern, cleaner signature |
| `NodePipeline.before()` | `on_llm_start` chain | Same semantics, simpler API |
| `NodePipeline.after()` | `on_llm_end` chain | Same semantics |

### Existing Middleware Compatibility

All existing middleware can be adapted with a thin wrapper:

```python
class LegacyMiddlewareAdapter(Middleware):
    """Adapt an AgentMMiddleware to the new Middleware protocol."""

    def __init__(self, legacy: AgentMMiddleware):
        self._legacy = legacy

    async def on_llm_start(self, messages, ctx):
        state = {"messages": messages}
        result = self._legacy.before_model(state)
        if result is not None:
            return result.get("llm_input_messages", result.get("messages", messages))
        return messages
```

This allows incremental migration — existing middleware works immediately, and can be rewritten to the new interface over time.

---

## 8. Migration Strategy

### Phase 1: Define Protocols, Wrap Existing Code (Non-Breaking)

- Define `AgentLoop`, `Middleware`, `AgentRuntime`, `CheckpointStore` protocols
- Implement `LangGraphAgentLoop` wrapping current compiled graphs
- Implement `LegacyMiddlewareAdapter`
- `AgentRuntime` initially delegates to existing `TaskManager`
- **Zero breaking changes** — existing scenarios continue to work

### Phase 2: Implement SimpleAgentLoop

- Build the pure-Python ReAct loop
- Migrate simple workers (data collection, verification) from LangGraph to `SimpleAgentLoop`
- Keep orchestrator on `LangGraphAgentLoop` (complex decision routing graph)
- Validate: same behavior, simpler code path

### Phase 3: Refactor Scenarios to Use AgentRuntime

- Replace `AgentPool` with on-demand `runtime.spawn()` calls
- Replace `TaskManager` usage with `AgentRuntime` methods
- Add bidirectional communication where scenarios need it
- `ReasoningStrategy` stays unchanged (it's scenario-level, correctly placed)

### What Does NOT Change

- `ReasoningStrategy[S]` protocol — correct abstraction, stays
- `ToolRegistry` — tool definitions are orthogonal to agent loops
- `TrajectoryCollector` — becomes an `EventHandler` adapter
- Config system (`scenario.yaml`, `system.yaml`) — stays, drives agent loop creation
- Middleware implementations (budget, compression, trajectory, dedup, loop_detection) — adapt to new protocol

---

## 9. Deferred Decisions

| Decision | Why Deferred |
|----------|-------------|
| Multi-process spawn backend (`ProcessBackend`) | Needs picklable agent loops; current asyncio model works. Revisit when batch eval needs it. |
| File-system persistence for AgentRuntime state | Wait for runtime interface to stabilize. In-memory is fine for v1. |
| Agent-to-agent peer messaging (worker ↔ worker) | No scenario needs it yet. Runtime supports it (just `send(to, msg)`), but no scenario tools expose it. |
| Distributed checkpointing (Redis/PostgreSQL) | Keep using LangGraph's checkpoint implementation via `LangGraphAgentLoop` for now. |
| Custom graph topologies beyond ReAct | `SimpleAgentLoop` covers ReAct. Non-ReAct topologies stay on `LangGraphAgentLoop`. Long-term: consider a lightweight graph builder. |
| Structured output (response_format) in SimpleAgentLoop | Use `LangGraphAgentLoop` for agents needing structured output until SimpleAgentLoop adds it. |

---

## 10. Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| `stream()` is primary, `run()` is derived | Runtime needs events for EventHandler. `stream()` provides both events and result. `run()` trivially wraps it. | Every AgentLoop impl must implement `stream()` (slightly more work than just `run()`) |
| 3 middleware hooks (llm_start, llm_end, tool_call) | `on_tool_call` wrapping pattern handles cache short-circuit (dedup), before/after recording (trajectory), and output transform (truncation) — all in one hook. Separate start/end hooks couldn't handle short-circuit. | Wrapping composition is slightly more complex than linear chaining |
| `parent_id` for cascading abort | Orchestrator spawns workers; when orchestrator terminates, orphan workers waste resources. Auto-cleanup prevents this. | Adds a tree structure to runtime (parent → children tracking) |
| `wait_any(agent_ids=)` with filter | Orchestrator may want to wait for specific worker subset. Without filter, must poll or create separate events. | Slightly more complex than parameterless `wait_any()` |
| `AgentResult.steps` + `tool_calls` as explicit fields | Every scenario needs these metrics. Forcing them into metadata is repetitive and error-prone. | Two more fields on a dataclass (negligible cost) |
| AgentLoop as Protocol (not base class) | Allows any implementation; no inheritance required | No shared default behavior (each impl is standalone) |
| SimpleAgentLoop as default for workers | Workers are straightforward ReAct — no need for graph overhead | Must reimplement checkpoint if needed (or use CheckpointStore directly) |
| LangGraphAgentLoop for orchestrator | Orchestrator's 3-node decision graph benefits from LangGraph's StateGraph | Keeps LangGraph as a dependency for now |
| AgentRuntime subsumes TaskManager | Unified interface; avoids two coordination layers | TaskManager's smart wait strategy is replaced by `wait_any()` with asyncio.Event |
| Inbox as list (not asyncio.Queue) | Simpler; single-thread asyncio makes list safe | Must change to Queue if multi-process backend is added |
| `langchain_core` message types kept | Huge ecosystem of tools and models use them | Light coupling to LangChain's data types (not framework) |

---

## 11. Related Documents

| Document | Relationship |
|----------|-------------|
| [system-design-overview.md](system-design-overview.md) | Current architecture — this design evolves it |
| [orchestrator.md](orchestrator.md) | Orchestrator stays on LangGraph via `LangGraphAgentLoop` |
| [sub-agent.md](sub-agent.md) | Workers migrate to `SimpleAgentLoop` |
| [generic-state-wrapper.md](generic-state-wrapper.md) | `ReasoningStrategy` unchanged; `GenericAgentSystemBuilder` evolves into scenario-level builder using `AgentRuntime` |
| [Claude Code architecture reference](../../docs/references/claude-code-agent-team-architecture.md) | External reference that motivated this design |
