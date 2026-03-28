# Design: AgentM System Architecture

**Status**: CURRENT
**Last Updated**: 2026-03-28

---

## Overview

AgentM is a **scenario-driven multi-agent orchestration framework** built on pure Python async primitives (no LangGraph in the execution path). A central orchestrator (`SimpleAgentLoop`) coordinates multiple worker agents via an `AgentRuntime`, dispatching tasks and collecting results through tool calls in a ReAct cycle.

### Design Goals

| Goal | Description |
|------|-------------|
| Scenario-driven extensibility | All domain-specific behavior lives in Scenario plugins; SDK is generic |
| Pure Python agent loops | No framework dependency in the execution path; `SimpleAgentLoop` implements ReAct directly |
| Config-driven system | Prompts, tools, models, middleware configured via YAML files |
| Middleware-based composition | Cross-cutting concerns (compression, dedup, trajectory, budget) as composable middleware |
| Trajectory recording | Structured JSONL event capture with async/sync dual API |
| Multi-scenario support | 3 built-in scenarios: RCA, trajectory analysis, general purpose |
| Visual debugging | Rich terminal console + web dashboard for real-time monitoring |

---

## Architecture Diagram

```
                    config/system.yaml
                    config/scenarios/<name>/scenario.yaml
                    config/tools/*.yaml
                            |
                            v
               build_agent_system(scenario_name, scenario_config)
                            |
            +---------------+----------------+
            |               |                |
            v               v                v
    _create_platform    scenario.setup()   _create_worker
    _resources()        -> ScenarioWiring  _infrastructure()
            |               |                |
            v               v                v
    ToolRegistry        tools, schemas    AgentRuntime
    Vault               hooks, context    WorkerLoopFactory
    Trajectory          middleware
            |               |                |
            +-------+-------+--------+-------+
                    |                |
                    v                v
         _assemble_orchestrator   _build_orchestrator
         _tools()                 _loop()
                    |                |
                    v                v
              tool list         SimpleAgentLoop
                                (orchestrator)
                                     |
                                     v
                              AgentSystem
                              .execute(input) / .stream(input)
```

### Runtime execution flow

```
AgentSystem.execute(input)
    |
    v
SimpleAgentLoop (orchestrator)
    |
    +---> LLM call (with middleware: DynamicContext, LoopDetection,
    |         Compression, Skill, Trajectory)
    |
    +---> Tool calls (parallel when multiple):
    |       dispatch_agent  --> AgentRuntime.spawn()
    |       check_tasks     --> AgentRuntime.wait_any() + get_status()
    |       inject_instruction --> AgentRuntime.send()
    |       abort_task      --> AgentRuntime.abort()
    |       update_hypothesis, query_service_profile, ...
    |
    +---> Repeat until should_terminate(response) == True
    |
    +---> _synthesize_output() --> structured output (optional)
    |
    v
AgentResult(output, status, steps, tool_calls)
```

Workers run as independent `asyncio.Task`s inside `AgentRuntime`:

```
AgentRuntime
    |
    +-- spawn(agent_id, loop, input)
    |     |
    |     +-- asyncio.create_task(_run_agent)
    |           |
    |           +-- SimpleAgentLoop.stream(input)
    |                 |
    |                 +-- LLM -> tools -> LLM -> ... -> result
    |
    +-- wait(agent_id) --> AgentResult
    +-- wait_any(ids)  --> [completed_ids]
    +-- send(to, msg)  --> inject into inbox
    +-- abort(id)      --> cancel task + cascade children
```

---

## Core SDK (Harness)

The harness layer (`src/agentm/harness/`) defines the execution primitives.

### Protocols (`protocols.py`)

| Protocol | Methods | Purpose |
|----------|---------|---------|
| `AgentLoop` | `run()`, `stream()`, `inject()` | Single agent's conversation loop |
| `Middleware` | `on_llm_start()`, `on_llm_end()`, `on_tool_call()` | Hook into the agent loop at 3 points |
| `CheckpointStore` | `save()`, `load()`, `list_checkpoints()` | Persist/recover agent state (defined, not yet integrated) |
| `EventHandler` | `on_event()` | Receive streaming events from all agents |

### Types (`types.py`)

| Type | Description |
|------|-------------|
| `AgentStatus` | Enum: `RUNNING`, `COMPLETED`, `FAILED`, `ABORTED` |
| `AgentResult` | Outcome of an agent loop: `output`, `status`, `steps`, `tool_calls`, `duration_seconds` |
| `AgentEvent` | Streaming event: `type` (llm_start, llm_end, tool_start, tool_end, inject, complete, error), `agent_id`, `data`, `step` |
| `RunConfig` | Per-run config: `max_steps`, `timeout`, `thread_id`, `metadata` |
| `LoopContext` | Read-only context for middleware: `agent_id`, `step`, `max_steps`, `tool_call_count` |
| `AgentInfo` | Runtime status snapshot of an agent |

### SimpleAgentLoop (`harness/loops/simple.py`)

Pure-Python ReAct implementation. No LangGraph dependency.

- **Cycle**: LLM call -> tool execution -> LLM call -> ... -> termination
- **Parallel tools**: Multiple tool calls in a single LLM response are executed via `asyncio.gather()`
- **Inbox**: Messages injected via `inject()` are drained before each LLM call
- **Middleware**: `on_llm_start` chains in order, `on_llm_end` chains in order, `on_tool_call` uses wrapping pattern with `call_next`
- **Termination**: Configurable `should_terminate(response)` callback; default parses `<decision>finalize</decision>` tags
- **Retry**: Exponential backoff on transient LLM errors (429, 5xx, timeout)
- **Structured output**: `_synthesize_output()` with retry and fallback when `output_schema` is set

### AgentRuntime (`harness/runtime.py`)

Manages multiple `AgentLoop` instances as `asyncio.Task`s.

- **`spawn()`**: Creates a task, returns an `AgentHandle`
- **`wait()` / `wait_any()`**: Block until agent(s) complete
- **`send()`**: Inject message into a running agent's inbox
- **`abort()`**: Cancel a task with cascading child abort
- **`get_status()`**: Snapshot of all agents
- **Trajectory**: Records `task_dispatch`, `task_complete`, `task_fail`, `task_abort` events

### AgentHandle (`harness/handle.py`)

Convenience wrapper for a spawned agent. Delegates to `AgentRuntime` for `wait()`, `send()`, `abort()`, `status`, `result`.

### WorkerLoopFactory (`harness/worker_factory.py`)

Creates fully configured `SimpleAgentLoop` instances for workers.

- Resolves tools from `ToolRegistry` + extra tools from `ScenarioWiring`
- Builds system prompt from base template + task_type overlay (Jinja2)
- Assembles middleware stack: extra middleware, `BudgetMiddleware`, `LoopDetectionMiddleware`, `CompressionMiddleware`, `TrajectoryMiddleware`, `DedupMiddleware`
- Selects `output_schema` from `answer_schemas` based on `task_type`

### Tool (`harness/tool.py`)

SDK-native tool type. No langchain dependency.

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]    # JSON Schema
    func: Callable[..., Any]

    async def ainvoke(self, args: dict[str, Any]) -> str: ...
    def to_openai_schema(self) -> dict[str, Any]: ...
```

- `tool_from_function()`: Create a `Tool` from any callable (including `functools.partial`)
- `@tool` decorator: Turn a function into a `Tool` (supports `@tool` and `@tool(name=...)`)
- Schema inference via `pydantic.TypeAdapter` from function signatures

### Middleware (`harness/middleware.py`)

`MiddlewareBase` provides default pass-through for all 3 hooks. Concrete implementations:

| Middleware | Hook(s) | Purpose |
|-----------|---------|---------|
| `DynamicContextMiddleware` | `on_llm_start` | Injects dynamic scenario state + round counter into system prompt |
| `LoopDetectionMiddleware` | `on_llm_start` | Detects repeated tool calls and think-stalls, injects warnings |
| `CompressionMiddleware` | `on_llm_start` | LLM-based summarization when token count exceeds threshold |
| `BudgetMiddleware` | `on_llm_start` | Injects urgency messages when step/tool budgets run low |
| `SkillMiddleware` | `on_llm_start` | Injects skill context from MarkdownVault into system prompt |
| `TrajectoryMiddleware` | `on_llm_start`, `on_llm_end`, `on_tool_call` | Records trajectory events for LLM calls and tool executions |
| `DedupMiddleware` | `on_llm_start`, `on_tool_call` | Deduplicates tool calls via caching + LLM reminders |

---

## Scenario System

### Scenario Protocol

The [Scenario Protocol](generic-state-wrapper.md) is the SDK extension point. Each scenario implements:
- `name` property -- string identifier
- `setup(ctx: SetupContext) -> ScenarioWiring` -- returns all domain-specific wiring

### Registration

`register_scenario()` + `get_scenario()` module-level functions in `harness/scenario.py`. `discover()` in `scenarios/__init__.py` auto-registers all built-in scenarios.

### Built-in Scenarios

| Scenario | `system.type` | Purpose | Key wiring |
|----------|--------------|---------|------------|
| **RCA** (`hypothesis_driven`) | `hypothesis_driven` | Root cause analysis with hypothesis tracking | HypothesisStore, ServiceProfileStore, format_context, 3 answer schemas, CausalGraph output, custom tools |
| **Trajectory Analysis** (`trajectory_analysis`) | `trajectory_analysis` | Skill-driven analysis of completed agent trajectories | 2 answer schemas (analyze, critique), AnalysisReport output |
| **General Purpose** (`general_purpose`) | `general_purpose` | Open-ended task execution | 1 answer schema (execute), minimal wiring |

---

## Configuration

### Config Types (`config/schema.py`)

```
SystemConfig                          ScenarioConfig
  models: dict[str, ModelConfig]        system: SystemTypeConfig
  storage: StorageConfig                orchestrator: OrchestratorConfig
  recovery: RecoveryConfig              agents: dict[str, AgentConfig]
  debug: DebugConfig
```

Key types:
- `ModelConfig`: api_key, base_url, rate_limit, provider (openai/anthropic)
- `OrchestratorConfig`: model, temperature, prompts, tools, max_rounds, retry, loop_detection, compression, output, disable_tool_binding
- `AgentConfig`: model, temperature, prompt, tools, tool_settings, task_type_prompts, execution (max_steps, timeout, tool_call_budget, dedup, retry, loop_detection, max_concurrent_workers)
- `CompressionConfig`: threshold, model, context_window, preserve_latest_n
- `RetryConfig`: max_attempts, initial_interval, backoff_factor
- `LLMConfig`: shared base for `OrchestratorConfig` and `AgentConfig` (model, temperature, compression, skills, include_think_tool)

### Config Loading (`config/loader.py`)

- `load_system_config(path)` -- YAML -> `SystemConfig`
- `load_scenario_config(path)` -- YAML -> `ScenarioConfig`
- `substitute_env_vars()` -- resolves `${VAR}` and `${VAR:default}` placeholders
- `load_tool_definitions(tools_dir)` -- loads all tool YAML files

### YAML Structure

```
config/
  system.yaml                    # Global: models, storage, recovery, debug
  tools/                         # Tool definitions (reusable across scenarios)
    infrastructure.yaml
    database.yaml
  scenarios/
    rca_hypothesis/
      scenario.yaml              # Orchestrator + agents config
      prompts/                   # Jinja2 templates
        orchestrator_system.j2
        agents/worker.j2
    trajectory_analysis/
      scenario.yaml
      prompts/...
    general_purpose/
      scenario.yaml
      prompts/...
```

### Validation

- **Phase 1 (schema)**: Pydantic model validation on YAML load
- **Phase 2 (cross-reference)**: `validate_references()` checks model names, tool names, prompt paths exist

---

## Build Pipeline

`build_agent_system()` in `src/agentm/builder.py` executes 4 phases:

### Phase 1: Platform Resources (`_create_platform_resources`)

- `ToolRegistry`: loads all `config/tools/*.yaml` definitions
- `MarkdownVault`: initialized from `knowledge_base_dir/vault`
- `TrajectoryCollector`: JSONL event capture (if `debug.trajectory.enabled`)
- Memory tools: `read_trajectory`, `get_checkpoint_history`, `jq_query`, `load_case_data`
- Model configs: resolved from `system_config.models`

### Phase 2: Worker Infrastructure (`_create_worker_infrastructure`)

- `AgentRuntime`: manages worker lifecycle (spawn, wait, abort, send)
- `WorkerLoopFactory`: configured with tool_registry, model_config, extra_tools/middleware from wiring, answer_schemas

### Phase 3: Orchestrator Tools (`_assemble_orchestrator_tools`)

Resolves tools listed in `orchestrator.tools` config from 4 sources:
1. SDK tools (`dispatch_agent`, `check_tasks`, `inject_instruction`, `abort_task`)
2. Scenario tools (from `ScenarioWiring.orchestrator_tools`)
3. Memory tools
4. Registry tools (from YAML definitions)

Appends `think` tool if `include_think_tool` is enabled.

### Phase 4: Orchestrator Loop (`_build_orchestrator_loop`)

Assembles the orchestrator's `SimpleAgentLoop`:
- **System prompt**: loaded from Jinja2 template
- **Middleware stack** (in order):
  1. `DynamicContextMiddleware` (format_context + round tracking)
  2. `LoopDetectionMiddleware` (if `think_stall_enabled`)
  3. `CompressionMiddleware` (if compression config enabled)
  4. `SkillMiddleware` (if skills configured)
  5. Scenario middleware (from `wiring.orchestrator_middleware`)
  6. `TrajectoryMiddleware` (if trajectory enabled)
- **Model**: `create_chat_model()` with tool binding via `bind_tools()`
- **Termination**: `wiring.should_terminate` or default `<decision>` tag parser
- **Retry**: configurable max_attempts, initial_interval, backoff_factor

Returns `AgentSystem(loop, scenario_config, runtime, trajectory, thread_id)`.

---

## Execution Flow

### Orchestrator Loop

1. Orchestrator `SimpleAgentLoop` receives the task input
2. Each iteration: drain inbox -> `on_llm_start` middleware -> LLM call with retry -> `on_llm_end` middleware
3. Check `should_terminate(response)`:
   - If `True`: synthesize output (optional structured output with retry) -> return `AgentResult`
   - If `False` with tool calls: execute tools (parallel if multiple) through middleware `on_tool_call` chain
4. After tool execution, loop back to step 2
5. If `max_steps` exceeded: return `FAILED` result

### Worker Dispatch

The orchestrator dispatches workers via the `dispatch_agent` tool:

1. `dispatch_agent(agent_id, task, task_type)` -> `WorkerLoopFactory.create_worker()` -> `AgentRuntime.spawn()`
2. Worker runs as an `asyncio.Task` with its own `SimpleAgentLoop`
3. **Auto-block optimization**: if this is the only running worker, `dispatch_agent` waits for completion and returns the result directly (saves an LLM roundtrip through `check_tasks`)
4. **Concurrency limit**: `asyncio.Semaphore` when `max_concurrent_workers` is configured
5. Orchestrator collects results via `check_tasks` (waits briefly for any completion) or reads them inline from `dispatch_agent`

### Orchestrator Tools

Created by `create_orchestrator_tools()` in `src/agentm/tools/orchestrator.py`:

| Tool | Purpose |
|------|---------|
| `dispatch_agent` | Launch a worker with agent_id, task, task_type. Auto-blocks for single worker |
| `check_tasks` | Query status of all dispatched workers. Waits briefly for completions |
| `inject_instruction` | Send a message into a running worker's inbox |
| `abort_task` | Cancel a running worker (cascades to children) |

---

## Infrastructure

### TrajectoryCollector (`core/trajectory.py`)

Structured JSONL event capture:
- **Dual API**: `record()` (async with lock) and `record_sync()` (sync with threading lock)
- **Shared core**: `_record_core()` handles sequencing, serialization, file I/O
- **Events**: `TrajectoryEvent` Pydantic model with `run_id`, `seq`, `timestamp`, `agent_path`, `event_type`, `data`, `task_id`, `metadata`
- **Listeners**: Fan-out to registered callbacks (async and sync)
- **File**: `{output_dir}/{run_id}.jsonl` with `_meta` header line

### ToolRegistry (`core/tool_registry.py`)

YAML-based dynamic tool loading:
- `ToolDefinition`: wraps a function with config schema
- `load_from_yaml()`: imports module, resolves function, registers
- `create_tool(**config)`: creates `Tool` instance with bound config parameters via `functools.partial`

### DebugConsole (`core/debug_console.py`)

Rich-based real-time terminal UI with 3 panels:
1. **Agent Status**: task_id, agent, status, duration
2. **Tool Timeline**: last N tool calls with name, args, time, agent
3. **Hypothesis Board**: hypothesis id, status (color-coded), description

Consumes events via `TrajectoryCollector.add_listener()`.

### Dashboard (`server/app.py`)

FastAPI web application:
- **WebSocket** (`/ws`): real-time event streaming with history replay on connect
- **REST API**: `/api/topology` (agent config), `/api/eval/*` (batch evaluation monitoring)
- **Broadcaster**: manages WebSocket clients, broadcasts events
- **Eval endpoints**: status, samples list, sample detail, sample events (JSONL reading)

### Vault

`MarkdownVault` -- Markdown + YAML frontmatter persistent knowledge store. Used for:
- Agent skills (loaded via `SkillMiddleware`)
- Domain knowledge

---

## CLI

Entry point: `agentm:main` registered as console script. Uses `typer`.

| Command | Purpose |
|---------|---------|
| `agentm analyze <trajectories> --task <desc>` | Analyze completed trajectories with evaluation feedback |
| `agentm analyze-batch <config>` | Batch analyze evaluation trajectories from config YAML |
| `agentm resume <trajectory> [--checkpoint <id>]` | Resume interrupted investigation from trajectory file |
| `agentm debug <trajectory>` | Analyze a trajectory JSONL file (summary, timeline, filters) |
| `agentm export-result <trajectory>` | Export case_dir + ground_truth + final outputs for one trajectory |
| `agentm export-batch <dir>` | Batch export from multiple trajectory files |

Options available across commands: `--scenario`, `--config`, `--debug`, `--verbose`, `--dashboard`, `--port`, `--max-steps`.

---

## Error Handling

### Exception Hierarchy (`exceptions.py`)

```
AgentMError
  ConfigError               -- config loading or validation
  DataInitError             -- data directory initialization
  ToolError                 -- tool execution
  AgentError                -- agent execution
  CheckpointError           -- checkpoint read/write
  StoreNotInitializedError  -- store used before init
```

### Error Recovery Layers

| Layer | Actor | Strategy |
|-------|-------|----------|
| 1. Tool self-heal | Worker LLM | Tool returns error string; LLM tries alternative |
| 2. LLM retry | `SimpleAgentLoop._invoke_with_retry()` | Exponential backoff on 429, 5xx, timeout |
| 3. Orchestrator decision | Orchestrator LLM | Failed worker visible in `check_tasks`; re-dispatch or skip |
| 4. Manual recovery | Human operator | Resume from trajectory file via `agentm resume` |

---

## OrchestratorHooks

```python
@dataclass
class OrchestratorHooks:
    think_stall_enabled: bool = True      # Enable LoopDetectionMiddleware
    synthesize_max_retries: int = 2       # Structured output retry count
```

Returned by `ScenarioWiring.hooks`. Controls orchestrator-level behavior.

---

## Cross-References

| Document | Scope |
|----------|-------|
| [Scenario Protocol](generic-state-wrapper.md) | Scenario protocol, ScenarioWiring, registration, examples |
| [Agent Harness](agent-harness.md) | AgentLoop, AgentRuntime, Middleware, CheckpointStore protocols |
| [SDK Consistency](sdk-consistency.md) | Unified Tool type, Scenario protocol refactoring |
| [Orchestrator](orchestrator.md) | Orchestrator loop, tools, state management |
| [Trajectory](trajectory.md) | TrajectoryCollector design |
| [Debug Console](debug-console.md) | Rich terminal UI design |
| [Frontend Architecture](frontend-architecture.md) | Dashboard UI and WebSocket protocol |
| [Trajectory Analysis](trajectory-analysis.md) | Trajectory analysis scenario design |
| [Memory Vault](memory-vault.md) | MarkdownVault knowledge store |
| [Tool Dedup](tool-dedup.md) | Tool call deduplication middleware |
| [Orchestrator](orchestrator.md) | RCA orchestrator prompt design |
| [Sub-Agent](sub-agent.md) | Worker agent configuration and behavior |
| [Testing Strategy](testing-strategy.md) | Test architecture |
