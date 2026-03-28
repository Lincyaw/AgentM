# Design: Sub-Agent (Worker)

**Status**: CURRENT
**Last Updated**: 2026-03-28

---

## 1. Overview

Workers (sub-agents) are **`SimpleAgentLoop` instances** -- pure-Python ReAct loops with no LangGraph dependency. They are created by `WorkerLoopFactory`, launched as `asyncio.Task`s by `AgentRuntime.spawn()`, and managed by the orchestrator through `dispatch_agent` / `check_tasks` tools.

Each worker runs independently with its own message history, tool set, middleware stack, and optional output schema. Workers receive a natural-language task instruction, execute a tool-calling loop, and return an `AgentResult`.

**Key files**:

| File | Purpose |
|------|---------|
| `src/agentm/harness/loops/simple.py` | `SimpleAgentLoop` -- the ReAct loop |
| `src/agentm/harness/worker_factory.py` | `WorkerLoopFactory` -- builds configured loops |
| `src/agentm/harness/runtime.py` | `AgentRuntime` -- lifecycle management |
| `src/agentm/harness/handle.py` | `AgentHandle` -- convenience API |
| `src/agentm/tools/orchestrator.py` | `dispatch_agent`, `check_tasks` tools |
| `src/agentm/harness/middleware.py` | All middleware implementations |
| `src/agentm/harness/tool.py` | `Tool` dataclass, `tool_from_function` |

---

## 2. WorkerLoopFactory

`WorkerLoopFactory` produces a fully configured `SimpleAgentLoop` for each `dispatch_agent` call. It assembles four things: tools, system prompt, middleware stack, and output schema.

The factory reads `scenario_config.agents["worker"]` to get the `AgentConfig`. All workers share the same base config; differentiation happens through `task_type`. Constructor accepts `extra_tools` (from `ScenarioWiring.worker_tools`), `extra_middleware`, `trajectory` collector, and `answer_schemas` (per task_type Pydantic models).

### create_worker(agent_id, task_type)

Each call builds a fresh loop:

1. **Tools** -- Registry tools (from `AgentConfig.tools`) + `extra_tools` (from `ScenarioWiring.worker_tools`) + optional `think` tool (if `include_think_tool` is true).
2. **System prompt** -- Base prompt template rendered via Jinja2, plus a `task_type` overlay if `task_type_prompts[task_type]` exists. Template variables include `agent_id` and `tools_description`.
3. **Middleware** -- Built in fixed order (see Section 4).
4. **Output schema** -- Looked up from `answer_schemas[task_type]`. When present, the loop uses structured output via `with_structured_output(schema, method="function_calling")`.
5. **Model** -- Created via `create_chat_model()` with tools bound via `bind_tools()`.
6. **Retry config** -- Passed through from `AgentConfig.execution.retry`.

---

## 3. Lifecycle

### Spawn

The orchestrator's `dispatch_agent` tool calls `WorkerLoopFactory.create_worker()` to get a loop, then `AgentRuntime.spawn()` to launch it. `spawn()` creates an `asyncio.Task` that runs `_run_agent()`, which iterates the loop's `stream()` method and forwards events to the optional `EventHandler`.

### Auto-Block

When `dispatch_agent` detects it is the **only running worker**, it blocks until that worker completes and returns the result directly -- saving an LLM roundtrip through `check_tasks`. When multiple workers are running, `dispatch_agent` returns immediately with `"status": "running"`.

### Monitoring and Collection

The orchestrator calls `check_tasks` to poll worker status. `check_tasks` calls `runtime.wait_any(running_ids, timeout=wait_seconds)` to briefly block for any worker to complete, then builds a status report with running/completed/failed buckets.

### Communication

- **inject_instruction** -- Sends a message to a running worker's inbox via `runtime.send()` -> `loop.inject()`. The message is drained before the next LLM call.
- **abort_task** -- Cancels the worker's `asyncio.Task` via `runtime.abort()`. Cascades to any children.

### AgentHandle

`AgentRuntime.spawn()` returns an `AgentHandle` -- a convenience wrapper exposing `agent_id`, `status`, `result` properties, and `wait()`, `send()`, `abort()` async methods. All delegate to the runtime.

### Cascade Abort

When a parent terminates, `AgentRuntime._cascade_children()` aborts all running children with that `parent_id`, preventing orphaned workers.

---

## 4. Middleware Stack

Middleware is assembled by `WorkerLoopFactory._build_middleware()` in this fixed order:

| Order | Middleware | Hook(s) Used | Purpose |
|-------|-----------|-------------|---------|
| 1 | Extra middleware (caller-injected) | varies | Scenario-specific hooks |
| 2 | `BudgetMiddleware` | `on_llm_start` | Injects urgency messages when step/tool budgets run low |
| 3 | `LoopDetectionMiddleware` | `on_llm_start` | Detects repetitive tool calls and think-stalls |
| 4 | `CompressionMiddleware` | `on_llm_start` | Summarizes older messages via LLM when token count exceeds threshold |
| 5 | `TrajectoryMiddleware` | `on_llm_start`, `on_llm_end`, `on_tool_call` | Records trajectory events |
| 6 | `DedupMiddleware` | `on_llm_start`, `on_tool_call` | Caches tool results, warns on repeat calls, short-circuits duplicates |

All middleware extends `MiddlewareBase` which provides pass-through defaults for all three hooks.

### Middleware Protocol

Three hooks (`on_llm_start`, `on_llm_end`, `on_tool_call`), all optional. `on_llm_start`/`on_llm_end` chain in order (output feeds the next). `on_tool_call` uses a wrapping pattern: middleware composed in reverse order so the first is the outermost wrapper. See [agent-harness.md](agent-harness.md) for the full `Middleware` protocol definition.

### BudgetMiddleware

Tracks step budget (`max_steps`) and optional tool call budget (`tool_call_budget`). Injects urgency messages at three levels: "start wrapping up" (1/3 remaining), "summarize NOW" (3 remaining), "BUDGET EXHAUSTED" (0 remaining).

### LoopDetectionMiddleware

Two detection modes:
- **Think-stall**: Counts consecutive AI messages where the only tool is `think`. Triggers warning at `think_stall_limit` (default 3).
- **Exact-match loop**: Counts identical `(tool_name, args)` pairs in a sliding window. Triggers warning at `threshold` (default 5) repeats.

### CompressionMiddleware

When total token count exceeds `compression_threshold * context_window`, summarizes older messages via an LLM call (preserving the latest `preserve_latest_n` messages). Uses chunked summarization for very long histories.

### TrajectoryMiddleware

Records `llm_start`, `tool_call`, `llm_end`, and `tool_result` events to `TrajectoryCollector` with an `agent_path` of `["orchestrator", agent_id]`.

### DedupMiddleware

Two-layer dedup:
1. **`on_llm_start`** -- Checks the last AI message for tools already in cache, injects a "DEDUP WARNING" reminder.
2. **`on_tool_call`** -- Short-circuits duplicate calls by returning cached results. Uses `OrderedDict` with FIFO eviction (`max_cache_size`, default 50). Excluded tools (default: `{"think"}`) bypass the cache.

---

## 5. Tool System

### Tool Dataclass

`Tool` is the SDK-native tool type (dataclass with `name`, `description`, `parameters` JSON Schema, `func`). Provides `ainvoke(args)` for execution and `to_openai_schema()` for model binding. `tool_from_function()` derives schema from a function's type hints via Pydantic `TypeAdapter`; works with `functools.partial`.

### Tool Sources

Workers receive tools from three sources:

1. **ToolRegistry** -- Loaded from YAML definitions. The worker's `AgentConfig.tools` lists tool names; the factory calls `registry.get(name).create_tool(**tool_settings)` for each.
2. **ScenarioWiring.worker_tools** -- Extra tools injected by the scenario's `setup()` method. These are passed as `extra_tools` to `WorkerLoopFactory`.
3. **Think tool** -- Optionally included when `AgentConfig.include_think_tool` is true (default).

### Answer Schemas

Per-task-type Pydantic `BaseModel` schemas are provided via `ScenarioWiring.answer_schemas`. When a schema exists for the worker's `task_type`, `SimpleAgentLoop._synthesize_output()` uses `with_structured_output(schema, method="function_calling")` to produce validated output.

Synthesis includes retry logic: up to `1 + synthesize_retries` attempts, with error feedback appended on validation failure. On final failure, falls back to plain LLM and returns `{"raw_text": ...}`.

---

## 6. Concurrency

`max_concurrent_workers` (from `ExecutionConfig`) controls parallelism via an `asyncio.Semaphore` created in `create_orchestrator_tools()`. `dispatch_agent` acquires before spawning; a background task releases when the worker finishes. When `None` (default), there is no concurrency limit.

---

## 7. SimpleAgentLoop

The core ReAct cycle implemented in pure Python:

```
Input -> [system prompt + user message]
  |
  v
+--- Loop (up to max_steps) ---+
|  1. Drain inbox               |
|  2. Middleware: on_llm_start   |
|  3. LLM call (with retry)     |
|  4. Middleware: on_llm_end     |
|  5. Check termination          |
|     - Yes -> synthesize output |
|     - No tool_calls -> next    |
|  6. Execute tools              |
|     - 1 tool: sequential       |
|     - N tools: asyncio.gather  |
|  7. Optional checkpoint        |
+-------------------------------+
```

### LLM Retry

`_invoke_with_retry()` retries transient errors (rate limits, 5xx, timeouts) with exponential backoff. Configured via `retry_max_attempts`, `retry_initial_interval`, `retry_backoff_factor`.

### Parallel Tool Execution

When the LLM returns multiple tool calls in one response, they execute in parallel via `asyncio.gather()`. Single tool calls execute directly without gather overhead. Exceptions from individual tools are caught and recorded as error results.

### Termination

The `should_terminate` callback (default: no tool_calls in response) determines when the loop ends. On termination, `_synthesize_output()` produces the final output, optionally using the `output_schema`.

### Events

The loop yields `AgentEvent`s for each stage: `inject`, `llm_start`, `llm_end`, `tool_start`, `tool_end`, `complete`. The runtime forwards these to `EventHandler`.

---

## 8. RCA Example

The RCA scenario (`scenarios/rca/scenario.py`) demonstrates the full wiring:

### Shared Service Profile Tools

Both orchestrator and workers share `ServiceProfileStore` access via `functools.partial`-bound functions wrapped with `tool_from_function()`. Workers query profiles before investigating (avoid redundancy) and update them during investigation (share discoveries in real-time).

### Task Types and Answer Schemas

| task_type | Answer Schema | Purpose |
|-----------|--------------|---------|
| `scout` | `ScoutAnswer` | Initial reconnaissance, anomaly discovery |
| `deep_analyze` | `DeepAnalyzeAnswer` | Focused deep dive into a data source |
| `verify` | `VerifyAnswer` | Hypothesis testing with verdict |

`ScenarioWiring` passes `worker_tools` (-> factory `extra_tools`), `answer_schemas` (-> factory per-task-type schemas), and `output_schema` (orchestrator final output).

---

## 9. Configuration

### AgentConfig

Worker configuration sits under `scenario_config.agents["worker"]`:

```python
class AgentConfig(LLMConfig):
    prompt: str | None                           # Base prompt template path
    tools: list[str]                             # ToolRegistry tool names
    tool_settings: dict[str, dict[str, Any]]     # Per-tool config overrides
    task_type_prompts: dict[str, str] | None     # task_type -> overlay template path
    execution: ExecutionConfig
    # Inherited from LLMConfig:
    #   model, temperature, compression, skills, include_think_tool
```

### ExecutionConfig

```python
class ExecutionConfig(BaseModel):
    max_steps: int = 20
    timeout: int = 120
    tool_call_budget: int | None = None
    dedup: DedupConfig | None = None
    retry: RetryConfig = RetryConfig()          # max_attempts=3, backoff_factor=2.0
    loop_detection: LoopDetectionConfig          # threshold=5, window_size=15, think_stall_limit=3
    max_concurrent_workers: int | None = None   # Semaphore-based concurrency limit
```

### YAML Example

```yaml
agents:
  worker:
    model: "gpt-4o-mini"
    temperature: 0.2
    include_think_tool: true
    prompt: "prompts/agents/worker.j2"
    task_type_prompts:
      scout: "prompts/task_types/scout.j2"
      verify: "prompts/task_types/verify.j2"
      deep_analyze: "prompts/task_types/deep_analyze.j2"
    tools: [check_metrics, query_logs, query_traces]
    execution:
      max_steps: 20
      timeout: 120
      tool_call_budget: 25
      max_concurrent_workers: 3
      retry: { max_attempts: 3, initial_interval: 1.0, backoff_factor: 2.0 }
      loop_detection: { threshold: 5, window_size: 15, think_stall_limit: 3 }
      dedup: { enabled: true, max_cache_size: 50 }
    compression: { compression_model: "gpt-4o-mini", compression_threshold: 0.8, context_window: 128000 }
```

---

## 10. Error Handling

Errors are handled at multiple layers:

1. **Tool errors** -- Returned as text to the LLM for self-correction. Unknown tools get a message listing available tools.
2. **LLM transient errors** -- `_invoke_with_retry()` handles rate limits, 5xx, timeouts with exponential backoff.
3. **Structured output failures** -- `_synthesize_output()` retries with error feedback, falls back to raw text.
4. **Task-level failures** -- `AgentRuntime._run_agent()` catches all exceptions, records `AgentStatus.FAILED` with error message, and notifies via `done_event`. The orchestrator sees failures through `check_tasks`.
5. **Cancellation** -- `asyncio.CancelledError` is caught and recorded as `AgentStatus.ABORTED`.

---

## 11. Related Documents

- [Agent Harness](agent-harness.md) -- SDK harness architecture (runtime, protocols, events)
- [Orchestrator](orchestrator.md) -- Orchestrator design and dispatch flow
- [Tool Dedup](tool-dedup.md) -- Deduplication middleware details
- [Trajectory](trajectory.md) -- Trajectory collection and analysis
