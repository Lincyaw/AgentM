# Design: Orchestrator

**Status**: CURRENT
**Last Updated**: 2026-03-28

---

## 1. Overview

The Orchestrator is a `SimpleAgentLoop` instance that runs a pure-Python ReAct cycle (LLM -> tool calls -> LLM -> ... -> final answer). There is no LangGraph dependency. The orchestrator uses tools to dispatch worker agents asynchronously via `AgentRuntime`, monitor their progress, inject instructions, and abort stuck agents.

The entire agent system (orchestrator + workers) is constructed by `build_agent_system()` in `src/agentm/builder.py` through a 4-phase build process. Domain-specific behavior is injected via the `Scenario` protocol, which returns a `ScenarioWiring` containing tools, schemas, hooks, context formatting, and middleware.

### Core Responsibilities

1. **Hypothesis reasoning** -- Generate and manage hypotheses using LLM reasoning (scenario-specific)
2. **Async task dispatch** -- Launch concurrent worker agents via `dispatch_agent` tool
3. **Monitoring & intervention** -- Check task progress, inject instructions, abort stuck agents
4. **State-aware context** -- Receive dynamic state context each round via `DynamicContextMiddleware`
5. **Structured output** -- Produce a schema-validated final report via `_synthesize_output()`
6. **Termination control** -- End the loop when the LLM emits `<decision>finalize</decision>`

---

## 2. Architecture

### 2.1 SimpleAgentLoop Cycle

`SimpleAgentLoop` (`src/agentm/harness/loops/simple.py`) implements the ReAct loop. The `stream()` method is the primary entry point; `run()` delegates to it.

Each iteration:

```
1. Drain inbox        -- consume injected messages
2. on_llm_start       -- middleware transforms message list
3. LLM call           -- with exponential-backoff retry on transient errors
4. on_llm_end         -- middleware post-processes response
5. Check termination  -- should_terminate(response)
6. Execute tools      -- single: direct call; multiple: asyncio.gather
7. Checkpoint         -- optional persistence
```

Key behaviors:

- **Parallel tool execution**: When the LLM returns multiple tool calls in one response, they run concurrently via `asyncio.gather`.
- **Inbox injection**: Any agent can receive mid-loop messages via `loop.inject(message)`. These are drained as `[Injected message]` before each LLM call.
- **Retry on transient errors**: LLM calls retry with exponential backoff for rate limits (429), server errors (5xx), and network timeouts. Configurable via `retry_max_attempts`, `retry_initial_interval`, `retry_backoff_factor`.

### 2.2 Termination Logic

Termination is controlled by the `should_terminate` callback passed to `SimpleAgentLoop`. The default implementation is `_orchestrator_should_terminate()` in `builder.py`:

```python
_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)

def _orchestrator_should_terminate(response: Any) -> bool:
    content = getattr(response, "content", "") or ""
    match = _DECISION_RE.search(content)
    if match:
        return match.group(1).strip().lower() == "finalize"
    # Fallback: no tool calls = terminate
    return not getattr(response, "tool_calls", None)
```

The LLM emits `<decision>finalize</decision>` when it wants to stop. If the tag is absent, the fallback heuristic treats "no tool calls" as termination. Scenarios can override this via `ScenarioWiring.should_terminate`.

### 2.3 Middleware Stack

Middleware classes extend `MiddlewareBase` and override three hooks: `on_llm_start`, `on_llm_end`, `on_tool_call`. The orchestrator's middleware stack is assembled in `_build_orchestrator_loop()`:

```
1. DynamicContextMiddleware   -- injects dynamic state + round counter into system prompt
2. LoopDetectionMiddleware    -- detects repetitive tool patterns and think-stalls
3. CompressionMiddleware      -- LLM-based summarization when token count exceeds threshold
4. SkillMiddleware            -- injects skill descriptions from MarkdownVault
5. (scenario middleware)      -- any middleware from ScenarioWiring.orchestrator_middleware
6. TrajectoryMiddleware       -- records all LLM calls and tool executions
```

Order matters: `DynamicContextMiddleware` runs first (replaces system prompt each round), `TrajectoryMiddleware` runs last (records the final prepared messages).

---

## 3. Orchestrator Tools

Created by `create_orchestrator_tools()` in `src/agentm/tools/orchestrator.py`. The function captures `AgentRuntime` and `WorkerFactory` in closures and returns a `dict[str, Callable]`.

### dispatch_agent

```python
async def dispatch_agent(
    agent_id: str, task: str, task_type: TaskType, metadata: dict[str, str] | None = None
) -> str
```

- Creates a worker loop via `WorkerLoopFactory.create_worker(agent_id, task_type)`
- Spawns it on `AgentRuntime.spawn()` as an `asyncio.Task`
- **Auto-block optimization**: If this is the only running worker, waits for completion and returns the result directly (saves an LLM roundtrip through `check_tasks`)
- **Concurrency limiting**: When `max_concurrent_workers` is configured, uses an `asyncio.Semaphore` to block until a slot is available

### check_tasks

```python
async def check_tasks(request: str) -> str
```

- Calls `runtime.wait_any()` with a brief timeout to catch freshly completed agents
- Returns JSON with `running`, `completed`, `failed` lists and counts
- Filters to only agents whose `parent_id == "orchestrator"`

### inject_instruction

```python
async def inject_instruction(task_id: str, instruction: str) -> str
```

- Calls `runtime.send(task_id, instruction)` which invokes `loop.inject()` on the target worker
- The instruction is consumed before the worker's next LLM call

### abort_task

```python
async def abort_task(task_id: str, reason: str) -> str
```

- Calls `runtime.abort(task_id, reason)` which cancels the worker's asyncio.Task
- Cascades: also aborts all running children of the aborted agent

---

## 4. State Management

### 4.1 DynamicContextMiddleware + format_context Pattern

The orchestrator does NOT use a graph state or notebook. Instead, dynamic state is injected into the system prompt each round by `DynamicContextMiddleware`:

```python
class DynamicContextMiddleware(MiddlewareBase):
    def __init__(self, format_context_fn: Callable[[], str], base_system_prompt: str, max_rounds: int):
        ...

    async def on_llm_start(self, messages, ctx):
        context_text = self._format_fn()    # scenario closure reads from stores
        # Rebuilds system prompt + appends <current_state> block + round counter
```

The `format_context_fn` is always zero-argument. Scenarios bind their own state via closures during `Scenario.setup()`. This keeps the SDK completely unaware of domain-specific state structures.

Each round, the middleware:
1. Calls `format_context_fn()` to get the current state text
2. Rebuilds the message list with a fresh system prompt
3. Appends a `<current_state>` block and `<round_context>` with round counter
4. Injects last-round urgency warnings when approaching `max_rounds`

### 4.2 ScenarioWiring

The `Scenario` protocol defines how domain-specific behavior is injected:

```python
class Scenario(Protocol):
    @property
    def name(self) -> str: ...
    def setup(self, ctx: SetupContext) -> ScenarioWiring: ...

@dataclass
class ScenarioWiring:
    orchestrator_tools: list[Tool]
    worker_tools: list[Tool]
    format_context: Callable[[], str]
    answer_schemas: dict[str, type[BaseModel]]
    output_schema: type[BaseModel] | None
    hooks: OrchestratorHooks
    should_terminate: Callable[[Any], bool] | None
    orchestrator_middleware: list[Any]
    worker_middleware: list[Any]
```

---

## 5. RCA Example: HypothesisStore + ServiceProfileStore

The RCA scenario (`src/agentm/scenarios/rca/scenario.py`) demonstrates the state management pattern with two independent stores.

### HypothesisStore

Thread-safe, run-scoped store for hypothesis lifecycle management. Orchestrator-only.

```python
@dataclass(frozen=True)
class HypothesisEntry:
    id: str
    description: str
    status: str  # formed|investigating|confirmed|rejected|refined|inconclusive
    evidence: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    parent_id: str | None
```

Exposed to the orchestrator via `update_hypothesis` and `remove_hypothesis` tools.

### ServiceProfileStore

Thread-safe, run-scoped store for cross-agent service knowledge. Shared by both orchestrator and workers.

```python
@dataclass(frozen=True)
class ServiceProfile:
    service_name: str
    is_anomalous: bool
    anomaly_summary: str
    upstream_services: tuple[str, ...]
    downstream_services: tuple[str, ...]
    observations: tuple[ServiceObservation, ...]
    data_sources_queried: tuple[str, ...]
    related_hypothesis_ids: tuple[str, ...]
```

Merge-update semantics: topology fields use set union, observations append, anomaly status can only upgrade (False -> True).

Exposed via `update_service_profile` and `query_service_profile` tools on both orchestrator and worker.

### Context Formatting

The RCA scenario creates a `format_rca_context` closure that reads from both stores:

```python
format_fn = partial(
    format_rca_context,
    profile_store=profile_store,
    hypothesis_store=hypothesis_store,
)
```

This closure is passed as `ScenarioWiring.format_context`, which `DynamicContextMiddleware` calls each round to build the `<current_state>` block.

---

## 6. Structured Output

`SimpleAgentLoop._synthesize_output()` produces the final output after the loop terminates.

**When no `output_schema` is set**: Returns the last AI message content as-is.

**When `output_schema` is set** (e.g., `CausalGraph` for RCA):

1. Creates a `structured_model` via `model.with_structured_output(schema, method="function_calling")`
2. Builds synthesis messages: `output_prompt` (system) + conversation history (non-system) + final instruction
3. Attempts structured output up to `1 + synthesize_retries` times
4. On validation failure: appends the error and raw output as feedback, retries
5. On final failure: falls back to plain LLM call, returns `{"raw_text": ...}`

The retry count is controlled by `OrchestratorHooks.synthesize_max_retries` (default: 2).

---

## 7. Configuration

### OrchestratorHooks

```python
@dataclass
class OrchestratorHooks:
    think_stall_enabled: bool = True
    synthesize_max_retries: int = 2
```

Returned by `ScenarioWiring.hooks`. Controls whether `LoopDetectionMiddleware` is added and the structured output retry budget.

### Key Config Parameters

From `ScenarioConfig.orchestrator`:

| Parameter | Effect |
|-----------|--------|
| `model` | LLM model name for the orchestrator |
| `temperature` | LLM temperature |
| `max_rounds` | Maximum ReAct iterations (used by `DynamicContextMiddleware` for urgency) |
| `tools` | List of tool names to resolve from SDK, scenario, registry, or memory |
| `prompts.system` | Jinja2 template path for the system prompt |
| `retry.max_attempts` | LLM retry attempts on transient errors |
| `retry.initial_interval` | Initial retry delay in seconds |
| `retry.backoff_factor` | Exponential backoff multiplier |
| `loop_detection.threshold` | Repetition count before warning |
| `loop_detection.window_size` | Number of recent AI messages to scan |
| `loop_detection.think_stall_limit` | Consecutive think-only rounds before warning |
| `compression.enabled` | Enable LLM-based context compression |
| `compression.compression_threshold` | Token ratio threshold to trigger compression |
| `compression.preserve_latest_n` | Number of recent messages to keep uncompressed |
| `output.prompt` | Template path for the structured output synthesis prompt |
| `include_think_tool` | Whether to add the `think` tool |
| `disable_tool_binding` | Skip `bind_tools` (for models that handle tools differently) |
| `skills` | Vault paths for SkillMiddleware |

---

## 8. Build Process

`build_agent_system()` in `src/agentm/builder.py` constructs the complete system in four phases:

1. **`_create_platform_resources`** -- ToolRegistry, MarkdownVault, TrajectoryCollector, memory tools, model configs
2. **`_create_worker_infrastructure`** -- `AgentRuntime` + `WorkerLoopFactory` from platform resources and scenario wiring
3. **`_assemble_orchestrator_tools`** -- Resolves tool names from config against SDK tools, scenario tools, memory tools, and the tool registry
4. **`_build_orchestrator_loop`** -- Assembles middleware stack, creates LLM model, builds `SimpleAgentLoop`

The result is an `AgentSystem` that wraps the orchestrator loop and provides `execute()` and `stream()` methods.

---

## 9. Cross-References

- [agent-harness](agent-harness.md) -- `SimpleAgentLoop`, `AgentRuntime`, middleware protocol, `LoopContext`
- [sub-agent](sub-agent.md) -- `WorkerLoopFactory`, worker middleware stack, answer schemas
- [trajectory](trajectory.md) -- `TrajectoryCollector`, `TrajectoryMiddleware`, trajectory recording
- [scenario-protocol](generic-state-wrapper.md) -- Scenario protocol, `build_agent_system()` pipeline
- [memory-vault](memory-vault.md) -- `MarkdownVault`, `SkillMiddleware`
