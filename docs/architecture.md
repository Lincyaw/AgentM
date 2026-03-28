# AgentM Architecture & Workflow

> Auto-generated from source code analysis. Last updated: 2026-03-28

## Table of Contents

- [1. High-Level Overview](#1-high-level-overview)
- [2. Module Map](#2-module-map)
- [3. Core Abstractions](#3-core-abstractions)
- [4. End-to-End Workflow](#4-end-to-end-workflow)
- [5. Data Flow Diagrams](#5-data-flow-diagrams)
- [6. Scenario System](#6-scenario-system)
- [7. Middleware Pipeline](#7-middleware-pipeline)
- [8. Tool System](#8-tool-system)
- [9. Trajectory & Observability](#9-trajectory--observability)
- [10. CLI & Server](#10-cli--server)

---

## 1. High-Level Overview

AgentM is a **multi-agent orchestration framework** built in pure Python (no LangGraph dependency at runtime). It uses an **Orchestrator-Worker** architecture where:

- One **Orchestrator** (LLM-powered) receives a task, reasons about strategy, and dispatches work to **Worker** agents via tool calls.
- Multiple **Workers** (LLM-powered) run concurrently as `asyncio.Task`s, each executing a focused sub-task with their own tool set.
- **Scenarios** (pluggable domain modules) provide all domain-specific behavior: tools, state, context formatting, output schemas.
- The **Harness SDK** provides the infrastructure: agent lifecycle, middleware pipeline, trajectory recording, and communication.

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / Server                            │
│  (typer commands: run, analyze, resume, batch, debug)           │
├─────────────────────────────────────────────────────────────────┤
│                     build_agent_system()                        │
│        ┌──────────────────────────────────────────┐             │
│        │            AgentSystem                    │             │
│        │  ┌──────────────────────────────────┐    │             │
│        │  │  Orchestrator (SimpleAgentLoop)   │    │             │
│        │  │  ┌────────────────────────────┐  │    │             │
│        │  │  │    Middleware Pipeline      │  │    │             │
│        │  │  │  Dynamic Context           │  │    │             │
│        │  │  │  Loop Detection            │  │    │             │
│        │  │  │  Compression               │  │    │             │
│        │  │  │  Skills                    │  │    │             │
│        │  │  │  Trajectory                │  │    │             │
│        │  │  └────────────────────────────┘  │    │             │
│        │  │  Tools: dispatch_agent,           │    │             │
│        │  │    check_tasks, inject, abort,    │    │             │
│        │  │    + scenario tools, think        │    │             │
│        │  └──────────┬───────────────────────┘    │             │
│        │             │ dispatch_agent()            │             │
│        │  ┌──────────▼───────────────────────┐    │             │
│        │  │       AgentRuntime                │    │             │
│        │  │  ┌─────────┐  ┌─────────┐        │    │             │
│        │  │  │Worker A │  │Worker B │  ...    │    │             │
│        │  │  │(Simple  │  │(Simple  │         │    │             │
│        │  │  │AgentLoop│  │AgentLoop│         │    │             │
│        │  │  └─────────┘  └─────────┘        │    │             │
│        │  └──────────────────────────────────┘    │             │
│        └──────────────────────────────────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Scenario   │  │   Vault      │  │ TrajectoryCollector  │   │
│  │   (RCA /     │  │   (Knowledge │  │ (JSONL recording)    │   │
│  │   GP / TA)   │  │    base)     │  │                      │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Map

```
src/agentm/
├── __init__.py              # Entry point → cli.main
├── builder.py               # ★ build_agent_system() — the canonical builder
├── exceptions.py            # AgentMError hierarchy
│
├── harness/                 # ★ Harness SDK — agent infrastructure
│   ├── __init__.py          # Public API re-exports
│   ├── protocols.py         # AgentLoop, Middleware, CheckpointStore, EventHandler
│   ├── types.py             # AgentEvent, AgentResult, AgentStatus, RunConfig, LoopContext
│   ├── runtime.py           # AgentRuntime — multi-agent lifecycle manager
│   ├── handle.py            # AgentHandle — convenience wrapper for spawned agents
│   ├── adapters.py          # TrajectoryEventAdapter — bridges events to trajectory
│   ├── scenario.py          # Scenario protocol, ScenarioWiring, SetupContext, registry
│   ├── tool.py              # Tool dataclass, @tool decorator, tool_from_function
│   ├── middleware.py         # 7 middleware implementations
│   ├── worker_factory.py    # WorkerLoopFactory — creates worker SimpleAgentLoops
│   └── loops/
│       └── simple.py        # ★ SimpleAgentLoop — pure-Python ReAct loop
│
├── config/                  # Configuration loading & schemas
│   ├── schema.py            # Pydantic models: ScenarioConfig, SystemConfig, etc.
│   ├── loader.py            # YAML loading with ${ENV_VAR} substitution
│   └── validator.py         # Config validation
│
├── core/                    # Shared core infrastructure
│   ├── backend.py           # StorageBackend protocol
│   ├── prompt.py            # Jinja2 prompt template loader
│   ├── tool_registry.py     # ToolRegistry — dynamic YAML-based tool binding
│   ├── trajectory.py        # TrajectoryCollector — JSONL event recording
│   ├── trajectory_converter.py  # Event-to-message format converter
│   └── debug_console.py     # Rich terminal UI for debug mode
│
├── models/                  # Shared data types
│   ├── data.py              # OrchestratorHooks
│   ├── state.py             # BaseExecutorState (TypedDict)
│   └── types.py             # TaskType alias
│
├── scenarios/               # ★ Domain-specific scenario plugins
│   ├── __init__.py          # discover() — registers all built-in scenarios
│   ├── rca/                 # Root Cause Analysis scenario
│   │   ├── scenario.py      # RCAScenario: setup() returns wiring
│   │   ├── state.py         # HypothesisDrivenState
│   │   ├── hypothesis_store.py  # In-memory hypothesis management
│   │   ├── service_profile.py   # ServiceProfileStore
│   │   ├── notebook.py      # DiagnosticNotebook (immutable append-only)
│   │   ├── data.py          # RCA data structures
│   │   ├── enums.py         # Phase, HypothesisStatus enums
│   │   ├── formatters.py    # format_rca_context()
│   │   ├── compression.py   # RCA-specific compression
│   │   ├── answer_schemas.py  # ScoutAnswer, VerifyAnswer, DeepAnalyzeAnswer
│   │   └── output.py        # CausalGraph (final structured output)
│   ├── trajectory_analysis/ # Trajectory Analysis scenario
│   │   ├── scenario.py      # TrajectoryAnalysisScenario
│   │   ├── answer_schemas.py  # AnalyzeAnswer, CritiqueAnswer
│   │   └── output.py        # AnalysisReport
│   └── general_purpose/     # General Purpose scenario
│       ├── scenario.py      # GeneralPurposeScenario
│       └── answer_schemas.py  # GeneralAnswer
│
├── tools/                   # Tool implementations
│   ├── _shared.py           # Shared utilities
│   ├── think.py             # think tool (scratch pad for LLM reasoning)
│   ├── orchestrator.py      # dispatch_agent, check_tasks, inject, abort
│   ├── case_data.py         # Load case data for analysis
│   ├── duckdb_sql.py        # DuckDB SQL query tool
│   ├── memory.py            # Trajectory/checkpoint reading tools
│   ├── trajectory_reader.py # Structured trajectory querying (jq_query)
│   ├── observability/       # Observability data access tools
│   │   ├── _core.py         # set_data_directory, parquet management
│   │   ├── _metrics.py      # OHLC metrics queries
│   │   ├── _traces.py       # Trace/span graph queries
│   │   ├── _logs.py         # Log search (abnormal/normal)
│   │   ├── _deployment.py   # Deployment graph
│   │   └── _builders.py     # Query builders
│   └── vault/               # Knowledge vault (Obsidian-like)
│       ├── store.py         # MarkdownVault (DuckDB-backed)
│       ├── tools.py         # 10 vault tool functions
│       ├── parser.py        # Markdown/frontmatter parser
│       ├── schema.py        # Note schema definitions
│       ├── search.py        # keyword/semantic/hybrid search
│       ├── graph.py         # Backlinks, traverse, lint
│       └── mcp_server.py    # MCP server for external vault access
│
├── backends/                # Storage backends
│   ├── filesystem.py        # FilesystemBackend
│   └── composite.py         # CompositeBackend (prefix routing)
│
├── agents/                  # Specialized agent implementations
│   └── eval_agent.py        # Eval agent for batch evaluation
│
├── cli/                     # CLI commands
│   ├── main.py              # typer app: run, analyze, resume, batch, debug
│   ├── run.py               # run_trajectory_analysis, run_investigation_headless
│   ├── batch.py             # Batch analysis orchestration
│   ├── debug.py             # Trajectory file analysis
│   └── export_eval.py       # Export evaluation results
│
└── server/                  # Web dashboard
    └── app.py               # FastAPI + WebSocket for real-time monitoring
```

---

## 3. Core Abstractions

### 3.1 Protocols (Interfaces)

```
┌───────────────────────────────────────────────────────┐
│                     Protocols                          │
│                                                       │
│  AgentLoop           Middleware          EventHandler  │
│  ┌──────────┐       ┌──────────────┐   ┌───────────┐ │
│  │ run()    │       │ on_llm_start │   │ on_event  │ │
│  │ stream() │       │ on_llm_end   │   └───────────┘ │
│  │ inject() │       │ on_tool_call │                  │
│  └──────────┘       └──────────────┘                  │
│                                                       │
│  CheckpointStore     Scenario                         │
│  ┌──────────────┐   ┌──────────────┐                  │
│  │ save()       │   │ name         │                  │
│  │ load()       │   │ setup(ctx)   │                  │
│  │ list()       │   │  → Wiring    │                  │
│  └──────────────┘   └──────────────┘                  │
└───────────────────────────────────────────────────────┘
```

| Protocol | Purpose | Key Implementations |
|----------|---------|-------------------|
| `AgentLoop` | Core agent execution cycle: LLM → tools → repeat | `SimpleAgentLoop` |
| `Middleware` | Hook into LLM calls and tool execution | `BudgetMiddleware`, `CompressionMiddleware`, `LoopDetectionMiddleware`, `TrajectoryMiddleware`, `DedupMiddleware`, `SkillMiddleware`, `DynamicContextMiddleware` |
| `EventHandler` | Receive streaming events from agents | `TrajectoryEventAdapter` |
| `CheckpointStore` | Persist/recover agent state | (protocol defined, implementations pending) |
| `Scenario` | Domain-specific plugin contract | `RCAScenario`, `TrajectoryAnalysisScenario`, `GeneralPurposeScenario` |

### 3.2 Key Data Types

```python
# Agent lifecycle
AgentStatus: RUNNING | COMPLETED | FAILED | ABORTED

# Per-run configuration
RunConfig(max_steps, timeout, thread_id, metadata)

# Agent execution outcome
AgentResult(agent_id, status, output, error, duration_seconds, steps, tool_calls)

# Streaming event
AgentEvent(type, agent_id, data, step, timestamp)
# Event types: llm_start, llm_end, tool_start, tool_end, inject, complete, error

# Middleware context (frozen/immutable)
LoopContext(agent_id, step, max_steps, tool_call_count, metadata)

# Runtime status snapshot
AgentInfo(agent_id, status, parent_id, current_step, started_at, metadata, result)
```

### 3.3 ScenarioWiring — The Plugin Contract

When a `Scenario.setup(ctx)` is called, it returns a `ScenarioWiring` that tells the SDK everything it needs:

```python
ScenarioWiring(
    orchestrator_tools: list[Tool],       # Tools only for orchestrator
    worker_tools: list[Tool],             # Tools injected into workers
    format_context: () -> str,            # Dynamic state context for LLM
    answer_schemas: dict[str, BaseModel], # Structured output per task_type
    output_schema: BaseModel | None,      # Final orchestrator output schema
    hooks: OrchestratorHooks,             # Behavior customization
    should_terminate: (response) -> bool, # Custom termination logic
    orchestrator_middleware: list[Any],    # Extra middleware for orchestrator
    worker_middleware: list[Any],          # Extra middleware for workers
)
```

---

## 4. End-to-End Workflow

### 4.1 System Construction (`build_agent_system`)

This is the single canonical entry point. Here is the full construction flow:

```
build_agent_system(scenario_name, scenario_config, system_config)
│
├── 1. discover() — register all built-in scenarios
│     (rca, trajectory_analysis, general_purpose)
│
├── 2. get_scenario(name) — look up Scenario implementation
│
├── 3. Create platform resources
│     ├── ToolRegistry — load tool definitions from YAML files
│     ├── MarkdownVault — initialize knowledge base
│     ├── vault_tools — 10 vault operations (read, write, search, etc.)
│     ├── memory_tools — read_trajectory, get_checkpoint_history, jq_query, load_case_data
│     └── TrajectoryCollector — JSONL event recording
│
├── 4. scenario.setup(SetupContext) → ScenarioWiring
│     SetupContext provides: vault, trajectory, tool_registry
│
├── 5. Create AgentRuntime (manages worker agent lifecycle)
│
├── 6. Create WorkerLoopFactory
│     ├── Injects: scenario worker tools
│     ├── Injects: worker middleware (scenario + SkillMiddleware)
│     ├── Injects: answer_schemas for structured worker output
│     └── Injects: TrajectoryCollector reference
│
├── 7. Create orchestrator tools
│     ├── SDK tools: dispatch_agent, check_tasks, inject_instruction, abort_task
│     ├── Scenario tools: (e.g., update_hypothesis, query_service_profile)
│     ├── Registry tools: (YAML-defined, vault tools, memory tools)
│     └── Always: think tool
│
├── 8. Build orchestrator middleware stack (in order)
│     ├── DynamicContextMiddleware  — injects state + round info
│     ├── LoopDetectionMiddleware   — detect think-stalls & loops
│     ├── CompressionMiddleware     — LLM-based history summarization
│     ├── SkillMiddleware           — inject vault skill descriptions
│     ├── Scenario middleware       — custom per-scenario
│     └── TrajectoryMiddleware      — record all events
│
├── 9. Build LLM model (ChatOpenAI or ChatAnthropic)
│     └── bind_tools() with all orchestrator tools
│
└── 10. Return AgentSystem(
          loop=SimpleAgentLoop,
          runtime=AgentRuntime,
          trajectory=TrajectoryCollector
       )
```

### 4.2 Execution Flow

```
AgentSystem.execute(input_data)
│
└── SimpleAgentLoop.stream(input_text)
    │
    │   ┌─────────────── REACT LOOP ─────────────────┐
    │   │                                              │
    │   │  1. Drain inbox (injected messages)          │
    │   │  2. Middleware: on_llm_start(messages)        │
    │   │     ├── DynamicContext: replace system msg    │
    │   │     │   with state snapshot + round info      │
    │   │     ├── LoopDetection: inject warnings        │
    │   │     ├── Compression: summarize old history    │
    │   │     ├── Skills: inject vault skill catalog    │
    │   │     └── Trajectory: record llm_start event   │
    │   │                                              │
    │   │  3. LLM.ainvoke(prepared_messages)            │
    │   │                                              │
    │   │  4. Middleware: on_llm_end(response)          │
    │   │     └── Trajectory: record tool_call events   │
    │   │                                              │
    │   │  5. Check termination:                       │
    │   │     ├── <decision>finalize</decision> tag?    │
    │   │     └── No tool_calls?                       │
    │   │     If terminate → synthesize_output() → done │
    │   │                                              │
    │   │  6. Execute tool calls:                      │
    │   │     For each tool_call:                      │
    │   │     ├── Build middleware chain (wrapping)     │
    │   │     ├── Middleware: on_tool_call              │
    │   │     │   ├── Dedup: cache check/populate      │
    │   │     │   └── Trajectory: record tool_result   │
    │   │     └── Tool.ainvoke(args) → result string   │
    │   │                                              │
    │   │  7. Append tool results to messages          │
    │   │  8. step += 1, repeat                        │
    │   │                                              │
    │   └─────────────────────────────────────────────┘
    │
    └── yield AgentEvent for each stage
```

### 4.3 Worker Dispatch Flow

When the orchestrator calls `dispatch_agent`:

```
Orchestrator LLM decides to call dispatch_agent(agent_id, task, task_type)
│
├── dispatch_agent() (closure over runtime + worker_factory)
│   │
│   ├── worker_factory.create_worker(agent_id, task_type)
│   │   ├── Build tools: registry tools + scenario worker tools + think
│   │   ├── Build system prompt: base template + task_type overlay (Jinja2)
│   │   ├── Build middleware: [Budget, LoopDetection, Compression, Trajectory, Dedup]
│   │   ├── Resolve output_schema from answer_schemas[task_type]
│   │   └── Return SimpleAgentLoop(model, tools, prompt, middleware, output_schema)
│   │
│   ├── runtime.spawn(unique_id, loop, input, parent_id="orchestrator")
│   │   ├── Create _AgentEntry
│   │   ├── Record task_dispatch trajectory event
│   │   └── asyncio.create_task(_run_agent)
│   │       └── Iterate loop.stream() → forward events → record completion
│   │
│   ├── Auto-block logic:
│   │   ├── If only 1 worker running → await completion, return result directly
│   │   └── If multiple workers → return immediately with "running" status
│   │
│   └── Return JSON: {task_id, agent_id, status, result?}
│
├── check_tasks() — Orchestrator polls for completed results
│   ├── Wait briefly for any agent to complete (5s timeout)
│   ├── Collect all status: running[], completed[], failed[]
│   └── Return JSON summary
│
├── inject_instruction(task_id, instruction) — Send message to running worker
│   └── runtime.send() → worker.inject() → message appears in next LLM call
│
└── abort_task(task_id, reason) — Kill a worker
    └── runtime.abort() → cancel asyncio.Task → cascade to children
```

---

## 5. Data Flow Diagrams

### 5.1 Overall Data Flow

```
                   ┌──────────┐
                   │  Config  │
                   │  YAMLs   │
                   └────┬─────┘
                        │ load
                        ▼
┌────────┐    ┌───────────────────┐    ┌──────────────┐
│  CLI   │───▶│ build_agent_system│───▶│  AgentSystem  │
│ input  │    └───────────────────┘    │              │
└────────┘              │              │  .execute()  │
                        │              │  .stream()   │
              ┌─────────┴──────┐       └──────┬───────┘
              │                │              │
        ┌─────▼─────┐  ┌──────▼─────┐  ┌─────▼──────────┐
        │  Scenario  │  │   Vault    │  │  Orchestrator  │
        │  .setup()  │  │ (knowledge │  │  (SimpleAgent  │
        │  → Wiring  │  │  base)     │  │   Loop)        │
        └────────────┘  └────────────┘  └─────┬──────────┘
                                              │
                            ┌─────────────────┼─────────────────┐
                            │                 │                 │
                     ┌──────▼──────┐   ┌──────▼──────┐   ┌─────▼─────┐
                     │   Worker A  │   │   Worker B  │   │  Worker C │
                     │  (async)    │   │  (async)    │   │  (async)  │
                     └──────┬──────┘   └──────┬──────┘   └─────┬─────┘
                            │                 │                 │
                            ▼                 ▼                 ▼
                     ┌──────────────────────────────────────────────┐
                     │         Tools (observability, vault,         │
                     │         DuckDB SQL, case data, ...)         │
                     └──────────────────────────────────────────────┘
                                              │
                                              ▼
                     ┌──────────────────────────────────────────────┐
                     │          External Data Sources               │
                     │  (Parquet files, knowledge vault, DB)        │
                     └──────────────────────────────────────────────┘
```

### 5.2 Message Flow Within SimpleAgentLoop

```
    system_prompt (static)
         │
         ▼
    ┌─────────────┐     ┌─────────────────────────────────────┐
    │  messages[]  │◄────│  DynamicContextMiddleware            │
    │  (mutable)   │     │  replaces system msg each iteration │
    └──────┬──────┘     │  with: base_prompt + state_context  │
           │            │  + round_info + urgency warnings     │
           ▼            └─────────────────────────────────────┘
    ┌──────────────┐
    │  inbox[]     │  ← injected messages from runtime.send()
    │  (drained)   │    prepended as [Injected message] role=human
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────┐
    │  Middleware.on_llm_start │  (chain: each transforms messages)
    │  → prepared messages     │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  LLM.ainvoke(prepared)   │  → response (AIMessage)
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Middleware.on_llm_end   │  (chain: each inspects/modifies response)
    └──────────┬───────────────┘
               │
               ├─── response has tool_calls?
               │       YES                          NO (or <decision>finalize</decision>)
               │                                          │
               ▼                                          ▼
    ┌──────────────────────┐                  ┌──────────────────────┐
    │  For each tool_call: │                  │  _synthesize_output  │
    │  ├── Build mw chain  │                  │  ├── If output_schema│
    │  ├── mw.on_tool_call │                  │  │   → structured    │
    │  │   (wrapping)      │                  │  │     output via    │
    │  │   ├── Dedup check │                  │  │     with_struct   │
    │  │   ├── Trajectory  │                  │  │     ured_output() │
    │  │   └── call_next() │                  │  └── Else: raw text │
    │  ├── tool.ainvoke()  │                  └──────────┬───────────┘
    │  └── Append result   │                             │
    │       to messages    │                             ▼
    └──────────┬───────────┘                     AgentResult
               │                                 (output, status,
               └──── step++, loop back ────      steps, tool_calls)
```

### 5.3 AgentRuntime Worker Lifecycle

```
    runtime.spawn(agent_id, loop, input)
         │
         ├── Create _AgentEntry(agent_id, loop, status=RUNNING)
         ├── Record trajectory: task_dispatch
         └── asyncio.create_task(_run_agent)
                  │
                  ▼
         ┌──────────────────────────────────────┐
         │  _run_agent(entry, input, config)     │
         │  │                                    │
         │  │  async for event in loop.stream(): │
         │  │    ├── Update entry.current_step   │
         │  │    ├── Forward to event_handler    │
         │  │    └── On "complete": save result  │
         │  │                                    │
         │  │  On success:                       │
         │  │    entry.status = COMPLETED        │
         │  │    Record trajectory: task_complete │
         │  │                                    │
         │  │  On CancelledError:                │
         │  │    entry.status = ABORTED          │
         │  │    Record trajectory: task_abort   │
         │  │                                    │
         │  │  On Exception:                     │
         │  │    entry.status = FAILED           │
         │  │    Record trajectory: task_fail    │
         │  │                                    │
         │  └── Finally:                         │
         │       entry.done_event.set()          │
         │       _cascade_children(agent_id)     │
         └──────────────────────────────────────┘

    runtime.wait(agent_id) — blocks on done_event
    runtime.wait_any(ids)  — returns first completed
    runtime.abort(agent_id) — cancels task, cascades to children
    runtime.send(to, msg)   — loop.inject(msg)
```

---

## 6. Scenario System

### 6.1 Registration & Discovery

```python
# scenarios/__init__.py — discover() is called once at startup
def discover():
    register_rca()       # → register_scenario(RCAScenario())
    register_ta()        # → register_scenario(TrajectoryAnalysisScenario())
    register_gp()        # → register_scenario(GeneralPurposeScenario())

# Module-level registry in scenario.py
_SCENARIOS: dict[str, Scenario] = {}
register_scenario(scenario)  # stores by scenario.name
get_scenario(name)            # retrieves by name
```

### 6.2 Scenario Comparison

| Aspect | RCA (`hypothesis_driven`) | Trajectory Analysis | General Purpose |
|--------|--------------------------|--------------------|-----------------|
| **Orchestrator tools** | `update_hypothesis`, `remove_hypothesis`, `update_service_profile`, `query_service_profile` | (none) | (none) |
| **Worker tools** | `update_service_profile` (worker), `query_service_profile` (worker) | (none) | (none) |
| **State stores** | `HypothesisStore`, `ServiceProfileStore` | (none) | (none) |
| **format_context** | `format_rca_context` (hypothesis board + service profiles) | `_empty_context` | `_empty_context` |
| **answer_schemas** | `scout: ScoutAnswer`, `deep_analyze: DeepAnalyzeAnswer`, `verify: VerifyAnswer` | `analyze: AnalyzeAnswer`, `critique: CritiqueAnswer` | `execute: GeneralAnswer` |
| **output_schema** | `CausalGraph` | `AnalysisReport` | None |
| **hooks** | think_stall=3, skip_context_on_think | defaults | defaults |

### 6.3 RCA Scenario Deep Dive

The RCA (Root Cause Analysis) scenario is the most complex:

```
RCAScenario.setup(ctx)
│
├── Create HypothesisStore (in-memory dict of hypotheses)
├── Create ServiceProfileStore (in-memory service health tracking)
│
├── Build orchestrator tools (closures over stores + trajectory):
│   ├── update_hypothesis(id, description, status, evidence, parent_id)
│   ├── remove_hypothesis(id)
│   ├── update_service_profile(service_name, is_anomalous, ...)
│   └── query_service_profile(request, service_names, anomalous_only)
│
├── Build worker tools (sync versions, same LLM-facing names):
│   ├── update_service_profile(...)  — workers also update shared profiles
│   └── query_service_profile(...)   — workers also query shared profiles
│
├── format_context = partial(format_rca_context,
│       profile_store=..., hypothesis_store=...)
│   → Returns current hypothesis board + anomalous services snapshot
│
└── Return ScenarioWiring(
        orchestrator_tools, worker_tools,
        format_context, answer_schemas, output_schema=CausalGraph, hooks)
```

The orchestrator's dynamic context (injected before every LLM call):

```xml
<current_state>
  [Hypothesis board: id, status, description, evidence]
  [Anomalous services: name, anomaly summary]
  [Healthy services: name]
</current_state>

<round_context>
  Round: 5/20
  <!-- urgency warnings when nearing limit -->
</round_context>

Based on the current state, my next action:
```

---

## 7. Middleware Pipeline

### 7.1 Three Hook Points

```
               on_llm_start              on_llm_end
messages ─────────┬────────── LLM ──────────┬────────── response
                  │                         │
     Sequential chain:              Sequential chain:
     mw1 → mw2 → mw3               mw1 → mw2 → mw3
     (each transforms               (each transforms
      messages list)                  response)


                   on_tool_call
tool_name, args ──────┬───────── actual tool
                      │
          Wrapping chain (onion model):
          mw3( mw2( mw1( actual_call ) ) )
          Can: pass-through, short-circuit,
               modify args, transform output
```

### 7.2 Middleware Implementations

| # | Middleware | Hook | Behavior |
|---|-----------|------|----------|
| 1 | **BudgetMiddleware** | `on_llm_start` | Injects urgency warnings when step/tool budgets run low. Sets `exhausted` flag at limit. |
| 2 | **CompressionMiddleware** | `on_llm_start` | When token count exceeds threshold, replaces older messages with LLM-generated summary. Preserves latest N messages. |
| 3 | **LoopDetectionMiddleware** | `on_llm_start` | Detects: (a) think-stall (N consecutive think-only rounds), (b) exact-match loops (same tool+args repeated 5+ times). Injects warning messages. |
| 4 | **TrajectoryMiddleware** | all 3 hooks | Records `llm_start`, `tool_call`, `llm_end`, `tool_result` events to TrajectoryCollector. |
| 5 | **DedupMiddleware** | `on_llm_start` + `on_tool_call` | Caches tool results (OrderedDict with FIFO eviction). Returns cached result on duplicate call. Injects dedup warnings. Excludes `think` tool. |
| 6 | **SkillMiddleware** | `on_llm_start` | Loads skill descriptions from vault. Injects `<skills>` XML section into system message with instructions to load full skills via `vault_read`. |
| 7 | **DynamicContextMiddleware** | `on_llm_start` | Replaces system message with static base prompt. Appends dynamic state context + round info as AI prefill message. Adds urgency on last rounds. |

### 7.3 Orchestrator vs Worker Middleware Stacks

**Orchestrator** (built in `build_agent_system`):
```
1. DynamicContextMiddleware    ← manages state + round context
2. LoopDetectionMiddleware     ← think-stall + loop detection
3. CompressionMiddleware       ← context window management
4. SkillMiddleware             ← vault skill injection
5. [Scenario middleware]       ← custom per-scenario
6. TrajectoryMiddleware        ← event recording
```

**Worker** (built in `WorkerLoopFactory.create_worker`):
```
1. [Scenario worker middleware] ← custom per-scenario
2. BudgetMiddleware            ← step/tool budget enforcement
3. LoopDetectionMiddleware     ← think-stall + loop detection
4. CompressionMiddleware       ← context window management
5. TrajectoryMiddleware        ← event recording
6. DedupMiddleware             ← tool call caching
```

---

## 8. Tool System

### 8.1 Tool Abstraction

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema derived from type hints
    func: Callable[..., Any]

    async ainvoke(args) -> str      # Execute, normalize to string
    to_openai_schema() -> dict      # For model.bind_tools()
```

Tools are created three ways:
- **`@tool` decorator**: Turn any function into a Tool
- **`tool_from_function(func)`**: Factory function for existing functions (incl. `functools.partial`)
- **`ToolRegistry.load_from_yaml()`**: Load from YAML definitions with module/function references

### 8.2 Tool Categories

```
SDK Tools (built by create_orchestrator_tools):
├── dispatch_agent     — spawn a worker agent
├── check_tasks        — poll worker status/results
├── inject_instruction — send message to running worker
└── abort_task         — kill a running worker

Scenario Tools (provided via ScenarioWiring):
├── update_hypothesis     (RCA)
├── remove_hypothesis     (RCA)
├── update_service_profile (RCA, orch + worker versions)
└── query_service_profile  (RCA, orch + worker versions)

Registry Tools (loaded from YAML or code):
├── Vault: vault_read, vault_write, vault_edit, vault_delete,
│          vault_rename, vault_list, vault_search, vault_backlinks,
│          vault_traverse, vault_lint
├── Observability: query_metrics_ohlc_*, query_trace_stats_*,
│                  get_service_call_graph_*, get_span_call_graph_*,
│                  get_deployment_graph, search_logs_*
├── Memory: read_trajectory, get_checkpoint_history, jq_query, load_case_data
└── DuckDB: duckdb_sql (SQL queries on registered parquet tables)

Built-in:
└── think — scratch pad for LLM reasoning (always available, excluded from dedup)
```

### 8.3 ToolRegistry

```
ToolRegistry
├── register(name, func, config_schema) → ToolDefinition
├── get(name) → ToolDefinition
├── has(name) → bool
└── load_from_yaml(path)
    └── For each tool entry:
        ├── importlib.import_module(tool["module"])
        ├── getattr(module, tool["function"])
        └── register(name, func, parameters)

ToolDefinition
├── create_tool(**config) → Tool
│   ├── Pop "description" from config
│   ├── functools.partial(func, **config) if config
│   └── tool_from_function(bound, name, description)
└── Parameters derived from function type hints via pydantic TypeAdapter
```

---

## 9. Trajectory & Observability

### 9.1 TrajectoryCollector

Records all execution events as a JSONL file:

```
{output_dir}/{run_id}.jsonl

Line 1: {"_meta": {"run_id": "...", "thread_id": "...", "checkpoint_db": "..."}}
Line 2: TrajectoryEvent JSON
Line 3: TrajectoryEvent JSON
...
```

Each `TrajectoryEvent`:
```json
{
  "run_id": "rca-20260328-120000-abc12345",
  "seq": 42,
  "timestamp": "2026-03-28T12:00:42.123456",
  "agent_path": ["orchestrator", "worker-scout-abc"],
  "node_name": "",
  "event_type": "tool_result",
  "data": {"tool_name": "query_metrics_ohlc_abnormal", "result": "..."},
  "task_id": "worker-scout-abc",
  "metadata": {"hypothesis_id": "H1"},
  "parent_seq": null
}
```

Event types recorded:
- **From middleware**: `llm_start`, `tool_call`, `llm_end`, `tool_result`
- **From runtime**: `task_dispatch`, `task_complete`, `task_fail`, `task_abort`
- **From scenario tools**: `hypothesis_update`

### 9.2 Listener System

TrajectoryCollector supports push-based listeners:

```
TrajectoryCollector
├── add_listener(callback) — register async or sync callback
├── record() → notify all listeners
└── record_sync() → notify via sync path (for tool functions)

Listeners:
├── DebugConsole.on_trajectory_event → Rich terminal UI updates
├── Broadcaster (WebSocket) → real-time dashboard
└── EvalTracker → batch evaluation monitoring
```

### 9.3 Debug Console

Rich terminal UI with three panels:
```
┌──────────── Agent Status ──────────────┐
│ Task ID    Agent      Status  Duration │
│ abc12345   scout-1    ✓ done    2.3s   │
│ def67890   verify-1   ● run      -     │
├──────────── Tool Timeline ─────────────┤
│ Time     Agent              Tool       │
│ 12:00:42 orch/scout-1      metrics    │
│ 12:00:43 orch/verify-1     traces     │
├──────────── Hypothesis Board ──────────┤
│ ID   Status         Description        │
│ H1   investigating  DB connection pool │
│ H2   rejected       Network timeout    │
└────────────────────────────────────────┘
```

---

## 10. CLI & Server

### 10.1 CLI Commands

```bash
agentm                         # Show help
agentm analyze <trajectories>  # Trajectory analysis
agentm analyze-batch <config>  # Batch trajectory analysis
agentm resume <trajectory>     # Resume interrupted investigation
agentm debug <trajectory>      # Analyze trajectory JSONL file
agentm export-result <traj>    # Export single case results
agentm export-batch <dir>      # Batch export results
```

### 10.2 Web Dashboard

```
FastAPI app (server/app.py)
├── GET  /              → HTML dashboard (static/index.html)
├── WS   /ws            → WebSocket for real-time events
│   ├── On connect: replay all trajectory events
│   ├── On connect: send eval snapshot (if active)
│   └── Live: broadcast trajectory + eval events
├── GET  /api/topology  → Agent configuration info
├── GET  /api/eval/status    → Eval tracker summary
├── GET  /api/eval/samples   → Paginated sample list
├── GET  /api/eval/samples/{id}      → Sample detail + ground truth
└── GET  /api/eval/samples/{id}/events → Trajectory events for sample
```

### 10.3 Configuration Structure

```yaml
# system.yaml — global configuration
models:
  gpt-4o:
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_BASE_URL:}
    rate_limit: {requests_per_second: 5, max_bucket_size: 10}
    provider: openai  # or "anthropic"
storage:
  checkpointer: {backend: sqlite, url: ./checkpoints.db}
  store: {backend: sqlite, url: ./store.db}
recovery: {mode: manual}
debug:
  trajectory: {enabled: true, output_dir: ./trajectories}
  console_live: false

# scenario.yaml — per-scenario configuration
system:
  type: hypothesis_driven
orchestrator:
  model: gpt-4o
  temperature: 0.7
  prompts: {system: prompts/orchestrator.j2}
  tools: [dispatch_agent, check_tasks, update_hypothesis, ...]
  max_rounds: 20
  output: {prompt: prompts/output.j2, schema_name: CausalGraph}
  compression: {enabled: true, compression_threshold: 0.8}
  skills: [skill/diagnose-sql, skill/trace-analysis]
agents:
  worker:
    model: gpt-4o-mini
    temperature: 0
    prompt: prompts/worker.j2
    tools: [query_metrics_ohlc_abnormal, search_logs_abnormal, ...]
    task_type_prompts: {scout: prompts/scout.j2, verify: prompts/verify.j2}
    execution: {max_steps: 20, timeout: 120}
    compression: {enabled: true}
```

---

## Appendix: Key Design Decisions

1. **No LangGraph at runtime**: The entire execution engine is pure Python (`SimpleAgentLoop` + `asyncio`). LangChain is only used for LLM client abstraction (`ChatOpenAI`/`ChatAnthropic`).

2. **Scenario as plugin**: Domain logic is completely isolated from SDK infrastructure via the `Scenario` protocol. Adding a new domain means implementing `setup()` → `ScenarioWiring`.

3. **Middleware over inheritance**: All cross-cutting concerns (budget, compression, dedup, trajectory, skills) are composable middleware with three clean hook points.

4. **Shared mutable stores**: RCA workers and orchestrator share `ServiceProfileStore` — workers write findings, orchestrator reads them. No message passing needed for shared state.

5. **Auto-block dispatch**: `dispatch_agent` automatically waits for single-worker completion to save an LLM roundtrip. Multiple concurrent workers return immediately.

6. **Structured output with retry**: `_synthesize_output` tries `with_structured_output` up to N times, with error feedback. Falls back to plain LLM on final failure.

7. **Termination via XML tag**: Orchestrator signals completion with `<decision>finalize</decision>` in its response. Falls back to "no tool calls = done" when tag is absent.
