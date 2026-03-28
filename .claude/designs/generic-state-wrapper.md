# Design: Scenario Protocol

**Status**: CURRENT
**Last Updated**: 2026-03-28

---

## Overview

The Scenario Protocol is the **single extension point** for plugging domain-specific behavior into the AgentM SDK. A Scenario encapsulates everything that differs between use-cases -- tools, context formatting, schemas, hooks, termination logic, and middleware -- while the SDK handles all platform concerns (runtime, build pipeline, trajectory, configuration).

```
SDK (shared)                          Scenario (domain-specific)
  build_agent_system()                  Scenario.setup(ctx) -> ScenarioWiring
  SimpleAgentLoop                       orchestrator_tools, worker_tools
  AgentRuntime                          format_context (closures)
  WorkerLoopFactory                     answer_schemas, output_schema
  Middleware stack                      hooks, should_terminate
  TrajectoryCollector                   orchestrator_middleware, worker_middleware
```

The core contract: a Scenario provides a `name` and a `setup()` method that receives platform resources and returns a `ScenarioWiring` dataclass. The SDK's `build_agent_system()` consumes this wiring to assemble the complete agent system.

---

## Scenario Protocol

Defined in `src/agentm/harness/scenario.py`:

```python
@runtime_checkable
class Scenario(Protocol):
    @property
    def name(self) -> str: ...

    def setup(self, ctx: SetupContext) -> ScenarioWiring: ...
```

Only two requirements:
1. **`name`** -- a string identifier matching the `system.type` in scenario YAML config (e.g. `"hypothesis_driven"`, `"trajectory_analysis"`, `"general_purpose"`).
2. **`setup(ctx)`** -- receives platform resources, returns all domain-specific wiring. This is where the scenario creates its stores, binds tools, and configures hooks.

Scenarios are plain classes (no inheritance required). The `@runtime_checkable` decorator enables `isinstance` checks but is not relied upon at build time.

---

## ScenarioWiring

`ScenarioWiring` is the return type of `setup()`. Every field has a sensible default, so minimal scenarios need only set what they use.

```python
@dataclass
class ScenarioWiring:
    # Tools
    orchestrator_tools: list[Tool] = []
    worker_tools: list[Tool] = []

    # Dynamic context (zero-arg callable; scenario binds state via closures)
    format_context: Callable[[], str] = _empty_context

    # Structured output
    answer_schemas: dict[str, type[BaseModel]] = {}
    output_schema: type[BaseModel] | None = None

    # Behavior customization
    hooks: OrchestratorHooks = OrchestratorHooks()

    # Custom termination (None = default <decision> tag parser)
    should_terminate: Callable[[Any], bool] | None = None

    # Scenario-specific middleware (appended after SDK middleware)
    orchestrator_middleware: list[Any] = []
    worker_middleware: list[Any] = []
```

### Field details

| Field | Purpose | Consumed by |
|-------|---------|-------------|
| `orchestrator_tools` | Domain-specific tools for the orchestrator (e.g. `update_hypothesis`, `query_service_profile`) | `_assemble_orchestrator_tools()` -- merged with SDK tools and registry tools |
| `worker_tools` | Extra tools injected into every worker (e.g. `update_service_profile`) | `WorkerLoopFactory` via `extra_tools` |
| `format_context` | Zero-arg callable returning dynamic state as a string. Injected into the system prompt before each LLM call by `DynamicContextMiddleware` | `_build_orchestrator_loop()` |
| `answer_schemas` | Map of task_type to Pydantic model. Workers use these for structured output based on the dispatched `task_type` | `WorkerLoopFactory` via `answer_schemas` |
| `output_schema` | Pydantic model for the orchestrator's final structured output. When set, `SimpleAgentLoop._synthesize_output()` uses `with_structured_output()` | `_build_orchestrator_loop()` |
| `hooks` | `OrchestratorHooks(think_stall_enabled, synthesize_max_retries)`. Controls loop detection and synthesis retry behavior | `_build_orchestrator_loop()` |
| `should_terminate` | Custom termination predicate. Receives the LLM response, returns `True` to stop the loop. When `None`, the SDK default parses `<decision>finalize</decision>` tags | `_build_orchestrator_loop()` |
| `orchestrator_middleware` | Extra middleware appended to the orchestrator's stack (after SDK-provided middleware) | `_build_orchestrator_loop()` |
| `worker_middleware` | Extra middleware prepended to every worker's stack | `_create_worker_infrastructure()` |

### Closure pattern for format_context

The `format_context` callable is always zero-arg. Scenarios bind their own mutable state via closures in `setup()`:

```python
# From RCAScenario.setup()
hypothesis_store = HypothesisStore()
profile_store = ServiceProfileStore()

format_fn = partial(
    format_rca_context,
    profile_store=profile_store,
    hypothesis_store=hypothesis_store,
)

return ScenarioWiring(format_context=format_fn, ...)
```

The SDK never needs to know about `HypothesisStore` or `ServiceProfileStore` -- it only calls `format_fn()` and injects the returned string.

---

## SetupContext

Platform resources the SDK provides to the scenario during `setup()`:

```python
@dataclass(frozen=True)
class SetupContext:
    vault: Any | None           # MarkdownVault instance
    trajectory: Any | None      # TrajectoryCollector instance
    tool_registry: Any          # ToolRegistry with all loaded tools
```

Scenarios use these to:
- Read skills from the vault (`vault.read(path)`)
- Record custom trajectory events (`trajectory.record(...)`)
- Check or use registered tools from YAML definitions

The `SetupContext` is frozen -- scenarios cannot modify platform resources, only read them.

---

## Registration

### register_scenario()

Module-level registration function:

```python
_SCENARIOS: dict[str, Scenario] = {}

def register_scenario(scenario: Scenario) -> None:
    _SCENARIOS[scenario.name] = scenario

def get_scenario(name: str) -> Scenario:
    if name not in _SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Available: ...")
    return _SCENARIOS[name]

def list_scenarios() -> list[str]:
    return list(_SCENARIOS)
```

### discover()

All built-in scenarios register through `src/agentm/scenarios/__init__.py`:

```python
_discovered = False

def discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True

    from agentm.scenarios.rca import register as register_rca
    from agentm.scenarios.trajectory_analysis import register as register_ta
    from agentm.scenarios.general_purpose import register as register_gp

    register_rca()
    register_ta()
    register_gp()
```

`discover()` is called once by `build_agent_system()` before looking up the scenario. New scenarios only need to add their `register()` call here.

Each scenario sub-package exposes a `register()` function that creates the scenario instance and calls `register_scenario()`:

```python
# scenarios/rca/__init__.py
def register() -> None:
    from agentm.harness.scenario import register_scenario
    from agentm.scenarios.rca.scenario import RCAScenario
    register_scenario(RCAScenario())
```

---

## Build Pipeline

`build_agent_system()` in `src/agentm/builder.py` is the single canonical entry point. It uses the Scenario protocol in a 4-phase build:

```
build_agent_system(scenario_name, scenario_config, system_config)
    |
    |-- 1. discover() + get_scenario(name)
    |
    |-- 2. _create_platform_resources()
    |       -> vault, trajectory, tool_registry, model configs
    |
    |-- 3. scenario.setup(SetupContext) -> ScenarioWiring
    |
    |-- 4. _create_worker_infrastructure(wiring)
    |       -> AgentRuntime + WorkerLoopFactory
    |
    |-- 5. _assemble_orchestrator_tools(wiring)
    |       -> SDK tools + scenario tools + registry tools
    |
    |-- 6. _build_orchestrator_loop(wiring)
    |       -> middleware stack + LLM + SimpleAgentLoop
    |
    +-> AgentSystem(loop, runtime, trajectory)
```

The scenario only participates in step 3 (`setup()`). Everything else is SDK infrastructure.

### Tool resolution order

In `_assemble_orchestrator_tools()`, tools listed in `orchestrator.tools` config are resolved in priority order:

1. **SDK tools** -- `dispatch_agent`, `check_tasks`, `inject_instruction`, `abort_task`
2. **Scenario tools** -- from `wiring.orchestrator_tools` (matched by name)
3. **Memory tools** -- `read_trajectory`, `get_checkpoint_history`, `jq_query`, `load_case_data`
4. **Registry tools** -- loaded from `config/tools/*.yaml` via `ToolRegistry`

If a tool name is not found in any source, `ConfigError` is raised.

---

## Examples

### RCAScenario (full wiring) -- `scenarios/rca/scenario.py`

Uses all ScenarioWiring fields: creates `HypothesisStore` and `ServiceProfileStore` in `setup()`, binds them into tools and `format_context` via `partial()` closures, provides 3 answer schemas (`scout`, `deep_analyze`, `verify`), `CausalGraph` output schema, and `think_stall_enabled` hooks. Uses `ctx.trajectory` to record hypothesis events.

### GeneralPurposeScenario (minimal wiring) -- `scenarios/general_purpose/scenario.py`

Returns `ScenarioWiring(answer_schemas={"execute": GeneralAnswer})`. Everything else uses defaults: no custom tools, no context formatting, default hooks, default termination.

### TrajectoryAnalysisScenario (medium wiring) -- `scenarios/trajectory_analysis/scenario.py`

Provides 2 answer schemas (`analyze`, `critique`) and `AnalysisReport` output schema. No custom tools or context formatting.

---

## Adding a New Scenario

1. Create `src/agentm/scenarios/<name>/scenario.py` with a class implementing `name` and `setup()`
2. Create `src/agentm/scenarios/<name>/__init__.py` with a `register()` function
3. Add the `register_<name>()` import to `src/agentm/scenarios/__init__.py` `discover()`
4. Create `config/scenarios/<name>/scenario.yaml` with orchestrator + agent configs
5. Create prompt templates under `config/scenarios/<name>/prompts/`

No SDK code changes required beyond the `discover()` import.

---

## Related Documents

- [System Architecture](system-design-overview.md) -- Overall system design and build pipeline
- [Agent Harness](agent-harness.md) -- AgentLoop, AgentRuntime, Middleware protocols
- [Orchestrator](orchestrator.md) -- Orchestrator loop, middleware, tools
- [Trajectory Analysis](trajectory-analysis.md) -- Trajectory analysis scenario
