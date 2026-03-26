# Design: SDK Semantic Consistency Refactoring

**Status**: APPROVED
**Created**: 2026-03-26
**Depends on**: [agent-harness.md](agent-harness.md)

---

## 1. Problem Statement

The Agent Harness migration (Phase 1-3) replaced the execution engine but left the
**interface layer** inconsistent. Five categories of debt remain:

1. **Tool type fragmentation** — `StructuredTool` (langchain), raw callables, and
   dict-of-functions coexist with no unified SDK type.
2. **Scenario integration is implicit** — `ReasoningStrategy` has 10 methods but
   only 5 are called in production; the key extension point (`create_scenario_tools`)
   is not even in the Protocol, accessed via `getattr`.
3. **Four global registries** — `STATE_SCHEMAS`, `_STRATEGY_INSTANCES`,
   `ANSWER_SCHEMA`, `OUTPUT_SCHEMAS` all populated by `discover()`.
4. **Dead code** — `ManagedTask`, `AgentRunStatus`, `SubAgentState`,
   `PhaseManager`, `recall_history`, and more.
5. **langchain leaks** — `HumanMessage` used in 6 places outside the LLM boundary;
   `StructuredTool` in 4 places; message format mismatch (langchain objects vs
   plain dicts in the same chain).

### Design Goal

One type per concept. One way to do each thing. langchain at the boundary only.

---

## 2. Tool Abstraction

### 2.1 `Tool` dataclass

```python
# src/agentm/harness/tool.py

@dataclass
class Tool:
    """SDK canonical tool type. No langchain dependency."""
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema (auto-derived from type hints)
    func: Callable[..., Any]     # sync or async

    async def ainvoke(self, args: dict[str, Any]) -> str:
        """Execute the tool. Normalizes return to str."""
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(**args)
        else:
            result = self.func(**args)
        return result if isinstance(result, str) else str(result)

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI function-calling format for model.bind_tools()."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

### 2.2 Creation API

Three creation paths, all return `Tool`:

```python
# 1. Decorator — primary path for scenario tools
@tool
async def update_hypothesis(id: str, status: Literal["formed", "confirmed"]) -> str:
    """Update hypothesis status."""
    ...

# 2. Decorator with overrides
@tool(name="think")
def think_tool(thought: str) -> str: ...

# 3. Factory — for ToolRegistry / partial-bound functions
t = tool_from_function(partial(query_prometheus, timeout=30), name="query_prometheus")
```

### 2.3 Parameter Schema Derivation

```python
def _schema_from_func(fn: Callable) -> dict[str, Any]:
    """Derive JSON Schema from function signature + type hints."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties, required = {}, []
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "return"):
            continue
        annotation = hints.get(name, str)
        field_schema = TypeAdapter(annotation).json_schema()
        if param.default is not param.empty:
            field_schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = field_schema
    return {"type": "object", "properties": properties, "required": required}
```

Uses Pydantic `TypeAdapter` for complex types (`Literal`, `Optional`, `list[str]`).

### 2.4 LLM Binding

```python
# Before: model.bind_tools(tools)  ← accepts StructuredTool
# After:  model.bind_tools([t.to_openai_schema() for t in tools])  ← accepts dict
```

`SimpleAgentLoop` internal change: `self._tools` becomes `dict[str, Tool]`.
`t.ainvoke(args)` interface unchanged.

### 2.5 ToolRegistry Migration

```python
class ToolDefinition:
    def create_tool(self, **config: Any) -> Tool:     # was create_with_config() -> StructuredTool
        bound = partial(self.func, **config) if config else self.func
        desc = config.pop("description", self.func.__doc__ or self.name)
        return tool_from_function(bound, name=self.name, description=desc)
```

---

## 3. Scenario Protocol

### 3.1 Interface

```python
# src/agentm/harness/scenario.py

@runtime_checkable
class Scenario(Protocol):
    """A scenario provides domain-specific behavior to the harness.

    Only two things required: a name, and a setup() that returns wiring.
    """

    @property
    def name(self) -> str: ...

    def setup(self, ctx: SetupContext) -> ScenarioWiring: ...
```

### 3.2 SetupContext (platform resources)

```python
@dataclass(frozen=True)
class SetupContext:
    """Resources the harness provides to a scenario during build."""
    vault: Any | None              # MarkdownVault instance
    trajectory: Any | None         # TrajectoryCollector instance
    tool_registry: Any             # ToolRegistry for resolving config-declared tools
```

### 3.3 ScenarioWiring (everything a scenario provides)

```python
@dataclass
class ScenarioWiring:
    """All wiring a scenario returns to the harness."""

    # Tools
    orchestrator_tools: list[Tool] = field(default_factory=list)
    worker_tools: list[Tool] = field(default_factory=list)

    # Dynamic context — always zero-arg. Scenario binds its own state.
    format_context: Callable[[], str] = field(default=lambda: "")

    # Structured output
    answer_schemas: dict[str, type[BaseModel]] = field(default_factory=dict)
    output_schema: type[BaseModel] | None = None

    # Behavior customization
    hooks: OrchestratorHooks = field(default_factory=OrchestratorHooks)

    # Termination logic (None = default <decision> tag parser)
    should_terminate: Callable[[Any], bool] | None = None

    # Scenario-specific middleware (appended after SDK middleware)
    orchestrator_middleware: list[Any] = field(default_factory=list)
    worker_middleware: list[Any] = field(default_factory=list)
```

**Key design decision**: `format_context` is **always zero-arg**. The scenario
creates its state containers in `setup()` and captures them in closures. This
eliminates `inspect.signature` detection, `app_state` dict, `initial_state()`,
and `state_schema()`.

### 3.4 ScenarioRegistry

Replaces `strategy_registry` + `state_registry`. Single registry:

```python
# src/agentm/harness/scenario.py

_SCENARIOS: dict[str, Scenario] = {}

def register_scenario(scenario: Scenario) -> None:
    _SCENARIOS[scenario.name] = scenario

def get_scenario(name: str) -> Scenario:
    ...

def list_scenarios() -> list[str]:
    ...
```

### 3.5 Scenario Registration

Before (4 registries):

```python
# scenarios/rca/__init__.py
def register():
    register_state("hypothesis_driven", HypothesisDrivenState)
    register_strategy("hypothesis_driven", HypothesisDrivenStrategy())
    ANSWER_SCHEMA.setdefault("scout", ScoutAnswer)
    OUTPUT_SCHEMAS.setdefault("CausalGraph", CausalGraph)
```

After (1 registry):

```python
# scenarios/rca/__init__.py
def register():
    from agentm.harness.scenario import register_scenario
    from agentm.scenarios.rca.scenario import RCAScenario
    register_scenario(RCAScenario())
```

Everything else moves into `RCAScenario.setup()`.

### 3.6 Scenario Implementations

**RCA** (complex):

```python
class RCAScenario:
    @property
    def name(self) -> str:
        return "hypothesis_driven"

    def setup(self, ctx: SetupContext) -> ScenarioWiring:
        hypothesis_store = HypothesisStore()
        profile_store = ServiceProfileStore()

        orch_tools = _build_rca_orchestrator_tools(ctx.trajectory, hypothesis_store, profile_store)
        worker_tools = _build_rca_worker_tools(profile_store)

        return ScenarioWiring(
            orchestrator_tools=orch_tools,
            worker_tools=worker_tools,
            format_context=partial(format_rca_context,
                                   profile_store=profile_store,
                                   hypothesis_store=hypothesis_store),
            answer_schemas={"scout": ScoutAnswer, "verify": VerifyAnswer, "deep_analyze": DeepAnalyzeAnswer},
            output_schema=CausalGraph,
            hooks=OrchestratorHooks(think_stall_enabled=True, think_stall_limit=3),
        )
```

**GP** (minimal):

```python
class GeneralPurposeScenario:
    @property
    def name(self) -> str:
        return "general_purpose"

    def setup(self, ctx: SetupContext) -> ScenarioWiring:
        return ScenarioWiring(
            answer_schemas={"execute": GeneralAnswer},
        )
```

---

## 4. Builder Simplification

### 4.1 Current state: 4 entry points

- `AgentSystemBuilder.build()` — 180-line legacy, does everything
- `GenericAgentSystemBuilder` — fluent builder, incomplete
- `AgentSystemBuilderFluent` — fluent wrapper around legacy
- `build_from_type()` — bridge function

### 4.2 New: single `build_agent_system()`

One function that:
1. Creates platform resources (vault, trajectory, tool_registry)
2. Calls `scenario.setup(ctx)` to get wiring
3. Creates `AgentRuntime` + `WorkerLoopFactory`
4. Creates SDK tools (dispatch, check, inject, abort)
5. Merges tools: config-declared + SDK + scenario
6. Builds middleware stack: SDK (context, compression, loop_detect) + scenario
7. Constructs `SimpleAgentLoop` for orchestrator
8. Returns `AgentSystem`

```python
def build_agent_system(
    scenario: Scenario,
    scenario_config: ScenarioConfig,
    system_config: SystemConfig | None = None,
    *,
    thread_id: str | None = None,
    tools_dir: Path | str | None = None,
    knowledge_base_dir: str | None = None,
) -> AgentSystem:
    ...
```

`AgentSystem` retains its current interface (`execute()`, `stream()`,
`__aenter__`/`__aexit__`), but `self.graph` is renamed to `self.loop`
(it's a `SimpleAgentLoop`, not a graph).

---

## 5. Dead Code Removal

### 5.1 Delete entirely

| Item | Location |
|------|----------|
| `ManagedTask` | models/data.py |
| `SubAgentState` | models/state.py |
| `SubAgentMessageSummary` | models/data.py |
| `CompressionRef` | models/data.py |
| `AgentRunStatus` | models/enums.py |
| `TaskStatus` | models/enums.py |
| `PhaseManager` | core/phase_manager.py |
| `recall_history` + `_set_graph_ref` | tools/orchestrator.py |
| `state_registry.py` | core/ (entire file) |
| `answer_schemas.py` global registry | models/ (keep `_BaseAnswer` only) |
| `output.py` global registry | models/ (delete entire file) |

### 5.2 Migrate then delete

| Item | Current | New home |
|------|---------|----------|
| `AgentRunStatus` in rca/data.py | `models/enums.py` | Use `AgentStatus` from harness/types.py |
| `PhaseDefinition` | models/data.py | Scenario-internal (only used by strategy tests) |
| `ScenarioToolBundle` | models/data.py | Replaced by `ScenarioWiring` |
| `ReasoningStrategy` | core/strategy.py | Replaced by `Scenario` protocol |
| `strategy_registry.py` | core/ | Replaced by `ScenarioRegistry` in harness/scenario.py |

### 5.3 Remove from models/enums.py

After cleanup, `models/enums.py` will be empty → delete the file.
`AgentStatus` in `harness/types.py` is the single source of truth.

---

## 6. langchain Boundary Convergence

### 6.1 Target boundary

```
INSIDE SDK (no langchain):          BOUNDARY (langchain OK):
─────────────────────────           ─────────────────────────
harness/*                           config/schema.py
  tool.py (Tool)                      create_chat_model() → ChatOpenAI/ChatAnthropic
  scenario.py (Scenario)              InMemoryRateLimiter
  loops/simple.py
  middleware.py
  runtime.py
scenarios/*
builder.py
tools/*
models/*
core/*
```

### 6.2 HumanMessage removal

All 6 usages of `HumanMessage` outside `config/schema.py`:

| Location | Current | Replacement |
|----------|---------|-------------|
| strategy.initial_state() | `messages=[HumanMessage(...)]` | Method deleted (Scenario has no initial_state) |
| cli/run.py L303, L821 | `HumanMessage(content=task)` | `{"role": "human", "content": task}` |
| middleware.py (compression) | `LCHumanMessage(content=prompt)` | `{"role": "human", "content": prompt}` |
| tools/orchestrator.py (recall) | `LCHumanMessage(content=prompt)` | Dead code — deleted with recall_history |

Note: `create_chat_model()` returns a langchain model that accepts both
plain dicts and message objects. Plain dicts are the SDK standard.

### 6.3 StructuredTool removal

| Location | Replacement |
|----------|-------------|
| `core/tool_registry.py` | `create_tool() → Tool` |
| `builder.py _func_to_tool()` | Deleted — scenario returns `Tool` directly |
| `scenarios/rca/tools.py` | `@tool` decorator |
| `tools/think.py` | `@tool` from harness/tool.py |

### 6.4 Remaining langchain (acceptable)

- `config/schema.py`: `create_chat_model()`, `InMemoryRateLimiter` — this IS the boundary
- `cli/export_eval.py`: `JsonPlusSerializer` — legacy data read, can be replaced later
- `tools/memory.py`: `JsonPlusSerializer` — legacy data read, can be replaced later

---

## 7. Implementation Phases

### Phase 1: Tool + Scenario Core

New files, no breaking changes yet.

1. Create `harness/tool.py` — `Tool`, `@tool`, `tool_from_function`, `_schema_from_func`
2. Create `harness/scenario.py` — `Scenario`, `SetupContext`, `ScenarioWiring`, `ScenarioRegistry`
3. Unit tests for both

### Phase 2: Migrate Scenarios + Builder

Wire scenarios to new protocol, rebuild builder.

1. Create `scenarios/rca/scenario.py` — `RCAScenario` implementing `Scenario`
2. Create `scenarios/general_purpose/scenario.py` — `GeneralPurposeScenario`
3. Create `scenarios/trajectory_analysis/scenario.py` — `TrajectoryAnalysisScenario`
4. Rewrite `scenarios/*/__init__.py` — single `register_scenario()` call
5. Migrate `tools/think.py` to `harness/tool.py @tool`
6. Migrate `core/tool_registry.py` to return `Tool`
7. Migrate `scenarios/rca/tools.py` to use `@tool`
8. Rewrite `builder.py` — single `build_agent_system()` function
9. Update `harness/loops/simple.py` — `tools: dict[str, Tool]`, `bind_tools` via schema
10. Update `harness/worker_factory.py` — accept `answer_schemas` from wiring, use `Tool`
11. Update `cli/run.py` — plain dicts, use `get_scenario()` + `build_agent_system()`
12. Update `server/app.py` if needed

### Phase 3: Cleanup

Remove all dead code and legacy paths.

1. Delete dead types: `ManagedTask`, `SubAgentState`, `SubAgentMessageSummary`,
   `CompressionRef`, `AgentRunStatus`, `TaskStatus`
2. Delete dead modules: `core/phase_manager.py`, `core/state_registry.py`,
   `core/strategy_registry.py`, `core/strategy.py`
3. Delete dead registries: `models/answer_schemas.py` (move `_BaseAnswer`),
   `models/output.py`
4. Delete dead code in `tools/orchestrator.py`: `recall_history`, `_set_graph_ref`
5. Delete `models/enums.py` (empty after cleanup)
6. Migrate `rca/data.py AgentOutcome.status` to `AgentStatus`
7. Remove `HumanMessage` imports from `cli/run.py`, `middleware.py`
8. Delete old builder classes: `GenericAgentSystemBuilder`, `AgentSystemBuilderFluent`,
   `build_from_type`
9. Delete `ScenarioToolBundle`, `PhaseDefinition` from `models/data.py`
10. Update all tests to match new APIs
11. Update `index.yaml`

---

## 8. Verification

After all phases:

- `grep -r "StructuredTool" src/` → 0 hits
- `grep -r "HumanMessage" src/` → 0 hits (outside config/schema.py)
- `grep -r "AgentRunStatus\|TaskStatus\|ManagedTask" src/` → 0 hits
- `grep -r "ANSWER_SCHEMA\|OUTPUT_SCHEMAS\|STATE_SCHEMAS" src/` → 0 hits
- `grep -r "ReasoningStrategy\|strategy_registry\|state_registry" src/` → 0 hits
- `uv run pytest` → all pass
- Only `config/schema.py` + 2 legacy files import from `langchain_core`/`langgraph`
