# Design: Pluggable Architecture (First Principles)

**Status**: PROPOSED
**Created**: 2026-04-30
**Reference codebase**: [`badlogic/pi-mono`](https://github.com/badlogic/pi-mono) — analyzed at commit on 2026-04-30, local clone at `/tmp/pi-analysis/pi-mono`.

---

## 1. First Principle

> **AgentM core is a mechanism, not a policy. Every policy is a port; every port has a default; every default is replaceable.**

Concrete consequences:

1. The SDK never imports scenario code. Scenarios import the SDK.
2. The SDK never assumes a delivery surface (CLI / Web / RPC / embedded). Each is a presenter on top of the same runtime.
3. Anything an extension can do, the user can do via the SDK with the same interface — there is no "private" path.
4. The SDK ships sensible defaults so a zero-config user gets a working agent. Defaults are themselves implementations of public ports, never special-cased.

This document is the **boundary contract**: which things are in core, which things are ports, and what each port's interface is. Everything else (`agent-harness.md`, `system-design-overview.md`, scenario designs) refines pieces under this contract.

---

## 2. Layered Package Boundary

Inspired by pi-mono's three-layer split (`pi-ai` → `pi-agent` → `pi-coding-agent`), AgentM is organized as:

```
┌──────────────────────────────────────────────────────────────────┐
│  agentm-modes/  CLI │ JSON │ RPC │ embedded SDK                  │ presenters
├──────────────────────────────────────────────────────────────────┤
│  agentm-harness/                                                 │
│    AgentSession (orchestrator) · ExtensionRunner · EventBus      │ harness
│    SessionManager · ResourceLoader · SettingsManager             │
├──────────────────────────────────────────────────────────────────┤
│  agentm-core/                                                    │
│    AgentLoop · Tool · Message · StreamFn · ToolOperations ports  │ pure SDK
├──────────────────────────────────────────────────────────────────┤
│  agentm-llm/  (provider layer; analogous to pi-ai)               │ provider
│    Model registry · StreamFn implementations · OAuth             │
└──────────────────────────────────────────────────────────────────┘
```

**Dependency rule**: arrows point downward only. `agentm-core` must be importable in a Jupyter notebook with no harness, no CLI, no filesystem touched.

| Layer | What it knows | What it does NOT know |
|---|---|---|
| `agentm-llm` | HTTP, OAuth, provider quirks | tools, agents, sessions |
| `agentm-core` | message turns, tool execution, streaming | sessions, files, scenarios, UI |
| `agentm-harness` | persistence, extensions, resource discovery | concrete scenarios, UI rendering |
| `agentm-modes` | I/O surface (stdin/stdout/TUI/HTTP) | LLM protocol, agent loop |

**Reference**: pi-mono enforces this physically as separate packages — see `packages/{ai,agent,coding-agent}/` and the import graph in `packages/coding-agent/src/core/sdk.ts:1-30`.

---

## 3. Five Pluggability Axes

Every axis below is a `typing.Protocol` in `agentm-core`. The harness ships a default implementation; extensions/users can substitute. **All five must be replaceable without forking.** In v0, Operations replacement is constitution-only: the harness selects the operations bundle when constructing a session, and atoms only consume it through `ExtensionAPI.get_operations()`.

### 3.1 LLM Stream (the model boundary)

```python
class StreamFn(Protocol):
    async def __call__(
        self,
        messages: list[Message],
        model: Model,
        tools: list[Tool],
        *,
        signal: AbortSignal | None = None,
        thinking: ThinkingLevel = "off",
    ) -> AsyncIterator[AssistantEvent]: ...
```

- Pure boundary: takes provider-shaped messages, returns events.
- Default implementations live in `agentm-llm` per provider; provider-internal stream assembly is shared by `agentm.llm._common.StreamAccumulator`, so new providers supply only event mapping plus a `ToolSpecAdapter`.
- Tool-call argument parse failures stay observable as typed stream/bus events (`ToolCallArgsParseError`) while preserving the kernel invariant that `ToolCallBlock.arguments` is a parsed dict.
- Extensions register additional providers via `register_provider(name, ProviderConfig)`. The harness chooses the active registration through the `ProviderResolver` port; the default `LastRegisteredWins` resolver preserves insertion-order behavior.
- **Crucial**: `StreamFn` is the only point that touches a real LLM API. The agent loop has zero hard-coded provider knowledge.

**Reference**: pi-mono `packages/agent/src/types.ts:18-26` (`StreamFn` type), `packages/coding-agent/src/core/extensions/types.ts:1212-1245` (`registerProvider` API with `streamSimple` override).

### 3.2 Tool Execution (the environment boundary)

Three-layer split. `ToolDefinition` and `Tool` are extension/runtime surfaces; `Operations` are a constitution-selected environment port in v0:

```python
@dataclass
class ToolDefinition(Generic[P, D]):
    name: str
    description: str
    parameters: TypeSchema           # JSON schema
    execute: Callable[..., Awaitable[ToolResult[D]]]
    # optional UI/semantics
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] = field(default_factory=list)
    execution_mode: Literal["sequential", "parallel"] = "parallel"
    prepare_arguments: Callable[[dict], P] | None = None
    render_call: Callable | None = None
    render_result: Callable | None = None

class Tool(Protocol):                 # what the loop sees
    name: str
    parameters: TypeSchema
    async def execute(self, args, signal, on_update) -> ToolResult: ...

class FileOperations(Protocol):       # the environment port
    async def read_file(self, path: str) -> bytes: ...
    async def write_file(self, path: str, content: bytes) -> None: ...
    async def access(self, path: str) -> None: ...

class BashOperations(Protocol):
    async def exec(self, cmd, cwd, *, on_data, signal, timeout, env) -> ExecResult: ...
```

- `ToolDefinition` is the harness/UI-facing record.
- `Tool` is the bare execution interface used by the agent loop.
- `XxxOperations` is the **smallest possible port** for swapping environments (local FS → SSH → sandbox → in-memory). It is replaceable by harness/session construction, not by an atom-level `register_operations` hook.

**Why three layers**: the "what" (definition), "how-to-call" (Tool), and "where-it-runs" (Operations) vary independently. Pi proves it: their `read.ts` tool body is unchanged whether running locally or over SSH; only `ReadOperations` is swapped.

**Reference**:
- Definition: `packages/coding-agent/src/core/extensions/types.ts:332-403` (`ToolDefinition`)
- Operations port pattern: `packages/coding-agent/src/core/tools/read.ts:30-46`, `bash.ts:42-66`, `edit.ts` (`EditOperations`)
- Default local impl: `packages/coding-agent/src/core/tools/bash.ts:75-100` (`createLocalBashOperations`)
- File mutation queue (cross-tool serialization): `packages/coding-agent/src/core/tools/file-mutation-queue.ts`

### 3.3 Session Persistence (the state boundary)

```python
@dataclass
class SessionEntry:
    type: str            # "message" | "compaction" | "branch_summary" | "model_change" | custom
    id: str              # uuid7
    parent_id: str | None
    timestamp: str
    payload: Any         # type-specific

class SessionManager(Protocol):
    def append(self, entry: SessionEntry) -> None: ...
    def get_active_branch(self) -> list[SessionEntry]: ...
    def fork_at(self, entry_id: str) -> "SessionManager": ...
    def navigate_to(self, entry_id: str) -> None: ...
    def find_by_id(self, entry_id: str) -> SessionEntry | None: ...
```

- Single JSONL file, append-only, tree-structured via `parent_id`.
- `payload: Any` (or extensible `details` field per entry type) lets extensions persist structured data without forking the format.
- Branching, forking, compaction, navigation are **operations on the entry tree**, not separate features.
- Default impl writes to `~/.agentm/sessions/`; SDK callers can pass `InMemorySessionManager` or `SqliteSessionManager`.

**Reference**: `packages/coding-agent/src/core/session-manager.ts:30-90` (entry types with `parentId`), `:60-78` (`CompactionEntry.details: T` for extension data), `1425` lines total — but the format is what matters, not the implementation size.

### 3.4 Resource Discovery (the project-context boundary)

```python
class ResourceLoader(Protocol):
    def get_skills(self) -> list[Skill]: ...
    def get_prompt_templates(self) -> list[PromptTemplate]: ...
    def get_context_files(self) -> list[ContextFile]: ...   # AGENTS.md / CLAUDE.md
    def get_extensions(self) -> ExtensionsResult: ...
    def reload(self) -> None: ...
```

- Default impl walks `cwd → parent dirs → ~/.agentm/` filesystem hierarchy.
- Embedded SDK callers pass a `ResourceLoader` backed by DB / HTTP / in-memory.
- Extensions can extend the discovery via the `resources_discover` event (returning extra paths) without replacing the loader.

**Why this matters for SDK reuse**: a web-app embedding AgentM has no filesystem. If `ResourceLoader` is hard-wired to `os.walk`, the SDK is unembeddable. Pi enforces this via the `ResourceLoader` interface and `DefaultResourceLoader` class.

**Reference**: `packages/coding-agent/src/core/resource-loader.ts` (interface + default impl), `packages/coding-agent/src/core/extensions/types.ts:495-510` (`ResourcesDiscoverEvent`).

### 3.5 Extension Bus (the policy-replacement boundary)

The mechanism by which "built-in features" become "default extensions". An EventBus + 25+ typed events with three semantics:

| Event family | Examples | Handler return semantics |
|---|---|---|
| Lifecycle (passive) | `session_start`, `agent_start`, `turn_end` | None — observers only |
| Mutating (active) | `tool_call`, `context`, `input` | Mutate payload in place; later handlers see prior changes |
| Replaceable (`before_*`) | `before_agent_start`, `session_before_compact`, `session_before_tree` | Return `{block?, cause?, cancel?, replacement?}` to override default flow |

**The killer property**: any built-in operation (compaction, fork, system-prompt assembly, tool execution) emits a `before_*` event whose handlers can `cancel: true` and supply a custom result. This is how plan-mode, sub-agent, permission gate, sandbox, sub-agent — all the things AgentM might want to add — become **default extensions** rather than core features.

```python
class EventBus(Protocol):
    def emit(self, channel: str, data: Any) -> Awaitable[list[Any]]: ...
    def on(self, channel: str, handler: Handler) -> Unsubscribe: ...

class ExtensionAPI(Protocol):
    on: EventSubscribe                       # typed overloads per event
    register_tool: Callable[[ToolDefinition], None]
    register_command: Callable[[str, CommandSpec], None]
    register_provider: Callable[[str, ProviderConfig], None]
    register_message_renderer: Callable[[str, Renderer], None]
    send_user_message: Callable[..., None]
    append_entry: Callable[[str, Any], None]
    events: EventBus                         # cross-extension comms
```

Registered slash-command execution is itself a policy port: the
`slash_commands` atom parses `/cmd args`, but command lookup, ownership, and
handler execution go through the typed `CommandDispatcher` service facade. The
harness default owns the live command registry and owner API selection; atoms do
not read raw harness registry dictionaries.

**Reference**:
- Minimal EventBus: `packages/coding-agent/src/core/event-bus.ts` (33 lines — copy this verbatim conceptually)
- Event taxonomy: `packages/coding-agent/src/core/extensions/types.ts:495-960` (full event type hierarchy)
- Mutation-in-place contract: `extensions/types.ts:854-868` (`ToolCallEvent.input is mutable`)
- Cancel/replace contract: `extensions/types.ts:535-546` (`SessionBeforeCompactResult.cancel | compaction`), `:563-580` (`SessionBeforeTreeResult.summary`)
- ExtensionAPI surface: `extensions/types.ts:1067-1290`

---

## 4. AgentSession: the Orchestrator Façade

Per pi-mono's `agent-session.ts` (3099 lines, but **zero business logic**), the orchestrator is intentionally fat-but-thin: it holds references to every subsystem and wires events, but every actual decision lives in a subsystem.

```python
class AgentSession:
    def __init__(self, config: AgentSessionConfig):
        self.agent: AgentLoop                # core
        self.session_manager: SessionManager
        self.settings: SettingsManager
        self.resources: ResourceLoader
        self.models: ModelRegistry
        self.extensions: ExtensionRunner
        self._event_bus: EventBus

    async def prompt(self, text: str, *, options: PromptOptions) -> None:
        # 1. emit "input" event (slash_commands and templates transform/handle)
        # 2. assemble user message + injected nextTurn entries
        # 3. emit "before_agent_start" (extensions can replace system or veto)
        # 4. await self.agent.run(...)
```

**Design rule**: `AgentSession.prompt` and friends are 100-line **dispatchers**. Any branch with logic-content >10 lines is a smell — extract it to a service or extension. Construction-only wiring can live beside the façade in harness factory/runtime modules; runtime dependency bundles should be passed as data (`SessionRuntime`) rather than long parameter lists.

**Reference**: `packages/coding-agent/src/core/agent-session.ts:942-1050` (`prompt` method — note how mechanical it is).

---

## 5. Mode Layer: presenters only

Each mode is a thin consumer of `AgentSession`:

| Mode | Role |
|---|---|
| `interactive` | TUI; subscribes to events, renders messages, captures keyboard, drives `prompt()` |
| `print` | One-shot: read stdin (+args), call `prompt()`, render assistant text, exit |
| `json` | Like print, but emits one JSON-line per event for machine consumers |
| `rpc` | LF-delimited JSONL on stdin/stdout; bidirectional; for non-Python integrations |
| `sdk` | No I/O; library users call `await session.prompt(...)` and subscribe to events directly |

Modes share **all** runtime; they only differ in:
- how user input arrives (`InputSource` field)
- how events are rendered/serialized
- which `ExtensionUIContext` they provide (TUI / no-op / structured)

**Reference**: `packages/coding-agent/src/modes/{print-mode.ts, rpc/, interactive/}` and `core/sdk.ts:createAgentSession` (the SDK entrypoint shared by all modes).

**Design rule for AgentM**: any feature added to a mode that *cannot* also be reached via the SDK is a bug. The SDK is the contract; modes are sugar.

---

## 6. Pluggability Test (acceptance criteria)

A change to the architecture is acceptable iff each of these is achievable **without forking core**:

1. **Replace the LLM provider** with a corporate proxy speaking a custom protocol → register a `StreamFn`.
2. **Run `bash` tool over SSH** to a remote host → construct the session with SSH-backed `BashOperations`.
3. **Persist sessions to Postgres** instead of JSONL → swap `SessionManager`.
4. **Embed AgentM in a Django app** with no filesystem access → swap `ResourceLoader`.
5. **Add a permission-prompt gate** before every tool call → register `on("tool_call", ...)` returning `{block, reason}`.
6. **Replace the default compaction strategy** with a domain-specific one → register `on("session_before_compact", ...)` returning `{compaction: ...}`.
7. **Add a sub-agent system** → an extension that registers a `dispatch_agent` tool whose `execute` spawns nested `AgentSession` instances. Core never learns about sub-agents.
8. **Add plan mode** → an extension that intercepts `before_agent_start`, prepends a planning system prompt, and adds a `submit_plan` tool. Core never learns about plan mode.

If any of these requires editing `agentm-core` or `agentm-harness`, the boundary is wrong.

**Reference**: cases 7 and 8 are exactly how pi argues for omitting these from core — see pi README "Philosophy" section and `packages/coding-agent/README.md:340-360`.

---

## 7. Mapping to Existing AgentM Concepts

This document's contract refines, but does not replace, these existing designs. The mapping:

| Existing concept | Lives in | Pluggable as |
|---|---|---|
All previous AgentM concepts are being **re-implemented as extensions** on the v2 kernel + ExtensionAPI. The legacy code tree is deleted in a single sweep at the end of Phase 2 (see [extension-as-scenario.md §8](extension-as-scenario.md#8-target-file-layout)). No coexistence layer, no middleware↔event-bus bridge.

| Legacy concept | Becomes |
|---|---|
| `AgentRuntime` / `WorkerLoopFactory` / `AgentHandle` | `extensions.builtin.sub_agent` (Group C) |
| `Middleware` 3-hook protocol + concrete middlewares | EventBus handlers in individual extensions (Group A/B) |
| `Scenario` Protocol + `ScenarioWiring` + `ScenarioRegistry` | extensions whose `install()` registers tools + system prompt + handlers (Group D) |
| `permission-mode`, `tool-filter`, `tool-dedup`, `cost_budget`, etc. | one extension each in `extensions/builtin/` |
| `agent_memory` | dropped from catalog; RCA's hypothesis store + `SessionManager.payload` covers the use case |
| `core/trajectory.py` `TrajectoryCollector` | `extensions.builtin.trajectory` |
| `core/tool.py @tool` decorator | deleted; new tools are `kernel.Tool` instances or `FunctionTool` adapters |
| `builder.py` `build_agent_system` | deleted; replaced by `AgentSession.create(...)` |

---

## 8. Reference Index (pi-mono ↔ AgentM)

Quick lookup for implementation. All paths relative to `pi-mono/packages/`.

### Layered package boundary (§2)
- Pure provider: `ai/src/{stream.ts,api-registry.ts,oauth.ts}`
- Pure agent loop: `agent/src/{agent.ts,agent-loop.ts,types.ts}` (~1300 LoC total)
- Harness orchestrator: `coding-agent/src/core/agent-session.ts`
- Mode presenters: `coding-agent/src/modes/`

### Stream / Provider port (§3.1)
- `StreamFn` type: `agent/src/types.ts:18-26`
- Provider registration: `coding-agent/src/core/extensions/types.ts:1212-1290`
- Model registry: `coding-agent/src/core/model-registry.ts`

### Tool / Operations port (§3.2)
- `AgentTool` (loop-facing): `agent/src/types.ts` (search `interface AgentTool`)
- `ToolDefinition` (UI/semantics): `coding-agent/src/core/extensions/types.ts:332-403`
- `defineTool` helper: `coding-agent/src/core/extensions/types.ts:415-419`
- `ReadOperations`: `coding-agent/src/core/tools/read.ts:30-46`
- `BashOperations`: `coding-agent/src/core/tools/bash.ts:42-66`
- Local default: `tools/bash.ts:75-150`
- File mutation queue: `tools/file-mutation-queue.ts`
- Schema validation hook (`prepareArguments`): `extensions/types.ts:362`

### Session port (§3.3)
- Entry types: `coding-agent/src/core/session-manager.ts:30-100`
- Tree navigation: same file, search `navigateTree` / `fork`
- Compaction extension hook (`details: T`): `session-manager.ts:60-78`
- Compaction logic (default impl): `coding-agent/src/core/compaction/compaction.ts`
- JSONL format spec: `coding-agent/docs/session-format.md`

### Resource port (§3.4)
- `ResourceLoader` interface: `coding-agent/src/core/resource-loader.ts` (top of file)
- `DefaultResourceLoader`: same file
- Skill spec & loader: `coding-agent/src/core/skills.ts`
- Discovery extension hook: `extensions/types.ts:495-510` (`ResourcesDiscoverEvent`)
- Settings layering: `coding-agent/src/core/settings-manager.ts`

### Extension bus (§3.5)
- Minimal bus impl: `coding-agent/src/core/event-bus.ts` (33 lines)
- Event taxonomy: `coding-agent/src/core/extensions/types.ts:495-960`
- ExtensionAPI surface: `extensions/types.ts:1067-1290`
- Runner (default dispatcher): `coding-agent/src/core/extensions/runner.ts`
- Loader (extension discovery): `coding-agent/src/core/extensions/loader.ts`
- ExtensionUIContext (per-mode UI surface): `extensions/types.ts:120-280`

### Orchestrator façade (§4)
- `AgentSession` class: `coding-agent/src/core/agent-session.ts:239-` (rest of file)
- `prompt()` flow: `agent-session.ts:942-1050`
- Streaming queue (steer / followUp): same file, search `_queueSteer` / `_queueFollowUp`
- Pending message queue (in core): `agent/src/agent.ts:103-150` (`PendingMessageQueue`)

### Modes (§5)
- SDK factory: `coding-agent/src/core/sdk.ts:createAgentSession`
- Print mode: `coding-agent/src/modes/print-mode.ts`
- RPC mode (JSONL): `coding-agent/src/modes/rpc/`
- Interactive: `coding-agent/src/modes/interactive/`

### Philosophy (§6 acceptance test)
- pi README "Philosophy" section: `coding-agent/README.md` (search "Philosophy")
- Why no MCP / no sub-agents / no plan mode in core: same section

---

## 9. Open Questions

1. **Python equivalent of TypeBox**: pi uses TypeBox for tool parameter schemas (compile-time + runtime types). Python options: pydantic v2 (heavy, popular), msgspec (fast, less ecosystem), plain JSON Schema + manual validation. Decide before locking `ToolDefinition.parameters`.
2. **Async event bus dispatch**: serial vs. concurrent. Pi runs serially per channel (so mutation order is stable). AgentM should match — concurrent extension execution has no upside and breaks the "later handlers see earlier mutations" contract.
3. **Middleware-to-event migration**: see §7 open decision. Keep `Middleware` as facade vs. fully replace.
4. **Mode parity**: do we need `interactive` TUI in v0.1? Pi has 7 modes; AgentM might ship `sdk + json` first and add CLI later. The architecture supports either order.
5. **Where do "Scenarios" sit?** Suggestion: a Scenario is a function that takes a fresh `ExtensionAPI` and registers everything needed (tools, middleware, prompts, default extensions). This makes scenarios trivially composable and indistinguishable from third-party extensions. See [generic-state-wrapper.md](historical/generic-state-wrapper.md) for the existing direction; harmonize in a follow-up plan.

---

## 10. Next Steps

1. Update `index.yaml` to register `pluggable_architecture` concept and link related docs.
2. Draft `plans/YYYY-MM-DD-pluggable-skeleton.md` — the implementation plan to:
   a. Carve `agentm-core` out of current package (move `AgentLoop`, `Middleware`, `Tool` types).
   b. Define the five Protocol ports.
   c. Implement minimal `EventBus` + 6 critical events (`input`, `before_agent_start`, `tool_call`, `tool_result`, `context`, `agent_end`).
   d. Refactor existing `AgentRuntime` to be the harness orchestrator.
3. Acceptance: a smoke-test scenario that swaps `StreamFn` (mock LLM) and constructs the session with `BashOperations` (in-memory FS) without modifying core.
