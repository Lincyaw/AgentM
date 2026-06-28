# Design: Pluggable Architecture (First Principles)

**Status**: PROPOSED
**Created**: 2026-04-30

---

## 1. First Principle

> **AgentM core is a mechanism, not a policy. Every policy is a port; every port has a default; every default is replaceable.**

**Corollary — the unreplaceable-substrate test.** The only thing that cannot be a replaceable extension is *the act of loading replacements itself*. Anything else — including all "defaults" — must live as an atom that the substrate loads. The judgement rule is:

> **"If it were replaceable, who would execute the replacement?"** Whatever has no answer is substrate; everything else is policy and belongs in `extensions/`.

Concrete consequences:

1. The SDK never imports scenario code. Scenarios import the SDK.
2. The SDK never assumes a delivery surface (CLI / Web / RPC / embedded). Each is a presenter on top of the same runtime.
3. Anything an extension can do, the user can do via the SDK with the same interface — there is no "private" path.
4. Defaults are themselves atoms (`extensions/builtin/<name>.py`) listed in the default scenario manifest. They are never special-cased by the substrate. A zero-config user gets a working agent because the default scenario enumerates a working set, not because the substrate hides defaults.
5. The substrate (formerly "harness") does **not** pick policy bundles. It loads the extension list and asserts the required services have been registered when freeze happens; if a scenario omits, say, an `Operations` provider, freeze fails loudly rather than the substrate silently filling in.

This document is the **boundary contract**: which things are substrate, which things are ports, and what each port's interface is. Everything else (scenario designs, atom designs) refines pieces under this contract.

---

## 2. Layered Package Boundary

AgentM collapses to **three layers**. The historical four-layer split (`core` / `harness` / `llm` / `presenters`) carved harness out of core to protect "Jupyter clean import", but that constraint is really about **no side effects at import time**, not about which classes are allowed to exist. With defaults moved to atoms, the harness shrinks to pure substrate and merges back into core.

```
┌──────────────────────────────────────────────────────────────────┐
│  agentm.cli  /  embedded SDK  /  (future HTTP, RPC)              │  presenters
├──────────────────────────────────────────────────────────────────┤
│  agentm.extensions.builtin/  +  contrib/extensions/              │  atoms (policy)
│    operations_local  · llm_anthropic · session_state_memory      │
│    read_file · write_file · edit · bash · skills · ...           │
├──────────────────────────────────────────────────────────────────┤
│  agentm.core (unreplaceable substrate, write-protected)          │  substrate
│    abi/      Protocols + dataclasses + typed events              │
│    runtime/  AgentSession · EventBus · SessionManager · Loader   │
│              · catalog freeze · reload transaction               │
│    lib/      pure helpers (edit_diff · frontmatter · path_utils) │
└──────────────────────────────────────────────────────────────────┘
```

**Dependency rule**: arrows point downward only. `agentm.core` must be importable in a Jupyter notebook **with zero side effects at import time**. That property is module-load-time, not subpackage-content: `agentm.core.abi` and `agentm.core.lib` are pure types + pure functions; `agentm.core.runtime` *contains* stateful classes (`AgentSession`, `GitBackedResourceWriter`, `JsonlSessionStore`) but its modules perform no I/O at `import agentm.core.runtime` time. Side effects only happen when a session is **constructed**.

| Layer | What it knows | What it does NOT know |
|---|---|---|
| `agentm.core` | how to load extensions, run the agent loop, emit events, freeze a catalog, persist a session | any concrete provider, any concrete tool, any scenario |
| `agentm.extensions.builtin` (and `contrib/`) | concrete policy — how to read a file locally, how to talk to Anthropic, how to compact | how the substrate works internally; atom-to-atom imports forbidden (§11) |
| `agentm.cli` / embedded SDK | I/O surface (stdin/stdout/TUI/HTTP), CLI argument parsing, exit codes | LLM protocol, agent loop |

**There is no `agentm.harness` package and no `agentm.llm` package.** Provider stream implementations are atoms (`extensions/builtin/llm_<provider>.py`); session orchestration lives in `core.runtime/`; the validator forbids atoms from importing `core.runtime.*`.

---

## 3. Five Pluggability Axes

Every axis below is a `typing.Protocol` in `agentm.core.abi`. **All five must be replaceable without forking.** Replacement happens through one of two mechanisms, decided by the **pre / post-atom-install criterion**:

| Criterion | Replacement mechanism | Reason |
|---|---|---|
| The substrate needs the value to **construct or run the loader itself** — i.e. it's consumed pre-atom-install | **Config-time injection** via `AgentSessionConfig.<field>`; substrate falls back to a documented default if `None` | An atom can't supply something the atom-loader needs before any atom has installed |
| The value is consumed **only after atoms install** | **Atom-registered** via `api.register_<axis>(...)`; substrate fails loud at freeze if nothing was registered | Atoms are the natural home for runtime policy; the default scenario manifest enumerates the working set |

Concretely:

- **Atom-registered** (post-install): `Operations` (read by every tool atom), `StreamFn` (read by AgentLoop). Defaults live in `extensions/builtin/operations_local.py`, `extensions/builtin/llm_<provider>.py`. The substrate has no fallback — if a scenario manifest omits them, freeze raises.
- **Config-injected** (pre-install): `SessionManager` (CLI resumes/forks need it before session construction), `ResourceLoader` (atoms read its content but the substrate constructs it from `cwd`), `ResourceWriter` (`AtomReloader` consumes it during scope wiring). Defaults: `InMemorySessionManager`, `InMemoryResourceLoader`, `GitBackedResourceWriter`. SDK consumers swap by passing alternatives through `AgentSessionConfig`.
- **Default-pluggable** (config-injected + atom-overridable): `ResourceWriter` additionally exposes `api.register_resource_writer(...)`. The substrate pre-populates the slot with the config-injected default so its own bookkeeping (catalog freeze, atom reload) is unblocked, but an atom — typically an environment atom like `operations_agent_env` — may replace it once at install time to redirect writes (e.g. into a sandbox). Register-once is enforced via a `replaced` flag on the holder; the substrate distinguishes "atom replacing the bootstrap default" (allowed) from "second atom replacing an earlier atom's writer" (rejected). This is the only port that combines both replacement mechanisms; it exists because writes are both substrate-internal infrastructure *and* a policy axis tied to where compute physically lives.

All three mechanisms preserve the axiom — the substrate never holds an unreplaceable concrete implementation. The split is mechanical (timing of first use + whether a sandbox-class atom needs to retarget the value), not philosophical.

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
- Default implementations live as atoms (`agentm.extensions.builtin.llm_<provider>`); provider-internal stream assembly is shared by `agentm.core.lib.stream.StreamAccumulator`, so new providers supply only event mapping plus a `ToolSpecAdapter`.
- Retry, transport, and reasoning round-trip behavior are policy, not provider folklore: providers may accept an injected `RetryPolicy`, expose security-relevant transport overrides such as `verify_ssl=False` via diagnostics, and surface provider-specific thinking/reasoning choices as constructor/config options.
- Tool-call argument parse failures stay observable as typed stream/bus events (`ToolCallArgsParseError`) while preserving the kernel invariant that `ToolCallBlock.arguments` is a parsed dict.
- Extensions register additional providers via `register_provider(name, ProviderConfig)`. `ProviderConfig` lives in `agentm.core.abi.provider` so provider atoms speak the same ABI as everyone else. The runtime substrate chooses the active registration through the `ProviderResolver` port; the default `LastRegisteredWins` resolver preserves insertion-order behavior.
- Presenter-side provider selection goes through `ProviderRegistry.build(provider, config)`: descriptors own CLI extension module paths, default model ids, aliases, and ambient env-var conventions, so adding a provider descriptor does not require editing CLI branches.
- **Crucial**: `StreamFn` is the only point that touches a real LLM API. The agent loop has zero hard-coded provider knowledge.

### 3.2 Tool Execution (the environment boundary)

Three-layer split. `ToolDefinition` and `Tool` are extension/runtime surfaces; `Operations` is the environment port, registered by an atom like any other axis:

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
    async def access(self, path: str) -> bool: ...
    async def is_dir(self, path: str) -> bool: ...
    async def list_dir(self, path: str) -> list[str]: ...

class BashOperations(Protocol):
    async def exec(self, cmd, cwd, *, on_data, signal, timeout, env) -> ExecResult: ...
```

- `ToolDefinition` is the runtime/UI-facing record.
- `Tool` is the bare execution interface used by the agent loop.
- `XxxOperations` is the **smallest possible port** for swapping environments (local FS → SSH → sandbox → in-memory). It is replaceable through `api.register_operations(file=..., bash=...)`, called by an early atom in the scenario manifest (default: `operations_local`). The substrate enforces "registered at most once before freeze" and "must be registered by freeze time, else fail loud".

**File IO seam decision (issue #89)**: AgentM uses the hybrid seam. Guarded
reads (`read`) consume `api.get_operations().file`; mutating file tools
(`write`, `edit`) consume `api.get_resource_writer()`. Local search is handled
through `bash` and existing CLI tools (`rg`, `find`, `git ls-files`) rather than
default thin wrappers. `FileOperations` is the environment read port, while
`ResourceWriter` is the mutation chokepoint that enforces managed-resource
versioning and constitution-path rejection. Scenario authors that need to
redirect reads override `FileOperations`; scenario authors that need to redirect
or audit writes override `ResourceWriter`. Atoms must not call both seams for one
write path.

**Why three layers**: the "what" (definition), "how-to-call" (Tool), and "where-it-runs" (Operations) vary independently. The same tool body can run locally or over SSH by swapping only the underlying `Operations` implementation.

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
- Presenters depend on `SessionStore` (`open`, `most_recent`, `create`) rather than globbing JSONL files directly. `JsonlSessionStore` wraps the current `SessionManager` format, while tests and future backends can provide in-memory, sqlite, or remote implementations without changing CLI/TUI construction.

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

**Why this matters for SDK reuse**: a web-app embedding AgentM has no filesystem. If `ResourceLoader` is hard-wired to `os.walk`, the SDK is unembeddable; keeping it behind a Protocol makes alternative backends (DB, HTTP, in-memory) trivial.

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
runtime default owns the live command registry and owner API selection; atoms do
not read raw runtime registry dictionaries.

Retry policy follows the same service-facade rule: `agentm.core.abi.retry.RetryPolicy`
is a tiny async port, the built-in `retry_policy` atom registers the default
exponential-backoff implementation with `api.set_service("retry_policy", ...)`,
and provider adapters use provider-typed retry predicates rather than string
sniffing wire errors.

---

## 4. AgentSession: the Orchestrator Façade

`AgentSession` lives in `agentm.core.runtime` (not in a separate harness package). It is the substrate's session-orchestration entry point — fat-but-thin: it holds references to every subsystem and wires events, but every actual decision lives in an atom-registered service. The orchestrator is part of substrate because it answers the unreplaceable-substrate test with "the runtime itself" (cf. §1 corollary).

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

**Design rule**: `AgentSession.prompt` and friends are 100-line **dispatchers**. Any branch with logic-content >10 lines is a smell — extract it to a service or extension. Construction-only wiring lives in `core.runtime/` factory modules; runtime dependency bundles should be passed as data (`SessionRuntime`) rather than long parameter lists.

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

Presenter-owned commands may add UI affordances, but command discovery and extension-registered command parity stay registry-driven: the Textual mode uses a `BuiltinCommandRegistry` for its local commands and mirrors `ExtensionAPI.register_command` registrations into the same palette/dispatch path. Kernel event identity uses prompt-local `turn_index` plus session-monotone `turn_id`; presenters key long-lived widgets by `turn_id`.

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

If any of these requires editing `agentm.core`, the boundary is wrong.

A 9th case has been promoted to first-class:

9. **Replace the file/bash environment** with an SSH bundle, a sandbox, or an in-memory fake → write/register an alternate `operations_*` atom; the default scenario's `operations_local` is interchangeable with it. The substrate has no `_default_local_operations()` and no `Local{File,Bash}Operations` import — those live in `extensions/builtin/operations_local.py`.

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

## 9. Open Questions

1. **Tool parameter schema choice**: pydantic v2 (heavy, popular), msgspec (fast, less ecosystem), or plain JSON Schema + manual validation. Decide before locking `ToolDefinition.parameters`.
2. **Async event bus dispatch**: serial vs. concurrent. AgentM runs handlers serially per channel so mutation order is stable; concurrent execution has no upside and breaks the "later handlers see earlier mutations" contract.
3. **Middleware-to-event migration**: see §7 open decision. Keep `Middleware` as facade vs. fully replace.
4. **Mode parity**: do we need `interactive` TUI in v0.1? AgentM may ship `sdk + json` first and add a TUI later. The architecture supports either order.
5. **Where do "Scenarios" sit?** Suggestion: a Scenario is a function that takes a fresh `ExtensionAPI` and registers everything needed (tools, middleware, prompts, default extensions). This makes scenarios trivially composable and indistinguishable from third-party extensions. See [generic-state-wrapper.md](historical/generic-state-wrapper.md) for the existing direction; harmonize in a follow-up plan.

---

## 10. Migration Roadmap — Collapsing harness into core (completed 2026-05-11)

This section documents the migration that landed 2026-05-11; it is kept as a
history note. The tree no longer has an `agentm.harness` package, an
`agentm.llm` package, or `core/_internal/operations_impl.py`. The §2
three-layer target is the current shape.

Summary of what landed:

1. **Default policies promoted to atoms.** `LocalFileOperations` /
   `LocalBashOperations` → `extensions/builtin/operations_local.py`; provider
   `StreamFn` defaults → `extensions/builtin/llm_<provider>.py`; the
   `GitBackedResourceWriter` config seam → `AgentSessionConfig.resource_writer`.
2. **`register_*` hooks** on `ExtensionAPI` enforce "register at most once
   before freeze"; the substrate fails loud if a required axis is unregistered.
3. **Default scenario manifests** enumerate the atom set explicitly; the
   substrate no longer knows any default atom's name.
4. **`agentm.harness/*` merged into `agentm.core.runtime/*`** (Stage 4a
   move-and-shim, Stage 4b ABI/impl split). The §11 atom validator forbids
   `agentm.core.runtime.*` imports; atoms speak in `agentm.core.abi.*` +
   `agentm.core.lib.*` only.
5. **`agentm.llm` deleted.** Each provider lives as
   `extensions/builtin/llm_<provider>.py`. The shared stream accumulator lives
   at `agentm.core.lib.stream`.
6. **Boundary contracts swept.** This doc, `.claude/index.yaml`, CLAUDE.md,
   `extensions/validate.py`, and README all describe the three-layer story
   without coexistence-layer caveats.

Acceptance held: a smoke-test scenario can swap `StreamFn` (mock LLM) and
register a sandbox `Operations` bundle, both as atoms in a manifest, without
editing `agentm.core`. The substrate has no remaining reference to any
concrete provider name or `Local*Operations` class.

### 5.1 Shared Presenter Rendering

CLI and Textual remain presenters, so shared display decisions live in pure
`agentm.core.lib.render` helpers rather than in either mode. The helpers produce
headless strings and token reports only; they do not import Rich, Textual,
runtime, filesystem state, or pricing tables.

Cost is a policy concern exposed as an ExtensionAPI service named
`cost_query`. The `cost_budget` atom owns pricing configuration and registers a
service with `estimate(usage, provider=...)`; presenters call it opportunistically
and fall back to token-only output when the service is absent.

Tool-result display uses tool-declared metadata (`metadata["result_format"]`) or
`api.register_tool_renderer(tool_name, renderer)`. Presenters may choose their
surface-specific rich widget for a returned format, but they must not infer atom
identity from exact tool-name strings.
