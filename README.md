# AgentM SDK Core

This worktree is the reduced AgentM SDK core. It keeps the substrate small:
session lifecycle, trajectory persistence, extension loading, provider streams,
tool execution, operations ports, and observability hooks. CLI, gateway peers,
contrib scenarios, catalog machinery, and product-specific policies are outside
this branch until the core abstraction is stable.

## Core Abstraction

The SDK is mechanism, not policy. Policy enters through atoms.

| Layer | Responsibility |
|---|---|
| `AgentSession` | Public session handle: create, run, prompt, interrupt, spawn, fork, resume, shutdown. |
| `Driver` | Persistent async loop: consume triggers, call the provider, execute tools, commit turns. |
| `Trigger` | Unified input shape for user input, background completion, monitor fire, subagent result, continuation, and injection. |
| `TriggerEnvelope` | Queue/routing metadata around a trigger: priority (`now`/`next`/`later`), target identity, origin, mode, and presenter/system flags. |
| `MessageMeta` | Control-plane metadata for synthetic messages, hidden attachments, no-response prompts, replay policy, token accounting, origin, mode, and target identity. |
| `Turn` | Durable trajectory unit: trigger, assistant rounds, tool records, structured tool extras, outcome, timing, and usage. |
| `TrajectoryStore` | The single replaceable persistence boundary for session metadata, incomplete checkpoints, committed turns, message-node indexes, heads, and cache/compaction state. |
| `TrajectoryNode` | A committed message or compact-boundary index record used for fork, resume, sidechains, compaction, and prompt-cache lookup. Content replacement is state attached atomically to a compact boundary, not another node kind. |
| `ContextPolicy` | Replaceable context reconstruction policy. Durable compaction and cache decisions use `ContentReplacementState` and `PromptCacheState` in the selected `TrajectoryStore`, so they survive resume/fork without a second policy-owned store. |
| `EventBus` | Immutable event dispatch surface for observation and policy hooks. |
| `AtomAPI` | The only surface atoms receive at install time. |
| `CancelSignal` | Cooperative cancellation boundary shared by provider streams, tools, operations, and optionally foreground child sessions. Its typed `reason` carries causes such as user cancel, submit interrupt, shutdown, or task stop. |
| `BackgroundTaskRegistry` | Shared helper for atom-owned detached work: task handles, slot caps, and cooperative cancel. |
| `StreamFn` / `ProviderConfig` | Replaceable LLM provider boundary. |
| `ProviderResolver` | Replaceable active-provider selection policy. |
| `ToolExecutor` | Replaceable tool execution boundary. Tool requirements declare isolation, filesystem/network access, concurrency, and interrupt behavior; executor capabilities declare what the backend can honor. |
| `ToolOrchestrator` | Replaceable batch scheduler for tool calls: exclusive vs parallel-safe partitioning, sibling-error cascading, and cooperative cancellation. |
| `PermissionPolicy` | Replaceable async permission boundary. It returns a final `allow` or `deny`; policies that require user interaction await it internally instead of exposing a deferred runtime state. |
| `BashOperations` / `ResourceWriter` | Replaceable external-world ports for execution and resource mutation. |

## Extension Contract

An atom is a Python module with:

```python
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="my_atom",
    description="What this atom contributes.",
    registers=("tool:my_tool",),
    requires=(),
)

def install(api, config):
    ...
```

Atoms may depend on `agentm.core.abi`, `agentm.core.lib`, and the public
`agentm.extensions` surface. The load-time validator rejects imports from
`agentm.core.runtime` and `agentm.core._internal`; atoms must reach stateful
runtime subsystems through `AtomAPI` methods and services.

## Session Creation

The public path is `AgentSession.create(AgentSessionConfig(...))`.

```python
from pathlib import Path

from agentm import AgentSession, AgentSessionConfig, LoopConfig, builtin_scenario_loader
from agentm.storage.trajectory import JsonlTrajectoryStore

session = await AgentSession.create(AgentSessionConfig(
    cwd=".",
    scenario="minimal",
    scenario_loader=builtin_scenario_loader,
    provider=("agentm.extensions.builtin.llm_openai", {"model": "gpt-4o"}),
    trajectory_store=JsonlTrajectoryStore(Path(".agentm/trajectory")),
    loop_config=LoopConfig(max_turns=8, max_tool_calls=32),
    tool_allowlist=["read", "bash"],
))

messages = await session.run("summarize src/agentm/core/abi")
await session.shutdown()
```

Use `extensions=[...]` to bypass scenario lookup entirely. The core runtime has
no built-in scenario registry; pass `builtin_scenario_loader` to opt into the
packaged scenario names in this reduced worktree:

| Scenario | Meaning |
|---|---|
| `empty` | No atoms; host code must provide provider/tools directly. |
| `minimal` | Observability, local bash operations, retry policy, result caps, file tools, bash tool, and system prompt atom. |

Hosts that need named scenarios beyond these should pass
`AgentSessionConfig.scenario_loader`.

Provider and backend implementations are optional package extras. For example,
install `agentm[provider-openai,packaged-minimal]` when using the example above.

## Persistence

`TrajectoryStore` owns durable session metadata, cumulative incomplete
checkpoints, committed turns, message nodes, explicit heads, and cache/compaction
state. A committed turn and its node/head indexes share one atomic publication
boundary. Root metadata is created automatically before the session starts;
child and forked sessions register their own metadata through the session graph
path. `SessionMeta.config` persists the minimal resumable context
(`root_session_id`, `depth`, `scenario`, and `scenario_dir`) so a child or fork
can be loaded in a later process without losing its lineage.

`AgentSessionConfig(trajectory_store=None)` lets the SDK host resolver select
the configured/default backend. Low-level core factories remain explicitly
ephemeral when no store is supplied. Host programs that
need resume or trace queries must select one store. Provider requests that fail
after retries are persisted as non-replayable `ProviderRequestFailed` turns
before the trigger receipt raises, so failed sessions do not collapse to an
empty session header.

Built-in stores:

| Store | Use |
|---|---|
| `InMemoryTrajectoryStore` | Tests and ephemeral embedding. |
| `JsonlTrajectoryStore` | Local append-only persistence, one JSONL file per session. |
| `PostgresTrajectoryStore` | Durable transactional session, turn, node/head, and policy-state persistence. |

The same `TrajectoryStore` exposes committed message-tree queries with stable
node ids and portable index fields:

| Field group | Fields | Purpose |
|---|---|---|
| Identity | `id`, `session_id`, `root_session_id`, `parent_session_id`, `seq` | Stable lookup, per-session append order, and trace-scope scans. |
| Links | `parent_id`, `logical_parent_id` | Visible chain reconstruction, fork prefix sharing, and compact-boundary lineage. |
| Ownership | `agent_id`, `is_sidechain` | Subagent sidechains and agent-specific resume. |
| Turn join | `turn_id`, `turn_index`, `round_index`, `message_index` | Join message nodes back to committed turns and tool records. |
| Shape | `kind`, `role`, `timestamp` | Distinguish committed message and compact-boundary nodes, then filter message roles such as user, assistant, and tool result. Incomplete checkpoints are turn-level records outside the committed node graph. |

SQL stores implement these as normal indexed columns. JSONL stores replay the
same per-session journal; they do not maintain a separately recoverable sidecar
truth. ClickHouse is an optional OTLP observability backend, not a trajectory
store. The SDK relies on the Protocol semantics, not on a JSONL layout.

## Verification

```bash
uv sync
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest --tb=short
```

Keep `project-index.yaml` synchronized with every code and test change, then
validate it with:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/validate_index.py project-index.yaml
```
