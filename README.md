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
| `TrajectoryStore` | Replaceable persistence boundary for session metadata and committed turns. |
| `TrajectoryNode` / `TrajectoryNodeStore` | Message-level append-only projection for fork, resume, sidechain, compact boundary, snip/rewind, content replacement, and prompt-cache state. |
| `ContextPolicy` | Replaceable context reconstruction policy. Policies may expose `PersistentContextPolicy` state for compaction and content replacement that must survive resume/fork. |
| `EventBus` | Immutable event dispatch surface for observation and policy hooks. |
| `AtomAPI` | The only surface atoms receive at install time. |
| `CancelSignal` | Cooperative cancellation boundary shared by provider streams, tools, operations, and optionally foreground child sessions. `ReasonedCancelSignal` carries host-facing reasons such as user cancel, submit interrupt, shutdown, or task stop. |
| `BackgroundTaskRegistry` | Shared helper for atom-owned detached work: task handles, slot caps, and cooperative cancel. |
| `StreamFn` / `ProviderConfig` | Replaceable LLM provider boundary. |
| `ProviderResolver` | Replaceable active-provider selection policy. |
| `ToolExecutor` | Replaceable tool execution boundary. Tool requirements declare isolation, filesystem/network access, concurrency, and interrupt behavior; executor capabilities declare what the backend can honor. |
| `ToolOrchestrator` | Replaceable batch scheduler for tool calls: exclusive vs parallel-safe partitioning, sibling-error cascading, and cooperative cancellation. |
| `PermissionPolicy` | Replaceable typed permission boundary for user, subagent, policy, classifier, and mode-based allow/deny/defer decisions. |
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

from agentm import AgentSession, AgentSessionConfig, LoopConfig
from agentm.core.runtime.stores.jsonl import JsonlTrajectoryStore

session = await AgentSession.create(AgentSessionConfig(
    cwd=".",
    scenario="minimal",
    provider=("agentm.extensions.builtin.llm_openai", {"model": "gpt-4o"}),
    store=JsonlTrajectoryStore(Path(".agentm/trajectory")),
    loop_config=LoopConfig(max_turns=8, max_tool_calls=32),
    tool_allowlist=["read", "bash"],
))

messages = await session.run("summarize src/agentm/core/abi")
await session.shutdown()
```

Use `extensions=[...]` to bypass scenario lookup entirely. The built-in scenario
names in this reduced worktree are:

| Scenario | Meaning |
|---|---|
| `empty` | No atoms; host code must provide provider/tools directly. |
| `minimal` | Observability, local bash operations, retry policy, result caps, file tools, bash tool, and system prompt atom. |

Hosts that need named scenarios beyond these should pass
`AgentSessionConfig.scenario_loader`.

## Persistence

`TrajectoryStore` owns durable session metadata and committed turns. When a store
is supplied through the SDK factory, root session metadata is created
automatically before the session starts. Child and forked sessions register their
own metadata through the session graph path. `SessionMeta.config` persists the
minimal resumable context (`root_session_id`, `depth`, `scenario`, and
`scenario_dir`) so a child or fork can be loaded in a later process without
losing its lineage.

Built-in stores:

| Store | Use |
|---|---|
| `InMemoryTrajectoryStore` | Tests and ephemeral embedding. |
| `JsonlTrajectoryStore` | Local append-only persistence, one JSONL file per session. |

`TrajectoryNodeStore` is a second persistence/query port, not a replacement for
`TrajectoryStore`. It stores or derives a message tree with stable node ids and
portable index fields:

| Field group | Fields | Purpose |
|---|---|---|
| Identity | `id`, `session_id`, `root_session_id`, `parent_session_id`, `seq` | Stable lookup, per-session append order, and trace-scope scans. |
| Links | `parent_id`, `logical_parent_id` | Visible chain reconstruction, fork prefix sharing, and compact-boundary lineage. |
| Ownership | `agent_id`, `is_sidechain` | Subagent sidechains and agent-specific resume. |
| Turn join | `turn_id`, `turn_index`, `round_index`, `message_index` | Join message nodes back to committed turns and tool records. |
| Shape | `kind`, `role`, `timestamp` | Filter message, compact boundary, content replacement, snip, checkpoint, and user/assistant/tool-result nodes. |

SQL stores should implement these as normal indexed columns. ClickHouse stores
should map the same fields to partition/order/skip-index choices. JSONL stores
may satisfy the same Protocol by scanning or maintaining a sidecar index. The
SDK relies on the Protocol semantics, not on a JSONL layout.

## Verification

```bash
uv sync
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest --tb=short
```

This worktree intentionally has no `project-index.yaml`; requirement-index
validation from main does not apply until the reduced SDK defines its own index.
