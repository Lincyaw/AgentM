# AgentM Core Abstraction Inventory

This document separates fixed SDK/core contracts from implementation backlog.
The refactor target is that future work implements Protocol backends and policy
atoms without reopening trajectory, store, or cancellation modeling.

## Boundary Rule

Do not add optional backends directly into minimal SDK core unless the backend
is required for embedded SDK sessions to run.

Core defines stable ports, invariants, service keys, and default local
behavior. Optional packages provide durable storage, remote execution,
gateway/presenter integrations, and deployment-specific policy.

## Fixed Core Contracts

| Contract | Fixed abstraction | Core files |
| --- | --- | --- |
| Message patterns | `MessageMeta` models synthetic, hidden, replay-only, no-response, target session/agent, mode, and tags. Concrete Claude Code-style producers remain policy atoms or presenters. | `src/agentm/core/abi/messages.py` |
| Cancellation | `CancelSignal` / `CancelSource` carry user cancel, submit interrupt, shutdown, sibling error, and task stop reasons across LLM streams, tool execution, and child sessions. Provider adapters race stream creation and next-event waits against the signal so interrupts do not depend on receiving another network chunk. | `src/agentm/core/abi/cancel.py`, `src/agentm/core/runtime/driver.py`, `src/agentm/core/lib/async_cancel.py` |
| Tool execution | `ToolExecutor` and `ToolOrchestrator` model isolation, interrupt behavior, concurrency, sibling cancellation, and ordered result completion. `FunctionTool` can carry typed `ToolExecutionRequirements`, and bundled file/bash tools declare their filesystem/process requirements. | `src/agentm/core/abi/tool_executor.py`, `src/agentm/core/abi/tool_orchestration.py`, `src/agentm/core/abi/tool.py` |
| Permission | `PermissionPolicy.decide()` is async and returns a final `allow` or `deny`. Policies that need user interaction await internally; the runtime has no half-modeled deferred state. | `src/agentm/core/abi/permission.py` |
| Branch/head trajectory model | Message nodes carry `branch_id` and `head_id`; `TrajectoryHead` is the explicit append point for a session/agent/sidechain chain. Appends never infer "current" from leaves; missing heads are repaired by rebuilding projection from the authoritative turn log. | `src/agentm/core/abi/trajectory.py`, `src/agentm/core/abi/store.py` |
| Store consistency | `TrajectoryStore` is the authoritative turn log. `TrajectoryNodeStore` is a rebuildable projection/read model with `replace_session_projection()` and `projection_status()` for crash/migration recovery. | `src/agentm/core/abi/store.py`, `src/agentm/core/runtime/driver.py` |
| Portable node indexes | Node query/index fields cover root/session/parent/logical parent/branch/head/agent/sidechain/kind/role/turn/round/message/seq/timestamp, plus logical tool-call, cache, content-ref, and visibility lookups. Head indexes are separate. | `src/agentm/core/abi/trajectory.py`, `src/agentm/core/abi/store.py` |
| Async SDK boundary | Store Protocols remain synchronous blocking ports for backend neutrality, but runtime async paths offload store calls with `asyncio.to_thread`. Constructors do not perform store I/O; resume/factory paths own loading. | `src/agentm/core/runtime/session.py`, `src/agentm/core/runtime/session_factory.py`, `src/agentm/core/runtime/driver.py` |
| Fork/resume anchors | `TrajectoryForkPoint` can name a turn prefix, node, or head. `Session.fork()` remains compatible with `TurnRef`; the default turn-based runtime resolves node/head anchors only when they point at a committed turn boundary. Exact mid-turn node-chain forks require `NodeChainContextProjection`. `Session.resume()` rebuilds stale node projection when a node store is present. | `src/agentm/core/abi/trajectory.py`, `src/agentm/core/abi/compaction.py`, `src/agentm/core/runtime/session.py` |
| Prompt cache state | `PromptCacheState` and `ContentReplacementState` persist cache/replacement identity, branch/head, leaf node, and clone provenance. Cloning preserves the source session/leaf while assigning the target leaf. Enforcement remains a context/provider policy. | `src/agentm/core/abi/trajectory.py`, `src/agentm/core/abi/store.py` |
| Content/reference boundary | Trajectory nodes record control facts such as compaction boundaries, cache identity, replacement identity, and `content_ref`. Large prompt content, summaries, files, and long tool results live behind resource/artifact references rather than inside the trajectory control stream. | `src/agentm/core/abi/trajectory.py`, `src/agentm/core/abi/resource.py` |
| Catalog/trajectory separation | Trajectory queries answer what happened in a session. `AtomCatalog.record_active_set()` persists `CatalogActiveSetInput` with root/parent/session/scenario/provider/timestamp context; `AtomCatalogQuery` answers which atoms, tools, providers, scenarios, and versioned resources were active. Presenter/CLI layers may join them for display, but core keeps the APIs separate because their lifecycles differ. | `src/agentm/core/abi/query.py`, `src/agentm/core/abi/catalog.py` |
| Node-chain context projection | `ProjectionInput` carries committed turns plus optional `TrajectoryNodeStore` chain identity. `ContextProjection` remains the turn-only compatibility floor; `NodeChainContextProjection.project_chain()` is the exact replay hook. | `src/agentm/core/abi/compaction.py`, `src/agentm/core/runtime/driver.py` |
| Config provenance and provider freeze | `SESSION_CONFIG_PRECEDENCE` and `ConfigValueProvenance` define comparable resolver output. `ProviderSessionIdentity` freezes provider/model identity after the first committed turn; resume restores the lock from session metadata. Any allowed mid-session change must be explicit `SessionConfigChange` / `config_change` control-node data. | `src/agentm/core/abi/session_api.py`, `src/agentm/core/abi/provider.py`, `src/agentm/core/abi/trajectory.py`, `src/agentm/core/runtime/session.py`, `src/agentm/core/runtime/session_meta.py` |
| Scenario composition boundary | Core session factories create empty SDK sessions from explicit extensions and resolve named scenarios only through host-provided `ScenarioLoader`. Packaged scenario helpers such as `builtin_scenario_loader` live outside `core.runtime`, so embedded SDK users do not inherit CLI/local-tool policy by default. | `src/agentm/core/abi/session_api.py`, `src/agentm/core/runtime/session_factory.py`, `src/agentm/scenarios.py` |
| Resource read/write authority | `ResourceWriter` remains the write transaction boundary. `ResourceReader` is the backend-neutral read side for `ResourceRef`; path-based writer reads remain compatibility for existing file tools. `ResourceRef` namespaces cover workspace, artifact, sandbox, summary, content, catalog, observability, and environment resources. | `src/agentm/core/abi/resource.py`, `src/agentm/core/runtime/session.py` |
| Prompt-cache provider bridge | `ProviderPromptCacheAdapter` maps `PromptCacheState` plus provider-bound messages into provider-specific cache hints. Context policy decides cache boundaries; provider atoms decide vendor syntax. | `src/agentm/core/abi/provider.py`, `src/agentm/core/abi/roles.py` |
| Environment restore policy | Resume restore is fail-fast by default through `EnvironmentRestorePolicy(on_failure="fail")`. Hosts that can enforce degraded behavior may choose `degraded_readonly`, which records `EnvironmentRestoreStatus` instead of silently masking restore failure. | `src/agentm/core/abi/lifecycle.py`, `src/agentm/core/runtime/session.py` |
| Tool/environment composition | `ToolExecutionRequirements` can target an environment id; `ToolExecutionCapabilities` and `ToolExecutionRequest` carry `EnvironmentRef`. `ToolExecutor` owns call isolation/killability/concurrency while `EnvironmentOperations` owns environment identity and world operations. | `src/agentm/core/abi/tool_executor.py`, `src/agentm/core/abi/operations.py`, `src/agentm/core/runtime/tool_executor.py` |
| Capability dependency validation | Manifest dependencies resolve against normalized capability keys such as `service:operations`, `tool:bash`, and `provider:openai`. Bare requirements remain compatibility aliases for `service:<name>` or `atom:<name>`, but builtin atoms use explicit capability references. | `src/agentm/core/abi/manifest.py`, `src/agentm/core/runtime/session_factory.py` |
| File toolbox state boundary | `agentm_toolbox` is the shared file-operation implementation for in-process and sandbox/env execution. CLI mode persists read-before-write state by explicit state file or a stable session/root/environment/cwd namespace and uses a per-namespace lock plus atomic replace. | `src/agentm/extensions/builtin/file_tools.py`, `src/agentm_toolbox/` |
| Observability implementation boundary | Core exposes only `SessionTelemetry` and query Protocols. OTel exporters, OTLP parsing, collector discovery, local JSONL trace files, and loguru bridging are extension/backend implementation details. The builtin observability atom makes export fallback explicit through `export = auto | local_file | otlp`. | `src/agentm/core/abi/telemetry.py`, `src/agentm/core/abi/query.py`, `src/agentm/extensions/builtin/observability.py`, `src/agentm/extensions/observability/` |

## Recovery Boundaries

| Path | Behavior | Reason |
| --- | --- | --- |
| Missing trajectory head with committed turns | Rebuild the node projection from the authoritative `TrajectoryStore` turn log, then retry explicit-head lookup. If the rebuilt projection still has no active head, fail fast. | Node projection is a repairable read model; append order must not fall back to leaf inference. |
| Node projection append/rebuild failure after turn commit | Log the projection failure; preserve the already committed turn log. | The turn log is authoritative. A broken projection should be repairable without rolling back a durable turn commit. |
| Node-chain projection without node store/head | Fail fast when a `NodeChainContextProjection` is installed but no `TrajectoryNodeStore` or active head can supply its chain. | Exact replay must not silently widen back to turn replay. |
| Environment restore failure on resume | Default `EnvironmentRestorePolicy` raises `EnvironmentRestoreError`. Only explicit `degraded_readonly` records `EnvironmentRestoreStatus` and continues. | External world state is part of trajectory correctness; degraded resume must be host-declared. |
| Provider resolver fallback before first commit | Resolver/fallback selection can choose the initial provider before history exists. After the first committed turn, `ProviderSessionIdentity` pins provider/model identity. | Provider selection is policy before history; after history, silent model drift corrupts replay semantics. |

## Store Backend Contract

All trajectory node backends should implement the same logical schema even when
physical storage differs.

Required node identity and ordering fields:

- `id`
- `session_id`
- `root_session_id`
- `parent_session_id`
- `seq`
- `parent_id`
- `logical_parent_id`
- `branch_id`
- `head_id`

Required turn/message join fields:

- `turn_id`
- `turn_index`
- `round_index`
- `message_index`
- `kind`
- `role`
- `visibility`
- `timestamp`

Required routing/query fields:

- `agent_id`
- `is_sidechain`
- `tool_call_id`
- `tool_name`
- `cache_key`
- `content_ref`

The `tool_call_id` and `tool_name` fields are logical inverted indexes. A
message node may contain multiple tool calls/results; SQL stores may normalize
them into a child table, while JSONL stores may scan arrays.

Head storage is separate from node storage. A backend must support a unique
current head per `(session_id, head_id)` and efficient branch/agent/sidechain
selection by `(root_session_id, session_id, branch_id, agent_id, is_sidechain)`.

## Remaining Implementation Backlog

| Capability | Design decision | Why it remains open | Fixed protocol to implement | Likely home |
| --- | --- | --- | --- | --- |
| Durable `TrajectoryNodeStore` backends | Land Postgres first as the durable owner of node/head state, add JSONL sidecar second for local portability, and use ClickHouse as a read/query mirror rather than the sole current-head owner. | Core has only the in-memory reference implementation. | Implement `TrajectoryNodeStore`, `TRAJECTORY_NODE_INDEXES`, `TRAJECTORY_HEAD_INDEXES`, projection status, head compare-and-advance, and rebuild for the selected backend. | SDK storage extra or host backend. |
| Exact node-chain context replay | The first exact replay abstraction is generic `NodeChainContextProjection`; Claude Code-style cache/content replacement is layered policy, not the replay substrate. | Runtime now passes `ProjectionInput.nodes` to chain-aware projections, but no concrete projection strategy exists. | Implement `NodeChainContextProjection.project_chain()` for exact mid-turn node/head replay, compact-boundary traversal, and sidechain visibility. | Optional builtin atom or host policy. |
| Prompt-cache/content-replacement policy | Cache and replacement boundaries are trajectory control facts. Large content, summaries, and replacement payloads live behind `content_ref` in a resource/artifact store. | State, projection hooks, resource dereference, and provider cache adapter ABI are typed, but no policy enforces deterministic replacement or provider cache identity. | Implement a `ContextPolicy`/`NodeChainContextProjection` plus provider adapter glue using `PromptCacheState`, `ContentReplacementState`, node `cache_key`, node `content_ref`, `ResourceReader`, and `ProviderPromptCacheAdapter`. | Optional builtin atom plus provider adapters. |
| Claude Code message-pattern policy atoms | Core owns metadata, cancellation, and message invariants. Builtin policy atoms may produce memory reminders, budget nudges, and content/cache policy messages; presenter/gateway packages own UI task notifications, plan-mode UI, teammate mailboxes, and local-command caveats. | Core supports the metadata and interrupt/tool-result mechanics, but concrete producers are still policy. | Implement atoms/presenters for hook output, task notifications, plan-mode messages, local-command caveats, memory reminders, teammate/channel mailboxes, and token/budget nudges. | Optional builtin atoms or presenter/gateway packages. |
| Fork/resume/cache E2E coverage | Add coverage only after durable node-store replay and cache/content replacement behavior are declared load-bearing. | Existing tests protect current runtime paths, but not full durable node-store replay/cache stability. | Add an E2E scenario around node-store fork/resume plus content-replacement cache stability. | Integration tests. |
| Durable catalog backend | Catalog storage remains separate from trajectory storage because catalog identity describes active capabilities, not session event order. | Current catalog/resource store is in-memory. Composition identity now includes root/parent/session/scenario/provider context, but it is not durable across process restarts unless the host persists it. | Implement durable `AtomCatalog`, `AtomCatalogQuery`, and `VersionedResourceStore` backends over `CatalogActiveSetInput` / `CatalogActiveSetRecord`. | SDK storage extra or host backend. |
| Indexed catalog query | Keep catalog query separate from `TrajectoryQueryStore`; CLI/UI can aggregate the two query surfaces when they need joined views. | The query Protocol and in-memory implementation exist, but no durable indexed implementation exists. | Implement `AtomCatalogQuery.query_active_sets()` predicates over session, atom name, module path, digest, version, scenario, provider, requirements, and provided capabilities. | Catalog/query extra. |
| Remote observability query | Events and spans remain observability data. They can be correlated with trajectory by ids, but they are not trajectory nodes or catalog records. | Local `TrajectoryQueryStore` covers sessions/turns only. Events/spans belong to observability backends. | Implement `TrajectoryQueryStore.events()` and `.spans()` over OTLP files, ClickHouse, collector export, or a host service. | Observability extra. |
| Concrete `SessionSpecResolver` | Precedence is explicit SDK/CLI args, then atom overrides or `--set`, then env and `.env`, then project config, then user config/profile/default model, then scenario manifest defaults, then provider defaults. | Resolver precedence and typed provenance are fixed, but no default host policy reads actual config files/env layers. | Implement `SessionSpecResolver` that emits `ResolvedSessionSpec.value_provenance` for CLI flags, env vars, user config, project config, scenario config, atom config, and provider profiles. | CLI/presenter package or SDK helper. |
| Provider/session config changes | Freeze active provider/model identity after the first committed turn. Switching provider requires a fork/new session or explicit `config_change` control-node data. | Runtime prevents silent drift and persists/restores `ProviderSessionIdentity`; `SessionConfigChange` models explicit changes, but no policy emits or authorizes them. | Implement a provider/config-change policy if mid-session changes become load-bearing. | Host policy or optional atom. |
| Resource/artifact backends | Workspace files, artifacts, sandbox files, summaries, and large content use one `ResourceRef` namespace model; backends map namespaces to physical storage. | Namespace constants and `ResourceReader` are fixed, but concrete resource/artifact backends are not implemented. | Implement `ResourceReader`/`ResourceWriter` backends that can resolve `content_ref` without coupling trajectory to storage layout. | Resource backend extra or host policy. |
| Sandbox/remote environment backend | Remote execution belongs in `EnvironmentOperations`; tool isolation composes with it through `ToolExecutor`. | Local `EnvironmentOperations` exists. Remote/sandbox/agent-env execution brings deployment dependencies and auth. | Implement `EnvironmentOperations` backends with environment identity, cwd mapping, file transfer, cancellation, logs, snapshots, and lifecycle. | Environment backend extra. |
| Environment snapshot persistence | Resume restore is fail-fast by default. A host can explicitly opt into degraded read-only resume, but that is a policy choice outside core invariants. | `EffectScope.fork_at()`, `restore()`, `EnvironmentRestorePolicy`, and `EnvironmentRestoreStatus` are wired, but concrete snapshot storage is backend-specific. | Implement snapshot persistence and host degraded-mode enforcement. | Environment backend extra plus host policy. |
| Process/sandbox `ToolExecutor` | Implement sandboxed tool execution as a composition: `ToolExecutor` owns call isolation, killability, and concurrency; `EnvironmentOperations` owns environment identity, files, processes, logs, and snapshots. | Requirements/capabilities include environment identity, but only direct execution is built in. | Implement `ToolExecutor` backends for process/sandbox isolation, killability, concurrency, filesystem, and network controls. | Tool execution extra or environment backend. |
| Concrete `ContextProjection` strategies | Compaction summaries are not catalog or observability records. Trajectory records the boundary and replacement control facts, summary payloads live behind `content_ref`, and `ProjectionReport` records projection metadata. | Projection service is wired, but no real compaction/summarization strategy exists. | Implement `ContextProjection.project()` and `ProjectionReport` persistence for summarization/compaction. | Optional builtin atom or host service. |
| Capability dependency validation | Atom dependencies should target capabilities/services, not atom names. Atom names remain useful for pinning, conflicts, provenance, and debugging. | Runtime install planning resolves capability dependencies and records normalized capability metadata. Static authoring lint can still be stricter. | Extend `agentm validate` authoring feedback to prefer explicit capability requirements and flag ambiguous bare requirements outside compatibility contexts. | Extension validation and catalog package. |
| Minimal SDK baseline packaging | Core baseline contains no scenario strategy. The SDK baseline is session/runtime protocols plus in-memory reference implementations; packaged `minimal`/`chatbot` scenarios compose provider, operations, file tools, observability, and safety atoms through explicit loaders. | Core package dependencies are reduced to core runtime needs; provider, OTel, and tokenizer dependencies are optional extras. A future distribution split may still move bundled atoms/scenarios into separate wheels. | Keep packaged scenario helpers outside `core.runtime`, and implement any later wheel split without changing session/trajectory protocols. | SDK packaging and bundled scenarios. |
| Presenter/gateway/authoring capabilities | Presenter, gateway, and authoring workflows stay outside minimal core and consume the stable SDK protocols. | These are workflows around the SDK, not minimal runtime invariants. | Implement frontmatter, renderers, child wire forwarding, trace CLI extensions, prompt/skill authoring helpers. | Presenter, gateway, or authoring packages. |

## Recommended Order

| Step | Capability | Gate |
| --- | --- | --- |
| 1 | Implement the Postgres durable `TrajectoryNodeStore` and migration harness against the fixed node/head index contract. | Head compare-and-advance and projection rebuild are reliable under concurrent appends. |
| 2 | Add the JSONL sidecar node-store backend for local portability. | It follows the same logical indexes as Postgres, even if implemented by scan plus sidecar metadata. |
| 3 | Add ClickHouse trajectory query mirroring. | ClickHouse is query/read-model storage only; it does not own current heads. |
| 4 | Implement exact node-chain `ContextProjection` replay. | Mid-turn fork/resume no longer has to widen to committed turn boundaries. |
| 5 | Implement prompt-cache/content-replacement policy and provider adapter glue. | Cache identity and `content_ref` behavior are deterministic across fork/resume. |
| 6 | Add fork/resume/cache E2E coverage once the behavior is declared load-bearing. | The E2E asserts through public CLI/SDK and trace surfaces. |
| 7 | Implement durable catalog backend and indexed catalog query. | Catalog query remains separate from trajectory query. |
| 8 | Implement concrete `SessionSpecResolver` policy and provider/session config freeze. | Config provenance is durable, and provider/model drift is explicit. |
| 9 | Implement resource namespace conventions and read/write authority policy. | `ResourceRef` can address workspace, artifact, sandbox, summary, and large-content storage consistently. |
| 10 | Implement real compaction/summarization `ContextProjection`. | Summary payloads live behind `content_ref`; trajectory records the control boundary. |
| 11 | Implement sandbox/remote `EnvironmentOperations` and snapshot persistence. | Resume restore policy is fail-fast unless a host explicitly opts into degraded mode. |
| 12 | Implement process/sandbox `ToolExecutor`. | Tool execution composes with environment capabilities and preserves cancellation semantics. |
| 13 | Implement remote observability query. | Events/spans correlate with trajectory ids without becoming trajectory records. |
| 14 | Implement Claude Code message-pattern policy atoms. | Core metadata is sufficient; producers are policy/presenter choices. |
| 15 | Split presenter/gateway/authoring packages. | Minimal SDK core remains scenario-neutral. |
