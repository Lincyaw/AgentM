# AgentM Trajectory Refactor Abstraction Inventory

This document is a design inventory only. It does not approve the current
worktree deletions as final implementation. Its purpose is to identify the
root invariant behind each removed, weakened, or suspicious abstraction before
any further code deletion or rewrite.

Current evidence used:

- Recent commits include `2d755b9a refactor: align core + extensions to v2
  trajectory model` followed by `392869c1 remove legacy impl` and
  `5b292c25 remove legacy impl`.
- Current worktree deletes several `core/abi`, `core/lib`, observability, and
  builtin backend files.
- Current runtime has the new trajectory model: immutable committed `Turn`s,
  one active `Execution`, `TrajectoryStore`, explicit `ScenarioLoader`,
  scoped `ServiceRegistry`, `ResourceWriter`, provider registry/resolver, and
  atom install priority.

Non-negotiable design rules for the next implementation pass:

- Do not delete an abstraction because it has few current in-repo references.
- First identify the invariant, then decide the new shape.
- Core capabilities are recovered or redesigned, not erased.
- Atoms should not mutate external world state through arbitrary lifecycle
  callbacks; world effects need typed ports.
- `ResourceWriter` may grow into an effect-aware resource layer, but it must not
  absorb catalog identity into an unbounded "everything writer".
- Catalog identity is SDK composition identity, not ordinary file mutation.
- Lifecycle is trajectory-to-world consistency, not a generic event hook.
- Every service must declare scope: session, tree, process, host, or resource
  namespace.

## Current Deletion Coverage

All currently deleted files are covered below.

| Deleted file | Inventory item |
| --- | --- |
| `src/agentm/core/abi/lifecycle.py` | Lifecycle / world state recovery |
| `src/agentm/core/abi/catalog.py` | Catalog / versioned identity |
| `src/agentm/core/abi/tool_executor.py` | ToolExecutor / execution isolation |
| `src/agentm/core/abi/compaction.py` | Compaction / turn numbering |
| `src/agentm/core/lib/turns.py` | Compaction / turn numbering |
| `src/agentm/core/abi/project_layout.py` | ProjectLayout / artifact namespace |
| `src/agentm/core/lib/artifact_files.py` | ProjectLayout / artifact namespace |
| `src/agentm/core/abi/prompt_template.py` | PromptTemplate / shared atom ABI |
| `src/agentm/core/abi/skill.py` | Skill / shared atom ABI |
| `src/agentm/core/abi/command.py` | Command / shared atom ABI |
| `src/agentm/core/abi/presenter.py` | Presenter / rendering surface |
| `src/agentm/core/lib/render.py` | Presenter / rendering surface |
| `src/agentm/core/lib/atom_config.py` | ConfigResolver / scenario loader |
| `src/agentm/core/lib/user_config.py` | ConfigResolver / scenario loader |
| `src/agentm/core/lib/trace_reader.py` | TraceReader / observability query |
| `src/agentm/core/observability/clickhouse.py` | TraceReader / observability query |
| `src/agentm/core/lib/message_codec.py` | Trajectory serialization |
| `src/agentm/core/lib/frontmatter.py` | Prompt/skill parsing utility |
| `src/agentm/core/lib/child_wire.py` | Child session visibility |
| `src/agentm/core/lib/shutdown.py` | BackgroundTask / shutdown semantics |
| `src/agentm/extensions/builtin/_agent_env.py` | Operations / environment backend |
| `src/agentm/extensions/builtin/bash/agent_env.py` | Operations / environment backend |
| `src/agentm/extensions/builtin/bash/local.py` | Operations / environment backend |
| `src/agentm/extensions/builtin/bash/__init__.py` | Operations / environment backend |
| `src/agentm/extensions/builtin/writer/agent_env.py` | ResourceWriter / environment backend |
| `src/agentm/extensions/builtin/writer/__init__.py` | ResourceWriter / environment backend |

## 1. Lifecycle / World State Recovery

1. Abstract name: Lifecycle / world state recovery.
2. Current related files:
   `src/agentm/core/abi/lifecycle.py`,
   `src/agentm/core/runtime/session.py`,
   `src/agentm/core/runtime/driver.py`,
   `src/agentm/core/runtime/trajectory.py`,
   `src/agentm/core/abi/resource.py`,
   `src/agentm/core/abi/operations.py`.
3. Root requirement / invariant:
   A committed trajectory is only replayable, forkable, and resumable if the
   external world state can be brought to the same logical point as the
   committed turns. File writes, artifacts, command side effects, sandbox
   state, and child sessions must not drift away from the trajectory branch
   they claim to represent.
4. Needed under the trajectory model:
   Yes. This is a core invariant. It becomes more important with immutable
   turns because the trajectory is now the durable causal record.
5. Problem with old design:
   `LifecycleHook` is too unconstrained. It lets any atom run arbitrary code on
   fork, resume, replay, or abandon. Ordering, idempotency, failure semantics,
   ownership, and resource coverage are implicit. It is a hook bus, not a
   typed effect boundary.
6. Best new abstraction candidate:
   Split lifecycle into typed effect ports:

   ```python
   class EffectScope(Protocol):
       async def begin_turn(self, turn_id: str) -> "EffectTxn": ...
       async def commit_turn(self, txn: "EffectTxn", turn: Turn) -> None: ...
       async def abandon_turn(self, txn: "EffectTxn") -> None: ...
       async def fork_at(self, ref: TurnRef, *, child_session_id: str) -> "EffectScope": ...
       async def restore(self, session_id: str, turns: Sequence[Turn]) -> None: ...
   ```

   For sandbox/process backends, add a narrower world backend:

   ```python
   class EnvironmentSnapshotter(Protocol):
       async def snapshot(self, *, session_id: str, ref: TurnRef) -> str: ...
       async def fork_from(self, snapshot_id: str, *, child_session_id: str) -> None: ...
       async def restore_to(self, snapshot_id: str) -> None: ...
   ```
7. Boundary with other abstractions:
   `Lifecycle` coordinates world consistency. `ResourceWriter` records durable
   resource mutations. `Operations` executes commands in a backend. Catalog
   records composition identity. Observability records evidence. Lifecycle
   should call typed ports, not arbitrary atom callbacks.
8. Recommendation:
   Restore/pause deletion. Rewrite the abstraction. Do not keep generic
   `LifecycleHook` as-is.
9. Migration steps:
   Restore the deleted lifecycle types enough to keep fork/resume semantics
   visible; replace `register_lifecycle_hook` with registration of typed
   effect/environment ports; make `Session.fork`, `Session.resume`, and turn
   abandon call those ports; add idempotency requirements before tests.
10. Questions to confirm:
   Should resource restore be automatic on `Session.resume`, or should hosts
   explicitly choose restore policy? Should a failed restore fail session
   creation or create a degraded read-only session?

## 2. Catalog / Versioned Identity / Active-Set Fingerprint

1. Abstract name: Catalog / versioned composition identity.
2. Current related files:
   `src/agentm/core/abi/catalog.py`,
   `core-manifest.yaml`,
   `src/agentm/core/abi/manifest.py`,
   `src/agentm/core/runtime/extension.py`,
   `src/agentm/extensions/validate.py`,
   observability session fingerprint events.
3. Root requirement / invariant:
   Every run must be attributable to the exact core, scenario, atom source, and
   manifest versions that produced it. Without this, trajectory evidence cannot
   support replay, audit, self-modification, rollback, or performance
   attribution.
4. Needed under the trajectory model:
   Yes. Trajectory records what happened; catalog identity records which
   executable composition made it happen.
5. Problem with old design:
   The old `CatalogService` mixed version storage, path layout helpers, run
   lookup, source hashing, manifest snapshots, and decision log paths. That
   made it look like an implementation detail rather than a core identity
   boundary.
6. Best new abstraction candidate:
   Separate generic versioned resource storage from typed atom identity:

   ```python
   @dataclass(frozen=True)
   class ResourceVersion:
       namespace: str
       name: str
       digest: str
       metadata: Mapping[str, object]

   class VersionedResourceStore(Protocol):
       def put(self, namespace: str, name: str, content: bytes, metadata: Mapping[str, object]) -> ResourceVersion: ...
       def get(self, version: ResourceVersion) -> bytes: ...
       def alias(self, namespace: str, name: str, version: ResourceVersion) -> None: ...
       def fingerprint(self, versions: Sequence[ResourceVersion]) -> str: ...

   class AtomCatalog(Protocol):
       def freeze_atom(self, module_path: str, source: bytes, manifest: ExtensionManifest) -> ResourceVersion: ...
       def active_set(self, *, core: ResourceVersion | None, scenario: ResourceVersion | None, atoms: Mapping[str, ResourceVersion]) -> str: ...
       def record_run(self, session_id: str, active_set: str) -> None: ...
   ```
7. Boundary with other abstractions:
   Catalog is not `ResourceWriter`. `ResourceWriter` mutates user/workspace
   resources. Catalog freezes SDK composition identity. They may share a
   content-addressed store implementation, but not the same public port.
8. Recommendation:
   Restore/pause deletion. Rewrite as `VersionedResourceStore` plus
   `AtomCatalog`.
9. Migration steps:
   Reintroduce the identity types; remove path helpers from ABI; make extension
   installation optionally freeze source/manifest; stamp active-set fingerprint
   into `SessionMeta` and observability.
10. Questions to confirm:
   Is catalog required in minimal embedded SDK sessions, or can hosts opt into
   catalog identity? Should missing catalog be allowed only for ephemeral tests?

## 3. ResourceWriter / Versioned Resources / Transaction Boundary

1. Abstract name: ResourceWriter / durable resource mutation.
2. Current related files:
   `src/agentm/core/abi/resource.py`,
   `src/agentm/core/runtime/session.py`,
   `src/agentm/core/runtime/session_factory.py`,
   `src/agentm/extensions/builtin/file_tools.py`,
   `src/agentm/extensions/builtin/writer/agent_env.py`.
3. Root requirement / invariant:
   Any durable resource mutation produced by an agent must go through a
   host-controlled boundary that can classify paths, enforce protection,
   audit rationale, group mutations, and integrate with trajectory effects.
4. Needed under the trajectory model:
   Yes. Resource mutations are external effects of turns.
5. Problem with old/current design:
   The current `ResourceWriter` is a good start but does not yet bind writes to
   turn ids, resource versions, effect transactions, or restore/fork semantics.
   It is also too file-shaped if artifacts and sandbox resources are first
   class.
6. Best new abstraction candidate:
   Add resource refs, versions, and transaction context:

   ```python
   @dataclass(frozen=True)
   class ResourceRef:
       namespace: str
       path: str

   @dataclass(frozen=True)
   class ResourceMutation:
       ref: ResourceRef
       op: Literal["write", "replace", "delete"]
       before: str | None
       after: str | None

   class ResourceTxn(Protocol):
       async def write(self, ref: ResourceRef, content: bytes) -> ResourceMutation: ...
       async def commit(self) -> Sequence[ResourceMutation]: ...
       async def abandon(self) -> None: ...
   ```
7. Boundary with other abstractions:
   ResourceWriter owns resource mutation. Lifecycle owns when to begin, commit,
   abandon, fork, or restore those mutations. Catalog owns SDK composition
   identity. Artifact namespace can be a resource namespace.
8. Recommendation:
   Keep and rewrite. Do not delete.
9. Migration steps:
   Keep current read/write methods temporarily; add transaction-aware methods;
   bind file tools to a per-turn resource transaction; reintroduce sandbox
   writer backend against the new protocol.
10. Questions to confirm:
   Should `ResourceWriter` cover reads, or should read access be a separate
   `ResourceReader` to make mutation authority explicit?

## 4. ToolExecutor / Execution Isolation / Backend Selection

1. Abstract name: ToolExecutor / execution isolation.
2. Current related files:
   `src/agentm/core/abi/tool.py`,
   `src/agentm/core/abi/tool_executor.py`,
   `src/agentm/core/runtime/tool_executor.py`,
   `src/agentm/core/runtime/driver.py`,
   `src/agentm/core/abi/operations.py`.
3. Root requirement / invariant:
   Tool execution must have clear cancellation, isolation, failure, timeout,
   and resource ownership semantics. Blocking or unsafe tools must not corrupt
   the session loop.
4. Needed under the trajectory model:
   Yes. Tool calls are recorded in `ToolRecord`; their execution boundary is
   part of turn causality.
5. Problem with old design:
   `metadata["execution_domain"]` couples tool declarations to runtime
   substrate decisions. Process-domain execution requires pickleable tools and
   silently changes what side effects can reach the parent. Sandbox was exposed
   as a declared domain before it had a complete backend contract.
6. Best new abstraction candidate:
   Keep execution policy runtime-owned, but let tools declare requirements:

   ```python
   @dataclass(frozen=True)
   class ToolExecutionRequirements:
       isolation: Literal["none", "thread", "process", "environment"]
       killable: bool = False
       filesystem: Literal["none", "read", "write"] = "none"
       network: bool = False

   class ToolExecutor(Protocol):
       async def execute(self, tool: Tool, args: Mapping[str, object], *, signal: CancelSignal | None, requirements: ToolExecutionRequirements | None) -> ToolResult | ToolOutcome: ...
   ```
7. Boundary with other abstractions:
   ToolExecutor executes the tool coroutine. `Operations` is the backend for
   shell/environment work. ResourceWriter records durable mutations. Lifecycle
   coordinates effect commit/abandon.
8. Recommendation:
   Pause deletion. Rewrite ABI as requirements plus runtime executor, or keep
   executor runtime-private while exposing requirement metadata in a typed way.
9. Migration steps:
   Restore capability inventory for thread/process/sandbox; remove direct
   domain constants from public ABI only after replacement requirements are
   present; route shell/sandbox isolation through Operations.
10. Questions to confirm:
   Should third-party tools be allowed to request process isolation, or should
   only host policy choose isolation?

## 5. Operations / BashOperations / Environment Backend

1. Abstract name: Operations / environment backend.
2. Current related files:
   `src/agentm/core/abi/operations.py`,
   `src/agentm/extensions/builtin/operations.py`,
   `src/agentm/extensions/builtin/tool_bash.py`,
   `src/agentm/extensions/builtin/bash/local.py`,
   `src/agentm/extensions/builtin/bash/agent_env.py`,
   `src/agentm/extensions/builtin/_agent_env.py`.
3. Root requirement / invariant:
   Tools that interact with execution environments must target a replaceable
   backend: local process, remote sandbox, ARL environment, or host-defined
   environment.
4. Needed under the trajectory model:
   Yes. Command execution is a world effect tied to turns.
5. Problem with old/current design:
   `BashOperations` is useful but narrow. Deleted local and agent-env backends
   were implementation-heavy, but removing them removes evidence of the
   required backend seam. Current operations are also session-scoped without a
   clear environment identity.
6. Best new abstraction candidate:
   Keep `BashOperations`, add environment identity and lifecycle:

   ```python
   @dataclass(frozen=True)
   class EnvironmentRef:
       id: str
       kind: Literal["local", "sandbox", "remote"]

   class EnvironmentOperations(Protocol):
       @property
       def ref(self) -> EnvironmentRef: ...
       @property
       def bash(self) -> BashOperations: ...
       async def snapshot(self) -> str | None: ...
       async def close(self) -> None: ...
   ```
7. Boundary with other abstractions:
   Operations executes side effects. Lifecycle snapshots/restores them.
   ToolExecutor chooses where tool code runs. ResourceWriter handles durable
   file/artifact mutations.
8. Recommendation:
   Restore/pause backend deletion. Move heavy ARL implementation out of core if
   needed, but keep the backend port and at least one local backend.
9. Migration steps:
   Restore local backend as minimal default; define optional sandbox backend
   package boundary; make operations registration host-injectable and
   environment-scoped.
10. Questions to confirm:
   Should minimal SDK ship local operations by default, or require host
   injection for all execution?

## 6. BackgroundTask / Detached Work / Shutdown

1. Abstract name: BackgroundTask / detached work lifecycle.
2. Current related files:
   `src/agentm/core/lib/background_tasks.py`,
   `src/agentm/core/lib/shutdown.py`,
   `src/agentm/core/runtime/session.py`,
   child/subagent/workflow/monitor atoms when present.
3. Root requirement / invariant:
   Detached work must have identity, cancellation, result delivery, shutdown
   drainage, and capacity accounting. A session must not exit while owned work
   is in an undefined state.
4. Needed under the trajectory model:
   Yes. Background completions are triggers or future turns, not loose tasks.
5. Problem with old/current design:
   Generic `BackgroundTaskRegistry` is useful, but status strings and delivery
   markers are owner-defined. Shutdown constants in a separate helper are easy
   to delete as "small", but they encode cross-atom behavior.
6. Best new abstraction candidate:
   Runtime-owned task supervisor:

   ```python
   class DetachedWorkSupervisor(Protocol):
       async def start(self, spec: WorkSpec) -> WorkHandle: ...
       async def cancel(self, work_id: str) -> None: ...
       async def drain(self, *, grace_seconds: float) -> None: ...
       def snapshot(self) -> Sequence[WorkStatus]: ...
   ```
7. Boundary with other abstractions:
   Supervisor owns task lifecycle. Trigger queue owns delivery to the driver.
   Lifecycle/effects own resource cleanup.
8. Recommendation:
   Keep/rewrite. Do not delete shutdown semantics.
9. Migration steps:
   Promote background task registry from lib utility to runtime service;
   replace free-text statuses with typed states; bind completion to triggers.
10. Questions to confirm:
   Should child sessions be supervised by the same detached-work service as
   background exec, or should child sessions have a dedicated supervisor?

## 7. ConfigResolver / ScenarioLoader / Atom Config Precedence

1. Abstract name: ConfigResolver / scenario loader.
2. Current related files:
   `src/agentm/core/lib/atom_config.py`,
   `src/agentm/core/lib/user_config.py`,
   `src/agentm/core/runtime/session_factory.py`,
   `src/agentm/core/abi/session_api.py`,
   `src/agentm/core/abi/manifest.py`.
3. Root requirement / invariant:
   Session composition must be reproducible: scenario, extensions, provider,
   atom config, env, user config, and explicit overrides need one deterministic
   precedence model.
4. Needed under the trajectory model:
   Yes. A trajectory without its resolved composition/config is incomplete
   evidence.
5. Problem with old/current design:
   Previous config logic mixed file lookup, env binding, CLI override, provider
   profile, and scenario discovery. Current `ScenarioLoader` correctly moves
   lookup to host policy, but atom config precedence is not yet a first-class
   host-resolved object.
6. Best new abstraction candidate:

   ```python
   @dataclass(frozen=True)
   class ResolvedSessionSpec:
       scenario: str
       extensions: tuple[ExtensionSpec, ...]
       atom_config: Mapping[str, Mapping[str, object]]
       provider: ExtensionSpec | None
       provenance: Mapping[str, object]

   class SessionSpecResolver(Protocol):
       def resolve(self, request: AgentSessionConfig) -> ResolvedSessionSpec: ...
   ```
7. Boundary with other abstractions:
   ScenarioLoader resolves names to extension specs. ConfigResolver resolves
   config sources. Catalog freezes the resolved composition. SessionFactory
   only instantiates.
8. Recommendation:
   Restore design, rewrite implementation. Do not embed home/contrib lookup in
   session factory.
9. Migration steps:
   Keep `ScenarioLoader`; add `SessionSpecResolver`; move env/user config
   parsing behind host policy; stamp resolved spec provenance into store/meta.
10. Questions to confirm:
   Should SDK itself parse `$AGENTM_HOME/config.toml`, or should CLI/host pass
   an already resolved provider and config?

## 8. TraceReader / Observability Query View / ClickHouse Backend

1. Abstract name: TraceReader / observability query view.
2. Current related files:
   `src/agentm/core/lib/trace_reader.py`,
   `src/agentm/core/observability/clickhouse.py`,
   `src/agentm/core/observability/event_otel.py`,
   `src/agentm/core/observability/otel_export.py`,
   `src/agentm/core/observability/otlp.py`,
   `src/agentm/core/abi/store.py`.
3. Root requirement / invariant:
   Trajectory and observability data must be queryable by session and trace
   across local and remote storage. Debugging and evaluation depend on a stable
   read model.
4. Needed under the trajectory model:
   Yes. The trajectory model increases the need for reliable query views.
5. Problem with old design:
   `TraceReader` was tied to OTLP JSONL details and re-exported through ABI.
   ClickHouse backend was CLI/query implementation inside core observability,
   with SQL details near core code.
6. Best new abstraction candidate:

   ```python
   class TrajectoryQueryStore(Protocol):
       def sessions(self, filter: SessionFilter) -> Iterable[SessionIdentity]: ...
       def turns(self, session_id: str) -> Iterable[Turn]: ...
       def events(self, session_id: str) -> Iterable[EventRecord]: ...
       def spans(self, session_id: str) -> Iterable[SpanRecord]: ...
   ```
7. Boundary with other abstractions:
   `TrajectoryStore` writes committed turns. Observability exports events/spans.
   QueryStore reads either local JSONL, ClickHouse, or another backend.
8. Recommendation:
   Do not delete query seam. Move ClickHouse to a backend module/package if
   minimal SDK should not own it.
9. Migration steps:
   Reintroduce query protocol; implement local JSONL and optional ClickHouse
   adapters; keep `agentm trace` or external tooling on the protocol.
10. Questions to confirm:
   Should remote trace querying be in the SDK package, or in an optional
   observability extra?

## 9. Compaction / Turn Numbering / Context Rebuild

1. Abstract name: Compaction / turn numbering / context rebuild.
2. Current related files:
   `src/agentm/core/abi/compaction.py`,
   `src/agentm/core/lib/turns.py`,
   `src/agentm/core/abi/context.py`,
   `src/agentm/core/abi/trajectory.py`.
3. Root requirement / invariant:
   Long-running sessions need bounded context while preserving recoverability
   and stable references to original turns.
4. Needed under the trajectory model:
   Yes, but the old branch-entry model may conflict with immutable `Turn`.
5. Problem with old design:
   It refers to `SessionEntry`, compaction entries, and branch summaries from a
   previous session tree design. Turn numbering was coupled to message-entry
   enumeration rather than the new `Turn.index`.
6. Best new abstraction candidate:

   ```python
   class ContextProjection(Protocol):
       def project(self, turns: Sequence[Turn], budget: ContextBudget) -> Sequence[AgentMessage]: ...
       def explain(self) -> ProjectionReport: ...
   ```

   Compaction becomes one projection strategy, not a second history model.
7. Boundary with other abstractions:
   Trajectory remains source of truth. Context projection decides what goes to
   the provider. Observability records projection metadata. ResourceWriter may
   store summaries as artifacts, but not as replacement trajectory.
8. Recommendation:
   Pause deletion. Rewrite around `Turn` projection.
9. Migration steps:
   Replace `SessionEntry` compaction types with projection records keyed by
   turn ranges; use `Turn.index`; define read-history as trajectory query, not
   branch materialization.
10. Questions to confirm:
   Should compaction summaries be committed into trajectory as synthetic turns,
   or stored as projection artifacts outside the trajectory?

## 10. ProjectLayout / Artifact Namespace

1. Abstract name: ProjectLayout / artifact namespace.
2. Current related files:
   `src/agentm/core/abi/project_layout.py`,
   `src/agentm/core/lib/artifact_files.py`,
   `src/agentm/core/abi/resource.py`,
   `src/agentm/core/abi/store.py`.
3. Root requirement / invariant:
   Session outputs, artifacts, prompts, skills, and observability files need
   predictable namespaces without hardcoding source checkout paths.
4. Needed under the trajectory model:
   Yes, but as host layout/resource namespace rather than scattered path
   helpers.
5. Problem with old design:
   `ProjectLayout` combined catalog, skills, prompts, artifacts, and
   observability paths. `artifact_files` was file-system metadata logic instead
   of a typed artifact resource.
6. Best new abstraction candidate:

   ```python
   class NamespaceLayout(Protocol):
       def resource_root(self, namespace: str, *, session_id: str | None = None) -> Path: ...

   class ArtifactStore(Protocol):
       async def put(self, artifact: Artifact) -> ResourceRef: ...
       async def list(self, session_id: str) -> Sequence[ArtifactMeta]: ...
   ```
7. Boundary with other abstractions:
   Layout maps namespaces to storage. ResourceWriter mutates resources.
   ArtifactStore is a typed view over resource namespace `artifact`.
8. Recommendation:
   Rewrite. Safe to delete old path helper only after replacement layout exists.
9. Migration steps:
   Define namespaces; move artifact metadata into ArtifactStore; stop exposing
   project-wide layout as atom ABI unless an atom needs it.
10. Questions to confirm:
   Are artifacts part of minimal SDK, or optional atom/package functionality?

## 11. PromptTemplate / Skill / Command Shared Atom ABI

1. Abstract name: PromptTemplate, Skill, Command shared atom ABI.
2. Current related files:
   `src/agentm/core/abi/prompt_template.py`,
   `src/agentm/core/abi/skill.py`,
   `src/agentm/core/abi/command.py`,
   related builtin atoms when present.
3. Root requirement / invariant:
   Multiple atoms need to share records and services without importing each
   other.
4. Needed under the trajectory model:
   Sometimes. These are agent authoring conveniences, not trajectory runtime
   substrate.
5. Problem with old design:
   They lived in the same core ABI surface as runtime invariants. That made
   optional authoring features look like mandatory substrate.
6. Best new abstraction candidate:
   Introduce an atom-shared ABI layer or capability package:

   ```text
   core/abi       -> runtime contracts every session may need
   extensions/abi -> optional atom capability contracts
   ```
7. Boundary with other abstractions:
   Runtime ABI is for session, trajectory, events, tools, resources,
   operations. Atom capability ABI is for prompt templates, skills, commands,
   slash parsing.
8. Recommendation:
   Move/downscope. Do not blindly delete if builtin atoms still consume them.
9. Migration steps:
   Inventory consuming atoms; move shared dataclasses to an atom capability
   module; keep service keys typed; remove from top-level runtime ABI.
10. Questions to confirm:
   Should `agentm` minimal SDK ship prompt/skill/command atoms, or should they
   be contrib/optional?

## 12. ProviderResolver / Provider Registry / Provider Identity

1. Abstract name: ProviderResolver / provider registry.
2. Current related files:
   `src/agentm/core/abi/provider.py`,
   `src/agentm/core/runtime/session.py`,
   `src/agentm/core/runtime/session_factory.py`,
   `src/agentm/extensions/builtin/llm_openai.py`,
   `src/agentm/extensions/builtin/llm_anthropic.py`.
3. Root requirement / invariant:
   A session must have exactly one active provider/model stream at run time,
   while provider atoms remain replaceable.
4. Needed under the trajectory model:
   Yes. Provider identity is part of turn metadata and replay/audit context.
5. Problem with old/current design:
   Removing `ProviderManifest` is directionally reasonable because provider
   atoms already use `ExtensionManifest`. However, provider choice, provider
   identity, and provider config provenance still need a first-class resolved
   record.
6. Best new abstraction candidate:

   ```python
   @dataclass(frozen=True)
   class ActiveProvider:
       name: str
       model: Model
       config_ref: str | None
       source_atom: str | None
   ```
7. Boundary with other abstractions:
   Provider atom registers `ProviderConfig`. Resolver chooses active provider.
   Catalog/config resolver records provider source/provenance. TurnMeta records
   model id.
8. Recommendation:
   Keep current simplification but add active provider provenance before
   declaring complete.
9. Migration steps:
   Record provider source extension on registration; stamp active provider into
   SessionMeta and TurnMeta; keep resolver tree-scoped.
10. Questions to confirm:
   Can provider change mid-session, or should it be frozen once the first turn
   commits?

## 13. ServiceRegistry Scope / Atom Priority / Dependency Ordering

1. Abstract name: Service scope and atom ordering.
2. Current related files:
   `src/agentm/core/abi/services.py`,
   `src/agentm/core/abi/manifest.py`,
   `src/agentm/core/runtime/session_factory.py`,
   builtin atom manifests.
3. Root requirement / invariant:
   Atom composition must be deterministic. Services must not accidentally leak
   session-local mutable state into child sessions.
4. Needed under the trajectory model:
   Yes. Child sessions and forked branches need precise inheritance.
5. Problem with old/current design:
   `scope="session" | "tree"` is a start but incomplete. There is no
   `process`, `host`, or resource namespace scope. `priority` is numeric bands
   without explicit install phase semantics. `requires` names atoms, not
   capabilities.
6. Best new abstraction candidate:

   ```python
   ServiceScope = Literal["session", "tree", "process", "host", "resource"]

   @dataclass(frozen=True)
   class AtomRequirement:
       capability: str
       optional: bool = False
   ```
7. Boundary with other abstractions:
   Manifest declares identity, provided capabilities, requirements, and default
   install phase. SessionFactory resolves ordering. ServiceRegistry only stores
   actual instances and scopes.
8. Recommendation:
   Keep/rewrite. Do not expand ad hoc service names without typed roles.
9. Migration steps:
   Define capability registry; split numeric priority into named phases; add
   collision policy; make child inheritance explicit by scope.
10. Questions to confirm:
   Should atom `requires` depend on atom names or provided capabilities?

## 14. Presenter / Rendering Surface

1. Abstract name: Presenter / rendering surface.
2. Current related files:
   `src/agentm/core/abi/presenter.py`,
   `src/agentm/core/lib/render.py`,
   README/API examples.
3. Root requirement / invariant:
   Hosts need a stable way to render messages, tool results, usage, and state
   without duplicating message traversal logic.
4. Needed under the trajectory model:
   Useful, but not substrate-critical.
5. Problem with old design:
   Glyphs and rendering helpers in core blur presentation with SDK substrate.
6. Best new abstraction candidate:
   Move to host/presenter utility package. Keep message dataclasses as the
   only core renderable shape.
7. Boundary with other abstractions:
   Core emits typed messages/events. Presenters render them.
8. Recommendation:
   Safe to move out of core SDK after presenter package exists. Do not block
   core refactor.
9. Migration steps:
   Remove top-level ABI export; create presenter utility module/package; keep
   compatibility only if needed by current hosts.
10. Questions to confirm:
   Is any terminal/gateway presenter still in this package scope?

## 15. Trajectory Serialization / Message Codec

1. Abstract name: Trajectory serialization.
2. Current related files:
   `src/agentm/core/lib/message_codec.py`,
   `src/agentm/core/abi/codec.py`,
   `src/agentm/core/runtime/stores/jsonl.py`,
   `src/agentm/core/abi/messages.py`.
3. Root requirement / invariant:
   Persisted trajectories must round-trip without losing typed message,
   trigger, tool result, image, usage, or outcome information.
4. Needed under the trajectory model:
   Yes.
5. Problem with old design:
   Separate `message_codec.py` and `codec.py` create duplicate serialization
   authority. Deleting one is fine only if `codec.py` fully owns all persisted
   shapes and compatibility policy.
6. Best new abstraction candidate:
   One `TrajectoryCodec` with explicit versioning:

   ```python
   class TrajectoryCodec(Protocol):
       version: str
       def serialize_turn(self, turn: Turn) -> Mapping[str, object]: ...
       def deserialize_turn(self, data: Mapping[str, object]) -> Turn: ...
   ```
7. Boundary with other abstractions:
   Store owns persistence. Codec owns bytes/schema. Observability owns event
   export, not store format.
8. Recommendation:
   Safe to delete old `message_codec.py` only if coverage proves current codec
   handles every persisted field.
9. Migration steps:
   Audit store tests; define codec version; ensure triggers/outcomes round trip.
10. Questions to confirm:
   Is backward migration from older JSONL required, or explicitly out of scope?

## 16. Prompt/Skill Parsing Utility

1. Abstract name: Frontmatter parsing utility.
2. Current related files:
   `src/agentm/core/lib/frontmatter.py`,
   prompt template and skill loader atoms.
3. Root requirement / invariant:
   Markdown-backed extension content often needs metadata plus body parsing.
4. Needed under the trajectory model:
   Not runtime-critical. It is authoring/tooling support.
5. Problem with old design:
   A parser wrapper in core exists mainly because prompt/skill systems lived
   close to core.
6. Best new abstraction candidate:
   Move parser utility next to prompt/skill capability package.
7. Boundary with other abstractions:
   No runtime session boundary. It is content loading support.
8. Recommendation:
   Safe to move/delete from core once prompt/skill capability is moved.
9. Migration steps:
   Find consuming atoms; colocate parser with them; keep no core ABI export.
10. Questions to confirm:
   Should SDK avoid `python-frontmatter` dependency entirely?

## 17. Child Session Visibility / Wire Forwarding

1. Abstract name: Child session visibility / wire forwarding.
2. Current related files:
   `src/agentm/core/lib/child_wire.py`,
   gateway/wire-driver related atoms when present,
   `src/agentm/core/runtime/session.py`.
3. Root requirement / invariant:
   Child sessions are full sessions. Hosts may need to observe, route input to,
   and render their trajectories.
4. Needed under the trajectory model:
   Yes for gateway/multi-agent hosts; optional for minimal embedded SDK.
5. Problem with old design:
   `child_wire.py` is a gateway-specific helper in core lib. It forwards
   through a service name instead of a typed child-session registry.
6. Best new abstraction candidate:

   ```python
   class SessionRegistry(Protocol):
       def register(self, session: SpawnedSession) -> None: ...
       def get(self, session_id: str) -> SpawnedSession | None: ...
       def children(self, session_id: str) -> Sequence[str]: ...
   ```
7. Boundary with other abstractions:
   Runtime owns session graph. Gateway owns wire routing/rendering. Registry
   bridges them.
8. Recommendation:
   Move gateway helper out of core; keep typed session registry if gateway
   remains in package.
9. Migration steps:
   Replace child wire forwarding service with session registry service; keep
   graph as data model.
10. Questions to confirm:
   Is gateway part of minimal SDK package, or an optional host package?

## 18. Observability Export Mapping

1. Abstract name: Observability event export mapping.
2. Current related files:
   `src/agentm/core/observability/event_otel.py`,
   `src/agentm/core/observability/otel_export.py`,
   `src/agentm/core/observability/otlp.py`,
   `src/agentm/extensions/builtin/observability.py`,
   `src/agentm/core/abi/events.py`.
3. Root requirement / invariant:
   Runtime events and turns must produce consistent observability spans/logs
   without every atom knowing exporter details.
4. Needed under the trajectory model:
   Yes.
5. Problem with old/current design:
   Event-to-OTel mapping can become a second trajectory model if it carries
   fields not grounded in `Turn`, `SessionMeta`, or typed events.
6. Best new abstraction candidate:
   Observability should be a projection of trajectory/events:

   ```python
   class ObservabilityProjection(Protocol):
       def on_event(self, event: Event, context: SessionContext) -> None: ...
       def on_turn(self, turn: Turn) -> None: ...
   ```
7. Boundary with other abstractions:
   Driver commits turns. Event bus emits lifecycle/tool/provider events.
   Observability projects them to OTel/local/remote sinks.
8. Recommendation:
   Keep, but audit every exported field against trajectory/event source.
9. Migration steps:
   Build coverage table from Event classes to OTel records; remove duplicate
   session identity paths; route query through `TrajectoryQueryStore`.
10. Questions to confirm:
   Should observability atom be a default minimal SDK atom, or host opt-in?

## 19. Package Surface / Minimal SDK Boundary

1. Abstract name: Minimal SDK package surface.
2. Current related files:
   `pyproject.toml`,
   `src/agentm/__init__.py`,
   `README.md`,
   `uv.lock`,
   builtin atoms.
3. Root requirement / invariant:
   The installed package should provide necessary SDK mechanisms while keeping
   optional hosts/backends/contrib out of the core surface.
4. Needed under the trajectory model:
   Yes.
5. Problem with old/current design:
   Removing dependencies and CLI surfaces can be correct, but only after
   capability boundaries are assigned. Otherwise core abstractions disappear
   together with optional implementations.
6. Best new abstraction candidate:
   Split by capability:
   - `agentm.core`: runtime, trajectory, ABI, stores.
   - `agentm.ext`: builtin atoms that are part of minimal SDK.
   - Optional extras: gateway, observability-remote, sandbox/agent-env,
     prompt/skill authoring, trace CLI.
7. Boundary with other abstractions:
   Package surface decides distribution. It must not decide whether an
   invariant exists.
8. Recommendation:
   Continue simplifying package dependencies, but only after abstraction
   inventory decisions.
9. Migration steps:
   Mark each abstraction as core, builtin, optional extra, or contrib; then
   update pyproject.
10. Questions to confirm:
   Which builtin atoms are mandatory in minimal SDK: observability,
   operations, file tools, bash, system prompt, retry policy?

## Must Pause Or Recover Before More Deletion

- Recover/redesign `Lifecycle` before accepting deletion.
- Recover/redesign `Catalog` before accepting deletion.
- Recover/redesign `ToolExecutor` isolation before accepting deletion of
  execution-domain capability.
- Recover/rewrite `Compaction` around `Turn` projection before accepting
  deletion.
- Recover a local `Operations` backend or require explicit host injection.
- Recover environment/sandbox backend abstraction before deleting agent-env
  implementations.
- Recover query abstraction before deleting `TraceReader`/ClickHouse support.
- Recover config resolver semantics before deleting atom/user config helpers.

## Likely Safe To Move Or Delete After Replacement

- `frontmatter.py`: safe to move with prompt/skill capability.
- `presenter.py` and `render.py`: safe to move to presenter package if no
  core host depends on them.
- `child_wire.py`: safe to move to gateway package after typed session
  registry exists.
- `message_codec.py`: safe to delete only after `core/abi/codec.py` is the
  single tested trajectory codec.
- Empty `__init__.py` files under deleted backend packages are safe only if
  the backend package is removed or replaced.

## Recommended Execution Order

1. Stop further deletion and freeze this inventory as the design checklist.
2. Restore or reintroduce minimal stubs for incorrectly deleted core
   abstractions: Lifecycle, Catalog, ToolExecutor capability, Compaction query
   types, Trace query seam, Config resolver.
3. Redesign the two highest-risk boundaries first:
   Lifecycle as typed effect/world lifecycle, and Catalog as versioned
   composition identity.
4. Redesign ResourceWriter and Operations around the Lifecycle decisions.
5. Redesign ServiceRegistry scopes and atom install requirements so the new
   ports have clear session/tree/process/host/resource scope.
6. Re-home optional authoring/presenter/gateway capabilities out of core ABI.
7. Only then delete obsolete implementations and update package dependencies.
8. Run adversarial review against each boundary before implementation merge.

## Open Design Questions

1. Is catalog identity mandatory in every SDK session or only in auditable
   sessions?
2. Should lifecycle restore failure fail session creation, create a degraded
   session, or ask host policy?
3. Should ResourceWriter include reads or should read/write authority split?
4. Should sandbox/environment backend ship in core package or optional extra?
5. Should provider config be frozen after first committed turn?
6. Should compaction summaries be trajectory records, context projection
   artifacts, or both?
7. Should atom dependencies target atom names or provided capabilities?
8. Which builtin atoms define the minimal SDK baseline?

## Implementation Decision Log

### TrajectoryQueryStore local adapter

- Status: implemented as `TrajectoryStoreQueryAdapter` in
  `src/agentm/core/runtime/stores/query.py`.
- Boundary: the adapter is a read-side projection over `TrajectoryStore` only.
  It provides local `sessions()` and `turns()` queries for in-memory and JSONL
  stores. It deliberately returns empty `events()` and `spans()` views because
  those records belong to observability exporters/backends, not trajectory
  persistence.
- Runtime wiring: `create_session(..., store=...)` registers the adapter under
  `TRAJECTORY_QUERY_STORE_SERVICE` with `scope="resource"` unless the host has
  already provided a query store. This lets remote/ClickHouse adapters replace
  the local view without changing atom-facing service names.
- Remaining work: implement optional observability-backed query adapters for
  event/span rows and decide whether remote trace querying ships as an SDK extra
  or an external package.

### ConfigResolver provenance persistence

- Status: partially implemented for resolved session specs.
- Boundary: `SessionSpecResolver` remains host-owned. The runtime does not
  search home directories, contrib trees, or env config by itself; it persists
  evidence for the already resolved composition.
- Runtime wiring: `session_meta_config()` now records
  `resolved_spec_digest` and `resolved_spec_provenance_json` when a
  `ResolvedSessionSpec` exists. Root sessions created through
  `create_from_config()` persist that evidence in `SessionMeta.config`; spawned
  and forked children preserve it through inherited resolved-spec services.
- Remaining work: implement concrete host resolvers for CLI/user/env/provider
  precedence; connect the resolved spec digest to Catalog active-set identity;
  decide how much provenance should be indexed by query backends versus stored
  only as JSON metadata.

### TrajectoryNodeStore message-tree projection

- Status: implemented as a core ABI Protocol in
  `src/agentm/core/abi/store.py`, with node/cache dataclasses in
  `src/agentm/core/abi/trajectory.py` and an in-memory reference
  implementation in `src/agentm/core/runtime/trajectory_nodes.py`.
- Boundary: `TrajectoryStore` remains the Turn commit log. `TrajectoryNodeStore`
  is the message-level projection needed for fork, resume, sidechain,
  compact-boundary, snip/rewind, content replacement, and prompt-cache state.
  A backend may implement both ports with one physical database, but atoms and
  SDK hosts depend on the two Protocols separately.
- Portable storage contract: all stores expose stable `id`, `session_id`,
  `root_session_id`, `parent_session_id`, `seq`, `parent_id`,
  `logical_parent_id`, `turn_id`, `turn_index`, `agent_id`, `is_sidechain`,
  `kind`, `role`, and `timestamp` fields. SQL backends map these to indexed
  columns; ClickHouse maps them to partition/order/skip-index fields; JSONL
  backends can scan or maintain sidecar indexes while preserving the same
  query semantics.
- Runtime wiring: the driver projects committed turns into nodes when a
  `trajectory_node_store` is registered. Projection failures are diagnostic
  errors, not turn-commit failures, because the Turn store remains the durable
  commit boundary.
- Remaining work: implement concrete Postgres/ClickHouse adapters, decide
  whether content-replacement entries should be materialized as standalone
  nodes in every backend, and add a provider-facing prompt-cache policy atom
  that consumes `PromptCacheState`.

### Catalog composition identity

- Status: implemented as a default in-memory composition catalog plus a
  content-addressed versioned resource store.
- Boundary: Catalog records SDK composition identity, not workspace/user
  resource mutation. `ResourceWriter` remains the write boundary for mutable
  resources; `VersionedResourceStore` stores immutable SDK inputs such as atom
  source, manifest metadata, and normalized atom config.
- Runtime wiring: `create_session()` installs default
  `VERSIONED_RESOURCE_STORE_SERVICE` and `ATOM_CATALOG_SERVICE` unless the host
  provides replacements. Before recording the active set, each resolved atom is
  frozen as `atom:<name>` with media type
  `application/vnd.agentm.atom-identity+json`; the resulting `ResourceVersion`
  is stored on the `AtomActivation`. The active-set fingerprint is registered
  under `ACTIVE_SET_FINGERPRINT_SERVICE` and persisted into `SessionMeta.config`.
- Remaining work: add durable/remote catalog backends, update existing store
  metadata when resuming a preexisting session, and decide whether provider,
  scenario spec, and generated prompt sources should be frozen as first-class
  catalog resources.

### Core ABI recovery stubs

- Status: minimal ports restored for abstractions that should not be deleted
  before replacement exists.
- Boundary: `Lifecycle` now exposes turn-scoped `EffectScope` plus a narrower
  `EnvironmentSnapshotter`; `Operations` exposes `EnvironmentRef` and
  `EnvironmentOperations` around `BashOperations`; `ContextProjection` is a
  projection protocol over immutable `Turn`; `ToolExecutor` again exposes a
  requirements-provider protocol.
- Runtime wiring: these ports are importable from the ABI surface and service
  role constants are present, but most runtime behavior remains intentionally
  narrow. This preserves the abstraction anchors without restoring the old
  generic lifecycle-hook or branch-entry compaction models.
- Remaining work: connect environment snapshot/fork/restore into
  `Session.fork()` and resume paths, route `tool_bash` through an
  `EnvironmentOperations` service while preserving existing bash service
  compatibility, implement `ContextProjection` in context construction, and
  enforce tool execution requirements inside the default `ToolExecutor`.
