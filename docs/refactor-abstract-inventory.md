# AgentM Remaining Optional Capability Backlog

This document tracks only unfinished optional capabilities and the design
questions that must be answered before implementation.

Completed core work is intentionally omitted here. The minimal SDK already has
the core ports and runtime wiring for catalog identity, lifecycle/effects,
operations, tool execution requirements, config provenance, trajectory query,
and context projection.

## Boundary Rule

Do not add optional backends directly into minimal SDK core unless the backend
is required for embedded SDK sessions to run.

Core should define stable ports, invariants, service keys, and default local
behavior. Optional packages should provide durable storage, remote execution,
gateway/presenter integrations, and deployment-specific policy.

## Open Backlog

| Capability | Why it remains open | Discussion needed | Likely home |
| --- | --- | --- | --- |
| Durable catalog backend | Current catalog/resource store is in-memory. Composition identity is captured, but not durable across process restarts unless host persists it. | JSONL vs SQLite vs host DB; where project-local state lives; whether catalog records are indexed by default. | SDK storage extra or host backend. |
| Indexed catalog query | Active-set identity exists, but there is no query model beyond session active-set lookup. | Query predicates: session, atom name, digest, version, scenario, provider, provenance. Whether this belongs with trajectory query or catalog API. | Catalog/query extra. |
| Remote observability query | Local `TrajectoryQueryStore` covers sessions/turns only. Events/spans belong to observability backends. | OTLP files vs ClickHouse vs collector export vs host service. Schema ownership and dependency boundary. | Observability extra. |
| Concrete `TrajectoryNodeStore` backends | The core message-tree/index Protocol exists, but there are no durable Postgres, ClickHouse, or JSONL sidecar implementations yet. | Canonical DDL, migration ownership, ClickHouse partition/order keys, JSONL sidecar index shape, and compatibility guarantees across stores. | SDK storage extra or host backend. |
| Node-store-backed fork/resume | `Session.fork()` and `Session.resume()` still primarily operate on Turn prefixes. The node projection can model sidechain leaves, logical parents, compact boundaries, and cache-identical prefixes, but the public flows do not yet consume it. | Whether fork/resume should require a `TrajectoryNodeStore`; how to resolve current leaf; how logical parent traversal interacts with compact boundaries; whether Turn-prefix fallback remains supported. | Core runtime glue plus storage-backed policy. |
| Concrete `SessionSpecResolver` | Resolver port and provenance persistence exist, but no default host policy for config search/precedence. | Precedence for CLI flags, env vars, user config, project config, scenario config, atom config, and provider profiles. | CLI/presenter package or SDK helper. |
| ResourceWriter read/write authority | Transactional resource mutation exists, but read authority, artifact namespace, and restore semantics still need sharper policy. | Should `ResourceWriter` cover reads, or should reads use a separate `ResourceReader`? Are artifacts just another `ResourceRef` namespace? How do resource mutations restore on fork/resume? | Core ABI decision plus optional writer backends. |
| Sandbox/remote environment backend | Local `EnvironmentOperations` exists. Remote/sandbox/agent-env execution brings deployment dependencies and auth. | Environment identity, cwd mapping, file transfer, cancellation, stdout/stderr logs, snapshot capability, and backend lifecycle. | Environment backend extra. |
| Environment snapshot persistence | `EffectScope.fork_at()` and `restore()` are wired, but concrete snapshot storage is backend-specific. | Restore failure policy: fail session creation, degraded read-only session, or host decision. Snapshot retention and id durability. | Environment backend extra plus host policy. |
| Process/sandbox `ToolExecutor` | Requirements/capabilities are enforced, but only direct execution is built in. | Capability matrix for isolation, filesystem read/write, network, killability, concurrency, and interrupt behavior. Relationship to `EnvironmentOperations`. | Tool execution extra or environment backend. |
| Concrete `ContextProjection` strategies | Projection service is wired, but no real compaction/summarization strategy exists. | Token accounting source; summary artifact storage; `ProjectionReport` persistence; whether summaries are catalog resources. | Optional builtin atom or host service. |
| Prompt-cache/content-replacement policy | `ContentReplacementState` and `PromptCacheState` are typed and persistable, but no context policy currently enforces deterministic tool-result replacement or provider cache identity. | Per-message budget rules, replacement text format, state cloning on fork, reconstruction on resume, and provider adapter contract for cache keys. | Optional builtin atom plus provider adapters. |
| Claude Code message-pattern policy atoms | Core now has synthetic message metadata, permission decisions, queue priorities, and tool orchestration, but concrete attachment/message producers are still policy. | Which patterns become builtin atoms: hook results, task notifications, plan-mode messages, local-command caveats, memory reminders, teammate/channel mailbox, and token/budget nudges. | Optional builtin atoms or presenter/gateway packages. |
| Fork/resume/cache E2E coverage | Existing tests protect the core runtime paths, but there is no end-to-end scenario proving node-store fork/resume plus content-replacement cache stability. | Load-bearing scenario definition; real vs stub provider; whether to test JSONL sidecar, Postgres, or an in-memory node store first. | Integration tests after behavior is declared load-bearing. |
| Presenter/gateway/authoring capabilities | These are workflows around the SDK, not minimal runtime invariants. | Package boundaries for frontmatter, renderers, child wire forwarding, trace CLI, prompt/skill authoring helpers. | Presenter, gateway, or authoring packages. |

## Discussion Queue

1. Which storage backend should be the first durable catalog implementation:
   JSONL, SQLite, or host-provided only?
2. Should catalog query and trajectory query share one query facade, or remain
   separate APIs?
3. Which `TrajectoryNodeStore` backend should be implemented first: JSONL
   sidecar, Postgres, ClickHouse, or host-provided only?
4. What is the canonical portable schema/index contract for trajectory nodes,
   content-replacement state, and prompt-cache state?
5. Should `Session.fork()` and `Session.resume()` require node-store-backed
   leaf reconstruction when a node store is available, or keep Turn-prefix
   semantics as the public default?
6. Should compact boundaries and content-replacement entries always be
   materialized as trajectory nodes, or can some stores keep them as side
   tables/projection state?
7. What is the exact config precedence for `SessionSpecResolver`?
8. Should `ResourceWriter` include reads, or should read/write authority split?
9. Are resource artifacts, workspace files, and sandbox files all `ResourceRef`
   namespaces, or do artifacts need a separate abstraction?
10. Should sandbox/environment backends ship as one optional extra or multiple
   backend-specific packages?
11. What happens when environment restore fails during resume?
12. Should process/sandbox tool execution be implemented as a `ToolExecutor`, an
   `EnvironmentOperations` capability, or a composition of both?
13. Should compaction summaries be trajectory records, context projection
   artifacts, catalog resources, or observability records?
14. How should prompt-cache keys and content-replacement state flow into
   provider adapters without making provider-specific cache policy core?
15. Which Claude Code message-pattern producers are minimal builtin policy and
   which belong to presenter/gateway packages?
16. Should provider config be frozen after the first committed turn?
17. Should atom dependencies target atom names or provided capabilities?
18. Which builtin atoms define the minimal SDK baseline?

## Recommended Order

1. Concrete `TrajectoryNodeStore` backend and schema.
2. Node-store-backed fork/resume flow, including sidechain leaf lookup.
3. Prompt-cache/content-replacement context policy.
4. Fork/resume/cache E2E scenario after the behavior is declared load-bearing.
5. Durable catalog backend.
6. Concrete `SessionSpecResolver` policy.
7. Real `ContextProjection` compaction strategy.
8. Resource read/write authority and restore policy.
9. Sandbox/remote `EnvironmentOperations`.
10. Process/sandbox `ToolExecutor`.
11. Remote observability and indexed catalog query.
12. Claude Code message-pattern policy atoms.
13. Presenter/gateway/authoring package split.
