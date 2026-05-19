# AgentM Glossary

Terms with precise meaning inside this project. Implementation details belong elsewhere — this file is a glossary, not a spec.

## Entry point

The CLI (`agentm`, `agentm-gateway`, `agentm-worker`, `agentm-feishu`, `agentm-terminal`) is the supported entry. `AgentSession.create` is the substrate-level constructor those CLIs call. Notebook / embedder use is not a current priority — there is no `agentm.Session` façade.

Power-users override defaults by passing fields into `AgentSessionConfig`.

## Substrate axiom (aspirational, not enforced)

The design doc `pluggable-architecture.md` says: *"substrate provides registration hooks and asserts at freeze time that the required services have been registered; it never instantiates a default."*

Today the code violates this — `build_extension_api_scope` and `session_factory` auto-instantiate `GitBackedResourceWriter`, `default_project_layout`, `default_catalog_service`, `InMemorySessionManager`, `InMemoryResourceLoader`, and `LastRegisteredWins`. Closing this gap was scoped, planned (see the grilling-session record in chat history), and then deliberately deferred: the cleanup is large, the user-visible benefit is zero, and the partially-done Phase 1 explored on branch `worktree-agent-phase1-retry` showed the constitution-path machinery (`is_constitution_path` / `load_core_manifest`) blocks any atom-based `GitBackedResourceWriter` until those helpers move out of `_internal/`.

If this work is revived, the load-bearing prerequisites are:
1. Move constitution-path helpers from `core/_internal/catalog/manifest.py` to `core/lib/` or `core/abi/`.
2. Decide JSONL vs in-memory as the floor `SessionManager` (CLI needs JSONL persistence; the in-memory default in `worktree-agent-phase1-retry` was a Phase 1 stub).
3. Promote `SessionManager` / `ResourceLoader` / `ProjectLayout` from `core.runtime` concrete classes to `core.abi` Protocols so atoms can satisfy them structurally.
4. Phase the substrate auto-instantiation removal so the CLI doesn't break mid-migration.

## Substrate-only (kernel singletons)

Two pieces are not pluggable by design and have no atom-replacement axis: **CatalogService** (`.agentm/catalog/` is constitution-layer) and **ProviderResolver** ("last-registered wins" is universal). The boundary axiom does not bind these.
