# Issue 96: CLI provider and session seams

## Requirement

REQ-096-cli-provider-session-seams implements the presenter-side pluggability seams for provider selection and session state lookup.

## Changes

- Move CLI provider extension-spec construction to `ProviderRegistry.build(...)` using descriptor-owned extension module/default/env metadata.
- Add `SessionStore` / `SessionState` protocols in `agentm.core.abi` and a default `JsonlSessionStore` wrapper around `SessionManager`.
- Share CLI diagnostic, extension-install warning, and `AgentSessionConfig` construction helpers between single-run and Textual flows.
- Add `AgentSessionConfig.with_bus(...)` and use it in Textual instead of cloning through `config.__dict__`.

## Verification

- `tests/unit/test_cli_pluggable_boundaries.py` verifies CLI resume/continue paths with an injected in-memory store.
- `tests/ai/test_provider_registry.py` verifies descriptor-only provider registration can build CLI extension config.
