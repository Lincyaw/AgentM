# Issue 94 — AI Provider Descriptor Registry

## Change

Refactor `src/agentm/ai/**` so provider ids, API names, aliases, and environment key precedence are owned by `ProviderDescriptor` records instead of parallel hand-maintained lists.

## Boundary

- Layer touched: provider boundary (`agentm.ai`), below harness and outside atoms.
- Pluggability axis: LLM stream/provider selection metadata.
- Atom/core impact: none; no ExtensionAPI or atom contract change.

## Verification

- Descriptor-backed env key resolution accepts in-memory mappings.
- Runtime provider registration uses one validating wrapper with shared API canonicalization.
- Adding a fake provider is covered by registering a descriptor and factory in a local registry.
