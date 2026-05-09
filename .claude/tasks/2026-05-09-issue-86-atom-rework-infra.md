# Issue 86 — Atom Rework Infrastructure Prep

## Requirement

Backs `REQ-086-atom-rework-infra` in `project-index.yaml`.

## Scope

- Add pure ExtensionAPI/core surfaces for C-2/C-3 atom rework follow-ups.
- Harden the §11 validator with AST checks for private harness reflection,
  ExtensionAPI mutation, mutable module globals, and dynamic `agentm.*` imports.
- Keep current builtin atoms passing by marking intentional module-level
  constant dictionaries/lists as `Final`; do not migrate atom behavior yet.

## Propagation

- `extension-as-scenario.md`: ExtensionAPI surface now includes observer,
  child-session kwargs, services, and resource-writer access.
- `self-modifiable-architecture.md`: self-modification API list now includes
  the additive prep surfaces used by upcoming atom rework batches.
