# Issue 101 — Tool Catalog Browse/Mutate Split

## Context

`contrib/extensions/tool_catalog.py` combined read-only catalog browsing with
self-modifying tools. Plan mode needs the browse surface without permitting
rollback, install, unload, or reload mutation.

## Changes

- Replace the flat contrib atom with `contrib.extensions.tool_catalog.browse`
  and `contrib.extensions.tool_catalog.mutate`.
- Keep shared path and git-log helpers package-internal and manifest-free.
- Mount only the browse atom in the plan-mode scenario.
- Preserve the existing full tool surface when both atoms are mounted.

## Verification

- Manifest registers are exact for both atoms.
- Plan mode exposes catalog browsing and omits mutating tools.
- Existing browse/rollback and live install/reload integration tests cover
  parity for result shapes and self-modification behavior.
