# Design: Search Tool Surface

**Status**: CURRENT
**Created**: 2026-05-01
**Last Updated**: 2026-06-29

## Overview

AgentM does not expose default `grep`, `find`, `ls`, or `glob` tool atoms.
Local workspace search is delegated to the existing `bash` tool and mature CLI
programs such as `rg`, `rg --files`, `find`, `fd`, and `git ls-files`.

The default file tool atom is intentionally limited to guarded file I/O:
`read`, `write`, and `edit`.

## Rationale

Search wrappers are only worth exposing when they add load-bearing semantics
that the command line cannot provide. Thin wrappers around `find`, `grep`, or
`rg` create a second, less familiar search language and can hide bugs behind
plausible empty results. The observed `glob` implementation did exactly that:
it delegated to `find -name`, so path patterns like `tests/**/*.py` returned
`No files found` even though matching files existed.

The shell already has the correct, inspectable behavior:

- `rg --files -g 'tests/**/*.py'`
- `rg -n 'pattern' path`
- `find path -name '*.py' -print`
- `git ls-files '*pattern*'`

Keeping those calls inside `bash` preserves normal CLI semantics and makes
failures easier to reproduce outside AgentM.

## Policy

1. Do not add default search tools that only wrap an existing CLI.
2. Prefer removing a redundant tool over repairing it if `bash` already gives
   an equivalent or clearer interface.
3. Keep tools when they enforce AgentM-specific safety or state:
   `read` tracks partial reads and max-size gates, while `write` and `edit`
   route mutations through `ResourceWriter`.
4. Add a dedicated search atom only for a real non-CLI requirement, such as a
   remote or in-memory `FileOperations` backend where `bash` is unavailable, or
   a safety/audit contract that cannot be expressed by shell policy.
5. If a future search atom is needed, it must be implemented through
   `FileOperations`, return explicit incomplete-result warnings, and avoid
   false-empty outputs on unreadable paths or malformed ignore metadata.

## Current Surface

`src/agentm/extensions/builtin/file_tools.py` registers:

- `read`
- `write`
- `edit`

`tool_bash` remains the search surface for local scenarios. Prompts and
workflows should instruct agents to use `bash` with `rg`/`find` instead of
calling typed `grep`/`glob` tools.

## Related Concepts

- [extension-as-scenario.md](extension-as-scenario.md) — atom contract and
  scenario composition
- [pluggable-architecture.md](pluggable-architecture.md) — `FileOperations`,
  `BashOperations`, and `ResourceWriter` seams
- [artifact-system.md](artifact-system.md) — artifact search remains a
  domain-specific surface, not a wrapper around workspace CLI search
