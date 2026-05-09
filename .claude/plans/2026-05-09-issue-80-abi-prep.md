# Issue 80 - ABI prep for harness/session.py thinning

Date: 2026-05-09
Issue: #80

## Scope

Pure-additive ABI changes only. This plan deliberately does not update the
current harness call sites that still poke private fields or special-case the
git-backed writer; those migrations belong to #85.

## Decisions

### B7 - AgentLoop provider replacement

Add `AgentLoop.set_stream_fn(fn: StreamFn) -> None` in `core/abi/loop.py`.
This gives `harness/session.py` a public, typed provider-swap accessor for #85
and makes direct `_stream_fn` mutation unsupported.

### B12 - ResourceWriter rollback

Add synchronous `ResourceWriter.restore(path: Path, version: str) -> None` in
`core/abi/resource.py`, matching the existing synchronous `classify`/`batch`
side of the protocol rather than making a git subprocess rollback async-only.
`GitBackedResourceWriter.restore` delegates to the existing git helper and
advisory/unversioned mode raises `NotImplementedError("git rollback requires a
versioned ResourceWriter")`.

### B18 - Operations replaceability

Chosen path: **Option A (document constitution-only)**.

Rationale: there is no concrete atom-level use case for replacing the entire
Operations bundle at runtime. The current architecture already supports the
important environment substitution use case by injecting an `Operations` bundle
when constructing the session. Adding `ExtensionAPI.register_operations` now
would create a stronger runtime-replacement promise without a caller or
fail-stop test to defend it.

Implication: Operations remain a pluggability axis, but a constitution-only one
in v0: harness/session construction selects the bundle; atoms consume it via
`api.get_operations()` and cannot replace it at runtime.

## Validation plan

- `uv run ruff check src/`
- `uv run mypy src/`
- `uv run pytest --tb=short`
- `uv run python -c "from agentm.extensions.validate import validate_builtin; ..."`
