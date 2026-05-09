# Issue 88 — Builtin Atom Hygiene

## Summary

Implemented the builtin atom hygiene sweep: manifest dependency declarations,
requires-aware load ordering, config-schema defaults for previously hard-coded
atom knobs, and diagnostics for ignored resource responses / unpriced cost
providers.

## Validation

- `uv run ruff check src/`
- `uv run mypy src/`
- `uv run pytest --tb=short`
- `validate_builtin()` returns 0 issues with `core-manifest.yaml` configured.
- `uv run python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`
