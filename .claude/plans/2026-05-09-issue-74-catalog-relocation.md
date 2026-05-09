# Issue #74 — relocate catalog filesystem/discovery operations to harness

Date: 2026-05-09
Branch: `worktree-agent-a0501ec9c59d2ea55`
Issue: https://github.com/Lincyaw/AgentM/issues/74

## Summary

`agentm.core._internal.catalog/` previously straddled two roles: pure
kernel functions (hashing, manifest parsing, browse) and filesystem +
discovery orchestration (freeze, migrate, indexer, layout). The latter
forced the kernel to import `agentm.extensions`, walked the filesystem
on first import via `Path(__file__).parents[5]`, and hardcoded
`.agentm/...` paths inside what is supposed to be a port-shaped layer.

That violated the CLAUDE.md invariant: "core must import in a Jupyter
notebook with no harness, no CLI, no filesystem touched."

## What changed

### Moves (4 files)

- `src/agentm/core/_internal/catalog/_layout.py` →
  `src/agentm/harness/catalog/_layout.py`
- `src/agentm/core/_internal/catalog/freeze.py` →
  `src/agentm/harness/catalog/freeze.py` (now imports
  `agentm.harness.catalog._layout` directly; no lazy import dance)
- `src/agentm/core/_internal/catalog/migrate.py` →
  `src/agentm/harness/catalog/migrate.py`
- `src/agentm/core/_internal/catalog/indexer.py` →
  `src/agentm/harness/catalog/indexer.py` (typer CLI default for
  `--observability` now resolves through `DefaultProjectLayout`)

### New module: `src/agentm/core/abi/project_layout.py`

`ProjectLayout` Protocol with five accessors:
`catalog_root()`, `skills_dirs()`, `artifacts_root(session_id)`,
`prompts_dirs()`, `observability_root()`. No filesystem touch at
construction or import time.

### New module: `src/agentm/harness/catalog/__init__.py`

- `DefaultProjectLayout` dataclass implementing `ProjectLayout` against
  today's `<cwd>/.agentm/...` defaults.
- `default_project_layout(cwd)` factory.
- `list_atoms(*, root=None)` (moved out of the core re-export shim
  because it imports `agentm.extensions.discover`).
- Re-exports `freeze_current`, `index_trace`, `rebuild_catalog`,
  `migrate_catalog_v2`, `IndexerResult`, `source_path_for_hash`.

### Slim core `__init__.py`

`src/agentm/core/_internal/catalog/__init__.py` now re-exports only
pure surface (browse + hashing + manifest predicate). No more lazy
imports inside re-export functions, no more `agentm.extensions` import
at the kernel.

### `core/_internal/catalog/manifest.py` — no filesystem at import

- Dropped `_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]`.
- `_MANIFEST_PATH` defaults to `None`.
- `load_core_manifest()` accepts optional `manifest_path` argument and
  raises `CoreManifestPathUnsetError` when neither path nor seam is
  configured.
- New `configure_manifest_path(path)` helper for harness/CLI startup
  (plus reload), so production code does not touch the private name.
- Test seam (`_MANIFEST_PATH` monkeypatch + `reload_manifest()`) still
  works.

### `core/_internal/catalog/browse.py`

Inlined the `runs_for` path computation (`<root>/.agentm/catalog/atoms/
<name>/<version>/runs`) so it does not depend on `_layout` (which moved
to the harness).

### Harness wiring

- `harness/session.py:create` now configures the manifest path from
  `<cwd>/core-manifest.yaml` if the global is still unset, before
  running migration. Imports `migrate_catalog_v2` and `index_trace`
  from `agentm.harness.catalog`.
- `harness/atom_reloader.py` imports `_layout` from
  `agentm.harness.catalog`.
- `harness/services.py:_DefaultCatalogService` imports `_layout` from
  the harness path; new `default_project_layout(cwd)` factory.
- `harness/extension.py` exposes `get_project_layout()` on
  `ExtensionAPI`; default-built from `cwd` if no override is supplied.

### Tests

- Updated import sites in
  - `tests/unit/core/catalog/test_freeze.py`
  - `tests/unit/core/catalog/test_browse.py`
  - `tests/unit/core/catalog/test_indexer.py`
  - `tests/integration/test_self_mod_mvp.py`
  - `tests/integration/test_browse_and_rollback.py`
- Added `tests/conftest.py` autouse fixture pinning `_MANIFEST_PATH`
  to the worktree's `core-manifest.yaml` before each test, so
  cross-test bleed from a previous test that ran `AgentSession.create`
  in a tmp cwd does not leak into manifest unit tests.

## Verification

- `uv run ruff check src/` — clean.
- `uv run mypy src/` — clean (91 source files).
- `uv run pytest --tb=short` — 91 passed, 14 deselected (UI tests).
- New invariant: in a fresh tmpdir cwd with no `.agentm/` and no
  `core-manifest.yaml`,
  `python -c "import agentm.core; import agentm.core.abi; import agentm.core._internal.catalog"`
  succeeds without filesystem access beyond Python's own import
  mechanism.

## Pragmatic deferrals

The issue lists `core/lib/artifact_files.py:11`, `core/_internal/skills.py:229`,
and `core/_internal/prompt_templates.py:165` as hardcoded `.agentm/...` sites
that should "consult `ProjectLayout` instead." Today these helpers accept
`cwd: str` and inline the same path math the default `ProjectLayout`
returns. Rewiring them to receive a `ProjectLayout` value object
propagates through every call site (services, scenarios, tests).
That breadth is out of scope for issue #74; the port is now defined
and exposed via `ExtensionAPI.get_project_layout()`, which lets a
follow-up CL replace those callers without re-litigating the contract.
