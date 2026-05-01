# Task: core-manifest — Constitution boundary file + parser + path predicate

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Designs**:
- [self-modifiable-architecture](../designs/self-modifiable-architecture.md) §3
- [evolution-substrate](../designs/evolution-substrate.md) §7.5
**Assignee**: implementer
**Wave**: 1 (sequential — pre-req for every other task)
**Size**: M

## Objective

Establish the constitution boundary as a data file at the repo root, a constitution-layer parser that loads it, and `is_constitution_path(path) -> bool` callable from anywhere in the constitution layer. Add `agentm.core.catalog` to the §11 import allow-list so the rest of the plan can deliver the catalog modules without retripping the validator.

This task creates **no new agent-facing behavior**. It is the foundation that the rest of the plan builds on.

## Inputs to read

- `.claude/designs/self-modifiable-architecture.md` §3 (the manifest schema in full)
- `src/agentm/extensions/validate.py` — current `_ALLOWED_PREFIXES` constant
- `src/agentm/extensions/__init__.py` — current `ExtensionManifest`

## Outputs

### New files

| Path | Purpose |
|---|---|
| `core-manifest.yaml` (repo root) | The constitution declaration. Contents follow design §3 verbatim, modulo our `tier_2_atoms` list which must agree with the per-atom `tier=2` declarations the next task will land. Layer-purity: this file lives at the repo root, NOT under `src/agentm/`, because it scopes the whole project. |
| `src/agentm/core/catalog/__init__.py` | Empty for now; this task only re-exports `is_constitution_path` and the parser entrypoint. The other modules land in later tasks. |
| `src/agentm/core/catalog/manifest.py` | Parser + `is_constitution_path(path: str) -> bool` predicate. Caches the parsed YAML at module scope; exposes a `reload_manifest()` for tests. Path matching uses `pathlib.PurePosixPath.match` for the glob patterns in the YAML; supports `**` recursion via stdlib `fnmatch.fnmatchcase` over normalized `posix` paths. |
| `tests/unit/core/catalog/__init__.py` | Empty package marker. |
| `tests/unit/core/catalog/test_manifest.py` | Tests for the parser and predicate. |

### Modified files

| Path | Change |
|---|---|
| `src/agentm/extensions/validate.py` | Add `agentm.core.catalog` to `_ALLOWED_PREFIXES`. (Single-line patch.) |

## Concrete shape — `core-manifest.yaml`

Match design §3 verbatim. The `tier_2_atoms` list is **informational** in the MVP per plan R4: per-atom `tier=2` declarations land in the next task and are the source of truth.

## Concrete shape — `core/catalog/manifest.py`

Public surface:

```python
def load_core_manifest() -> CoreManifest: ...
def reload_manifest() -> CoreManifest: ...           # tests
def is_constitution_path(path: str) -> bool: ...     # the predicate

@dataclass(frozen=True, slots=True)
class CoreManifest:
    version: int
    constitution_paths: tuple[str, ...]              # raw glob patterns
    extension_api_current: int
    extension_api_grace: int
    tier_2_atoms: tuple[str, ...]
```

Path resolution rules for `is_constitution_path`:

- Input is normalized via `Path(path).resolve()` only if it points inside the repo; for relative paths it is treated as repo-root-relative.
- Match is performed in **POSIX form** so the YAML's forward-slash patterns work on any OS.
- A path matches if **any** of the YAML patterns match. `**` semantics follow `pathlib.PurePath.match`/`fnmatch` POSIX expectations (recursive directory traversal).
- The manifest itself (`core-manifest.yaml`) is in the constitution list so a self-modifying agent cannot rewrite the manifest to lift its own constraint — verified by a test.

## Test cases

In `tests/unit/core/catalog/test_manifest.py`:

| Test | Asserts |
|---|---|
| `test_loads_constitution_paths` | Parser surfaces all `constitution.paths` entries verbatim |
| `test_extension_api_current_and_grace_are_ints` | Type contract; default is `current=1, grace=1` |
| `test_kernel_path_is_constitution` | `is_constitution_path("src/agentm/core/kernel/loop.py")` is `True` |
| `test_extension_atom_is_not_constitution` | `is_constitution_path("src/agentm/extensions/builtin/tool_read.py")` is `False` |
| `test_catalog_dir_is_constitution` | `is_constitution_path(".agentm/catalog/atoms/x/y/metrics.jsonl")` is `True` (covers E4) |
| `test_manifest_yaml_is_self_referential` | `is_constitution_path("core-manifest.yaml")` is `True` |
| `test_S10_manifest_change_moves_constitution_boundary` | Write a temp manifest with kernel removed; `reload_manifest()` reflects the change; predicate now returns `False` for the kernel path. (Uses monkeypatch to point the loader at a tmp path; documents the test convention for §S10 here.) |

## Acceptance Conditions

- [ ] `uv run pytest tests/unit/core/catalog/test_manifest.py -v` passes
- [ ] `uv run ruff check src/agentm/core/catalog/ tests/unit/core/catalog/` clean
- [ ] `uv run mypy src/agentm/core/catalog/` clean
- [ ] `uv run pytest tests/unit/extensions/test_extension_contract.py -q` still green (verifies the allow-list change does not break existing atoms)
- [ ] `core-manifest.yaml` literally matches design §3 (paths list, `extension_api`, `reload.tier_2_atoms`)
- [ ] No imports from `agentm.harness.*` in `core/catalog/manifest.py` (constitution layer must not import harness)

## Notes

- **Layer purity**: `core/catalog/manifest.py` lives in the constitution layer. It imports only stdlib (`pathlib`, `fnmatch`, `dataclasses`) + `yaml`. No harness imports.
- **Caching**: parse once on first call; expose `reload_manifest()` for tests that mutate the YAML file. Production code never needs to reload — manifest changes require a process restart by design (it's a constitution-layer artifact).
- **No `tool_catalog` atom yet** — that lands in Wave 3.
- **No `freeze_current` yet** — that lands in `catalog-storage`.
- **No agent-facing API changes yet** — `is_constitution_path` will be exposed via `ExtensionAPI` in `transactional-reload`.
```

============ END FILE ============