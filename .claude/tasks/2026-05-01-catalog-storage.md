# Task: catalog-storage — Hashing, freeze_current, directory layout contract

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Design**: [evolution-substrate](../designs/evolution-substrate.md) §3, §5.2
**Assignee**: implementer
**Wave**: 2 (parallel)
**Size**: M
**Depends on**: [core-manifest](2026-05-01-core-manifest.md), [manifest-schema](2026-05-01-manifest-schema.md)

## Objective

Establish the on-disk layout for `.agentm/catalog/` and the constitution-layer modules that read/write it. Lock the layout as a contract so the indexer (`indexer-mvp`) and the read API (`tool-catalog-atom`) cannot drift. **No autonomy code touches anything in this task.**

The deliverables are intentionally narrow: hashing, freezing one atom version, computing the active-set fingerprint. Querying lives in `tool-catalog-atom`; aggregation lives in `indexer-mvp`.

## Inputs to read

- `evolution-substrate.md` §3 (full directory layout — verbatim contract)
- `evolution-substrate.md` §5.2 (idempotence requirement)
- `self-modifiable-architecture.md` §5.2 (when freeze is called from reload)
- `src/agentm/extensions/discover.py` (for `discover_builtin` shape — used by `compute_active_set_fingerprint`)
- `src/agentm/extensions/__init__.py` (the post-`manifest-schema` `ExtensionManifest`)

## Outputs

### New files

| Path | Purpose |
|---|---|
| `src/agentm/core/catalog/_layout.py` | Path constants. `CATALOG_ROOT = Path(".agentm/catalog")`, plus `atoms_dir(name)`, `atom_version_dir(name, hash)`, `atom_runs_dir(name, hash)`, `atom_metrics_path(name, hash)`, `atom_manifest_path(name, hash)`, `atom_source_path(name, hash)`, `atom_current_symlink(name)`, `core_dir(hash)`. **Single source of truth for paths.** |
| `src/agentm/core/catalog/hashing.py` | `compute_atom_hash(source: str) -> str` (sha256, hex truncated to 12 chars per design examples like `e5f6...`); `compute_active_set_fingerprint(loaded: dict[str, str], scenario: str | None, core_hash: str | None) -> dict` returning a dict in the shape from evolution-substrate §4 ready for JSON serialization. |
| `src/agentm/core/catalog/freeze.py` | `freeze_current(name: str, source: str, manifest: ExtensionManifest, *, root: Path | None = None) -> str`: writes `{atom_version_dir}/source.py` and `{atom_version_dir}/manifest.yaml`; updates `current` symlink atomically (write to temp name, `os.replace`); returns the content hash. **Idempotent**: returns existing hash without rewriting if the version dir is already populated. |
| `src/agentm/core/catalog/__init__.py` | Re-export public surface: `compute_atom_hash`, `compute_active_set_fingerprint`, `freeze_current`, `is_constitution_path`, `list_atoms` (the last is a thin enumerate over `_layout.atoms_dir(name).iterdir()`). |
| `tests/unit/core/catalog/test_hashing.py` | |
| `tests/unit/core/catalog/test_freeze.py` | |

### Modified files

None — this task is purely additive.

## Concrete shape — `compute_active_set_fingerprint`

Returns a dict shaped per evolution-substrate §4 — ready to embed under `attributes.fingerprint` of an OTel record:

```python
{
    "core": "core@<hash>" or None,
    "scenario": "<name>@<hash>" or None,
    "atoms": {
        "<atom_name>": "<atom_name>@<hash>",
        ...
    },
}
```

For the MVP, `core_hash` is supplied by callers (initially `None`; the harness can pass a build commit hash later); `scenario` is the recipe name + content hash if known. Atoms are enumerated from `discover_builtin()`'s output, hashing the *current* `inspect.getsource(module)`.

The function takes already-resolved `loaded: dict[name -> hash]` so the caller decides which atoms to include — observability passes the actually-loaded set (a subset of discovery if a recipe filters), not all discovered atoms. This matches §4.1 ("every loaded atom").

## Concrete shape — `freeze_current` directory layout

After a freeze of `tool_read` whose source hashes to `e5f6abc12345`:

```
.agentm/catalog/
└── atoms/
    └── tool_read/
        ├── current → e5f6abc12345
        └── e5f6abc12345/
            ├── source.py        # frozen verbatim
            └── manifest.yaml    # serialized ExtensionManifest
                                 # + parent_hash, author, authored_at
                                 # (parent_hash defaults None for the genesis;
                                 #  author defaults "human"; reload caller can override)
```

Symlink update is atomic: create `current.tmp` → `os.replace(current.tmp, current)`. On filesystems without symlink support (e.g. Windows non-admin), fall back to a `current` text file containing the hash; `tool_catalog` reads either form. This fallback is justified because catalog is constitution-layer and we cannot fail an entire dev install over a Windows POSIX gap.

## Test cases

### `test_hashing.py`

| Test | Asserts |
|---|---|
| `test_hash_is_deterministic` | Same source → same hash twice |
| `test_hash_distinguishes_whitespace_difference` | Two sources differing by one space have different hashes |
| `test_hash_length_matches_design_examples` | 12 hex chars |
| `test_active_set_fingerprint_includes_all_loaded_atoms` (M2 contributor) | Given `{tool_read: h1, tool_bash: h2}`, the result's `atoms` map has exactly two entries with `<name>@<hash>` form |
| `test_active_set_fingerprint_optional_core_and_scenario` | `None` core/scenario serializes as `None` (not missing key) |

### `test_freeze.py`

| Test | Asserts |
|---|---|
| `test_freeze_writes_source_and_manifest` | After `freeze_current("tool_read", source, manifest, root=tmp)`, both files exist with expected content |
| `test_M1_idempotent_no_rewrite_when_hash_exists` | Freezing the same source twice does not modify mtimes (the second call returns early; verify via `os.stat`) |
| `test_freeze_updates_current_symlink` | `current` resolves to the latest hash; the previous hash dir still exists (no deletion) |
| `test_freeze_returns_content_hash` | Return value equals `compute_atom_hash(source)` |
| `test_freeze_creates_runs_dir` | `runs/` exists empty after freeze (so the indexer can `os.symlink` into it without `FileNotFoundError`) |

## Acceptance Conditions

- [ ] `uv run pytest tests/unit/core/catalog/ -v` all green
- [ ] `uv run ruff check src/agentm/core/catalog/` clean
- [ ] `uv run mypy src/agentm/core/catalog/` clean
- [ ] No autonomy-layer file modified
- [ ] Public surface from `agentm.core.catalog` is exactly: `compute_atom_hash`, `compute_active_set_fingerprint`, `freeze_current`, `is_constitution_path`, `list_atoms`. (Verified by an `__all__` test.)
- [ ] Layer purity: `core/catalog/*.py` imports only stdlib + `yaml` + `agentm.extensions` (for `ExtensionManifest` type) + `agentm.core.catalog.*` siblings. **No `agentm.harness.*`, no `agentm.extensions.builtin.*`.**

## Acceptance scenarios covered

- **M1** (idempotence) — `test_M1_idempotent_no_rewrite_when_hash_exists`
- Layout contract for **E5** is locked here (test in `indexer-mvp` will exercise it)

## Notes

- **Layout is a contract**: any path computation outside `_layout.py` is a bug. The `transactional-reload` task imports `_layout` and `freeze.freeze_current`; it does not compute paths itself.
- **`list_atoms` is the read primitive used by `tool_catalog.list_versions`** — but `list_atoms` returns plain dicts/structs from `agentm.core.catalog`, NOT from `agentm.extensions.builtin.tool_catalog`. The autonomy atom calls into the constitution module.
- **No `affects` consumption yet** — the manifest is serialized verbatim; the indexer ignores `affects.primary` until Phase 2.
- **`parent_hash` is recorded but defaults `None`** in MVP — the reload mechanism's `freeze_current` caller (in `transactional-reload`) supplies the prior hash explicitly.
- **No scenario-version directories** in MVP (deferred per plan §1.1). The fingerprint records the scenario hash, but `.agentm/catalog/scenarios/` is empty in MVP.
```

============ END FILE ============